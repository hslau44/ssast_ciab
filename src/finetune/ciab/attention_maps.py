'''
file: attention_maps.py
author: Harry Coppock
Qs: harry.coppock@imperial.ac.uk
'''
import torch
import torch.nn as nn
import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../../')
from models.ast_models import ASTModel
import dataloader

def load_trained_model(model_path, device):
    args = load_args(model_path)
    args.wandb = False
    audio_model = ASTModel(
            label_dim=args.n_class, 
            fshape=args.fshape, 
            tshape=args.tshape, 
            fstride=args.fstride, 
            tstride=args.tstride,
            input_fdim=args.num_mel_bins,
            input_tdim=args.target_length,
            model_size=args.model_size,
            pretrain_stage=False,
            load_pretrained_mdl_path='/home/ec2-user/SageMaker/jbc-cough-in-a-box/ssast_ciab/src/finetune/ciab/SSAST-Base-Patch-400.pth'
            )
    sd = torch.load(os.path.join(model_path, 'models/best_audio_model.pth'), map_location=device)
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd, strict=False)
    return audio_model, args

def load_args(model_path):
    with open(os.path.join(model_path, 'args.pkl'), "rb") as file:
        args = pickle.load(file)
        print(args)
    return args

def get_dataset(args, path_data='/home/ec2-user/SageMaker/jbc-cough-in-a-box/ssast_ciab/src/finetune/ciab/data/datafiles/audio_sentence_url/ciab_standard_test_data_1.json'):
    print(os.path.exists(path_data))
    val_audio_conf = {
            'num_mel_bins': args.num_mel_bins, 
            'target_length': args.target_length, 
            'freqm': 0, 'timem': 0, 
            'mixup': 0, 
            'dataset': args.dataset, 
            'mode': 'evaluation', 
            'mean': args.dataset_mean, 
            'std': args.dataset_std, 
            'noise': False
            }
    eval_dataset = dataloader.AudioDataset(
        path_data,
        label_csv='/home/ec2-user/SageMaker/jbc-cough-in-a-box/ssast_ciab/src/finetune/ciab/data/ciab_class_labels_indices.csv',
        audio_conf=val_audio_conf,
        pca_proj=True)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size*2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)
    return eval_dataset, eval_loader

def get_attention(audio_model, loader, device, args):
    fbank, label_indices, index = next(iter(loader))
    fbank = fbank.to(device)
    audio_model = audio_model.to(device)
    audio_model.eval()
    with torch.no_grad():
        output, tup_out = audio_model(
            fbank,
            args.task,
            pca_proj=True,
            return_attention=True
            )
    pca_proj, attention = tup_out
    return attention, fbank.cpu().transpose(1,2)

def format_attention_map(attentions, audio_model, method, args, threshold=False):
    '''
    reshape attention so that it is the same size as orignal fbank
    '''
    print(attentions.shape)
    nh = attentions.shape[1] # number of head

    # we keep only the output patch attenion
    print(audio_model.module.f_dim)
    print(audio_model.module.t_dim)
    attentions = attentions[0, :, 0, 2:].reshape(nh, -1)
    if threshold:
        attentions = threshold_att(attentions, nh, audio_model, args)
        #plt.imsave(fname='first3.png', arr=attentions[2], format='png')
        return attentions, nh
    if method == 'frame':
        attentions = attentions.reshape(nh, audio_model.module.t_dim)
        print(attentions[0])
        attentions = attentions[0].cpu().numpy()
        #fig = plt.figure()
        #plt.plot(list(range(len(attentions))), attentions)
        #plt.savefig('attention.png')
        return attentions, nh
    else:
        print('assuming patch based approach')
        print(attentions.size())
        attentions = attentions.reshape(nh, audio_model.module.f_dim, audio_model.module.t_dim)
        attentions = nn.functional.interpolate(
                attentions.unsqueeze(0), 
                scale_factor=(args.fshape, args.tshape), 
                mode="nearest")[0].cpu().numpy()
        print(np.shape(attentions[0]))
        #plt.imsave(fname='first2.png', arr=attentions[0], format='png')
        return attentions, nh

def threshold_att(attentions, nh, audio_model, args):
    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - 0.1)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, audio_model.module.f_dim, audio_model.module.t_dim).float()
    # interpolate
    th_attn = nn.functional.interpolate(
            th_attn.unsqueeze(0), 
            scale_factor=(args.fshape, args.tshape), 
            mode="nearest"
            )[0].cpu().numpy()
    return th_attn

def plot_attentions(attensions, fbank, nh, mean, std):
    fig, axs = plt.subplots(nh+1,1, figsize=(8,20), sharex=True)
    axs[0].imshow(fbank[0])
    for i in range(nh):
        #plot for each head
        axs[i+1].imshow(attensions[i])
    plt.savefig('attentions4.png')
    np.save('fbank', (fbank[0]*std**2)+mean)

#def reverse_mel(fbank):
#    audio = librosa.feature.inverse.mel_to_audio(fbank, sr=16000)
#    print(audio)

def main(model_path, method='patch'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model, args = load_trained_model(model_path, device)
    if args.loss == 'BCE':
        args.loss_fn = torch.nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        args.loss_fn = torch.nn.CrossEntropyLoss()
    eval_dataset, eval_loader = get_dataset(args)   
    attention, fbank = get_attention(audio_model, eval_loader, device, args)
    attentions, nh = format_attention_map(attention, audio_model, method, args)
    plot_attentions(attentions, fbank, nh, eval_dataset.norm_mean, eval_dataset.norm_std)

if __name__ == '__main__':
    main('/home/ec2-user/SageMaker/jbc-cough-in-a-box/ssast_ciab/src/finetune/ciab/exp/test01-ciab_sentence-f16-16-t16-16-b18-lr1e-4-ft_cls-base-unknown-SSAST-Base-Patch-400-1x-noiseTrue-standard-train-2/fold1')
