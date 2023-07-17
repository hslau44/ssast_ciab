import os
import io
import sys
from base64 import b64decode
import datetime
import time
from pathlib import Path
import ffmpeg
import soundfile as sf
import numpy as np
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as wav_write
import librosa
import torchaudio
import gdown
import streamlit as st
from recorder.st_custom_components import st_audiorec
from ssast_ciab.src.finetune.ciab.demonstration import main_demo


CUR_DIR = os.path.dirname(__file__)

CONTENT_FOLDER = os.path.join(CUR_DIR,'content')

DEMO_FOLDER = os.path.join(CONTENT_FOLDER,'demo')

MODEL_MODALITY_GDLINK = {
   'sentence': "https://drive.google.com/drive/folders/1Kb04tyDUaSCUDjLdC4EjzKzSIhMocQnw?usp=share_link",
   'exhalation': "https://drive.google.com/drive/folders/17GEBwClHYFhKWnZ6DpfhasFfG5RCC6mD?usp=share_link",
   'cough': "https://drive.google.com/drive/folders/1WWw4BNUobhi9Jm9Vl5WhFxLkktIObCtP?usp=share_link",
   'three_cough': "https://drive.google.com/drive/folders/1FszBlApEzEcdVNMKdHZx8i9hYB2pbtH6?usp=share_link",
}


def get_time():
    x = datetime.datetime.now()
    x = x.strftime("%m-%d_%H-%M")
    return x


def binary2wav(data):
    # binary = b64decode(data)

    # process = (ffmpeg
    #     .input('pipe:0')
    #     .output('pipe:1', format='wav')
    #     .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
    # )
    # output, err = process.communicate(input=binary)

    # riff_chunk_size = len(output) - 8
    # # Break up the chunk size into four bytes, held in b.
    # q = riff_chunk_size
    # b = []
    # for i in range(4):
    #     q, r = divmod(q, 256)
    #     b.append(r)

    # # Replace bytes 4:8 in proc.stdout with the actual size of the RIFF chunk.
    # riff = output[:4] + bytes(b) + output[8:]

    # sr, audio = wav_read(io.BytesIO(riff))

    audio, sr = sf.read(io.BytesIO(data))
    return audio, sr


def write_audio(audio, sr, folder):
    audio_filename = get_time() + '.wav'
    wav_write(os.path.join(folder, audio_filename), sr, audio)
    return audio_filename


def download_model(modality, folder):
    mdl_path = os.path.join(folder,modality)
    if not os.path.exists(mdl_path):
        Path(mdl_path).mkdir(parents=True, exist_ok=True)
        folder_link = MODEL_MODALITY_GDLINK[modality]
        gdown.download_folder(
            folder_link, 
            quiet=True, 
            use_cookies=False, 
            output=mdl_path,
        )
    return mdl_path


def setup_folder(participant_id=None):
    Path(DEMO_FOLDER).mkdir(parents=True, exist_ok=True)
    if participant_id is not None:
        folder = os.path.join(DEMO_FOLDER, participant_id)
        Path(folder).mkdir(parents=True, exist_ok=True)
        return folder
    else:
        return DEMO_FOLDER


def check_inputs(**inputs):
    result = True
    message = None
    for k,i in inputs.items():
        if i is None or i == '':
            result = False
            message = f"Please provide correct '{k}'"
            break
    return result, message


def fake_process(**kwargs):
    time.sleep(5)
    outputs = {k:v for k,v in kwargs.items()}
    return outputs


def run(inputs, verbose=False):
    print('Set up folder') if verbose else 0
    audio_folder = setup_folder()
    print('Decoding binary')  if verbose else 0
    audio, sr = binary2wav(inputs['audio'])
    print('Saving audio')  if verbose else 0
    audio_file = write_audio(audio, sr=sr, folder=audio_folder)
    print('Download model')  if verbose else 0
    mdl_folder = download_model(inputs['modal'], folder=CONTENT_FOLDER)
    # results = main_demo(mdl_folder, audio_folder, audio_file, name=inputs['name'])
    results = {'audio_file': audio_file}
    return results


if __name__ == "__main__":

    st.title("COVID 19 Sound Detect")

    inputs = {}
    inputs['name'] = st.text_input('Enter your name')
    inputs['modal'] = st.selectbox('Select the type of sample:', ['sentence', 'cough', 'exhalation'])
    inputs['audio'] = st_audiorec()

    b = st.button('Predict')

    if b:
        result, message = check_inputs(**inputs)
        
        if result == True:
            with st.spinner(text='In progress'):
                c = run(inputs, verbose=True)
            st.header('Result')
            st.write(c)
        else:
            st.write(message)