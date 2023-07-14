import os
import datetime
from pathlib import Path
from base64 import b64decode
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as wav_write
import io
import ffmpeg
import gdown
import librosa
import torchaudio
from ssast_ciab.src.finetune.ciab.demonstration import main_demo


CONTENT_FOLDER = 'content'

DEMO_FOLDER = os.path.join(CONTENT_FOLDER,'demo')

MODEL_MODALITY_GDLINK = {
   'sentence': "https://drive.google.com/drive/folders/1Kb04tyDUaSCUDjLdC4EjzKzSIhMocQnw?usp=share_link",
   'exhalation': "https://drive.google.com/drive/folders/17GEBwClHYFhKWnZ6DpfhasFfG5RCC6mD?usp=share_link",
   'cough': "https://drive.google.com/drive/folders/1WWw4BNUobhi9Jm9Vl5WhFxLkktIObCtP?usp=share_link",
   'three_cough': "https://drive.google.com/drive/folders/1FszBlApEzEcdVNMKdHZx8i9hYB2pbtH6?usp=share_link",
}


def process_data(data):
    return binary, participant_id, modality


def binary2wav(data):
    binary = b64decode(data)

    process = (ffmpeg
        .input('pipe:0')
        .output('pipe:1', format='wav')
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
    )
    output, err = process.communicate(input=binary)

    riff_chunk_size = len(output) - 8
    # Break up the chunk size into four bytes, held in b.
    q = riff_chunk_size
    b = []
    for i in range(4):
        q, r = divmod(q, 256)
        b.append(r)

    # Replace bytes 4:8 in proc.stdout with the actual size of the RIFF chunk.
    riff = output[:4] + bytes(b) + output[8:]

    sr, audio = wav_read(io.BytesIO(riff))
    return audio, sr


def get_time():
    x = datetime.datetime.now()
    x = x.strftime("%m-%d_%H-%M")
    return x


def write_audio(audio, sr, participant_id, folder):
    filepath = os.path.join(folder, participant_id)
    Path(filepath).mkdir(parents=True, exist_ok=True)
    audio_filename = get_time() + '.wav'
    wav_write(os.path.join(filepath, audio_filename), sr, audio)
    return filepath, audio_filename


def download_model(modality):
    folder_link = MODEL_MODALITY_GDLINK[modality]
    gdown.download_folder(
        folder_link, 
        quiet=True, 
        use_cookies=False, 
        output=CONTENT_FOLDER,
    )


def main(data):
    # setup folders
    Path(DEMO_FOLDER).mkdir(parents=True, exist_ok=True)
    # process data from frontend
    binary, participant_id, modality = process_data(data)
    # decode binary aas wav and save file
    audio, sr = binary2wav(binary)
    filepath, audio_filename = write_audio(audio, sr, participant_id)
    # download model accordingly 
    download_model(modality)
    # run demo
    main_demo(mdl_folder, filepath, audio_filename, name=participant_id)
    


if __name__ == '__main__':
    mdl_folder = 'content/sentence/'
    folder = 'content/demo/'
    audio_file = 'ssast_ciab_sample.wav'
    main_demo(mdl_folder, folder, audio_file, name='participant')