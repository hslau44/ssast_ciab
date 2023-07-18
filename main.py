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

MODAL_TEXT = """
**Sentence**: Record a sentence “I love nothing more than an afternoon cream tea.” This sentence contains\
      some key sounds (‘aaah’, ‘oooh’, ‘eee’) which may help to indicate your respiratory health Press \
        the Record button, and read the following sentence: “I love nothing more than an afternoon cream \
            tea.” Use the Stop button to stop recording.

**Exhalation**: Record a 'ha' sound Please make this recording in a quiet environment. Press the Record \
    button, and breathe out loudly three times, making a ‘ha’ sound, as if you were trying to fog up a \
        window, or see your breath in cold weather. Use the Stop button to stop recording. You will see \
            an audio player, which you can use to playback your recording. Breathe out loudly three \
                times, making a ‘ha’ sound, as if you were trying to fog up a window, or see your breathz
                  in cold weather. Press the Record button to begin recording

**Cough**: Record a cough. Coughing is a potential risk to others around you. Make sure you are alone\
      in a room or vehicle when coughing. For this recording, move an arm’s length away from your \
        desktop computer, laptop, phone or tablet. Press the Record button, and cough, forcing a cough\
              if it doesn’t come naturally. Use the Stop button to stop recording. You will see an audio\
                  player, which you can use to playback your recording. Cough once — with your desktop \
                    computer, laptop, phone or tablet an arm’s length away from you.
"""


RESULT_KEY_TEXTS = {
    "waveform":'Here is the waveform representation of your voice sample',
    "prediction":"The AI employs a very interesting algorithm called 'Attention', as the named suggested\
        the model learnt to pay attention to specific timestep of your voice to make prediction, the 1st\
        figure shows parts of the spectrogram paid the most attention to, highlighted in color\nAnd the\
        2nd figure shows the COVID prediction/logit for each timestep, where logit > 0.5 is considered as\
        'Positive'.",
}


def get_time():
    x = datetime.datetime.now()
    x = x.strftime("%m-%d_%H-%M")
    return x

def get_pid(name):
    return ''.join(e for e in name if e.isalnum())


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


def write_audio(audio, sr, folder, name):
    audio_filename = name + '.wav'
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


def setup_folder(name=None):
    Path(DEMO_FOLDER).mkdir(parents=True, exist_ok=True)
    if name is not None:
        pid = get_pid(name)
        folder = os.path.join(DEMO_FOLDER, pid)
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


def run(inputs, **kwargs):

    pid = get_pid(inputs['name'])

    audio_folder = setup_folder()

    audio, sr = binary2wav(inputs['audio'])

    audio_file = write_audio(audio, sr=sr, folder=audio_folder, name=pid)

    mdl_folder = download_model(inputs['modal'], folder=CONTENT_FOLDER)

    results = main_demo(mdl_folder, audio_folder, audio_file, name=pid)

    return results


if __name__ == "__main__":

    st.title("COVID 19 Sound Detect")

    st.write("**WARNING: This is NOT a COVID-19 diagnostic test, this demo is \
             purely educational, does not provide any medical recommendation nor\
              should any action be taken following use**")
    
    st.markdown("This app is a demonstration of the Audio-based AI classifiers \
             developed by the Alan Turning Institute and the UK Health Security\
             Agency, you can find the original source [here](https://colab.research.google.com/drive/1Hdy2H6lrfEocUBfz3LoC5EDJrJr2GXpu?usp=sharing#scrollTo=giQTg9YoGBpI).")
    
    st.header("Instruction")

    st.markdown("**1. Provide your name below:**")
    
    inputs = {}
    inputs['name'] = st.text_input('Name')

    st.text(' ')
    
    st.markdown("**2. Select one of the three modalities below.**") 
    
    st.markdown("Depending on your selection please \
                follow instructions of the tabs to make your \
                recording in the next step.")
    
    tab1, tab2, tab3 = st.tabs(['sentence', 'cough', 'exhalation'])

    tab1.write("Record a sentence “I love nothing more than an \
               afternoon cream tea.” This sentence contains some key sounds \
               (‘aaah’, ‘oooh’, ‘eee’) which may help to indicate your \
               respiratory health Press the Record button, and read the \
               following sentence: “I love nothing more than an afternoon \
               cream tea.”")
    
    tab2.write("Record a 'ha' sound Please make this\
                recording in a quiet environment. Press the Record \
                button, and breathe out loudly three times, making a ‘ha’ \
                sound, as if you were trying to fog up a window, or see \
                your breath in cold weather. Use the Stop button to stop \
                recording. You will see an audio player, which you can use \
                to playback your recording. Breathe out loudly three times, \
                making a ‘ha’ sound, as if you were trying to fog up a\
                window, or see your breath in cold weather.")

    tab3.write("Record a cough. Coughing is a potential risk to \
                others around you. Make sure you are alone in a room or \
                vehicle when coughing. For this recording, move an arm’s \
                length away from your desktop computer, laptop, phone or \
                tablet. Press the Record button, and cough, forcing a cough \
                if it doesn’t come naturally. Use the Stop button to stop \
                recording. You will see an audio player, which you can use \
                to playback your recording. Cough once — with your desktop \
                computer, laptop, phone or tablet an arm’s length away from \
                you.")
    

    inputs['modal'] = st.selectbox(
        'Modality', 
        ['sentence', 'cough', 'exhalation']
    )

    st.text(' ')

    st.markdown("**3. Select the buttoms to make your recording:**")

    inputs['audio'] = st_audiorec()

    st.text(' ')

    st.markdown("**4. Click 'Predict', it will take about \
                5 minutes to complete and show the results.**")

    b = st.button('Predict')

    if b:
        result, message = check_inputs(**inputs)
        
        if result == True:
            with st.spinner(text='In progress'):
                c = run(inputs)
            st.header('Result')
            for k, text in RESULT_KEY_TEXTS.items():
                if k in c.keys():
                    st.subheader(k)
                    st.write(text)
                    st.image(c[k])
        else:
            st.write(message)