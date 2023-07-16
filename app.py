import numpy as np
import streamlit as st
import time
from recorder.st_custom_components import st_audiorec
from utils import write_audio, setup_folder, download_model, CONTENT_FOLDER, main_demo, binary2wav


def run(inputs):
    audio_folder = setup_folder()
    mdl_folder = download_model(inputs['modal'], folder=CONTENT_FOLDER)
    audio = binary2wav(inputs['audio'])
    audio_file = write_audio(audio, sr=16000, folder=audio_folder)
    results = main_demo(mdl_folder, audio_folder, audio_file, name=inputs['name'])
    return results


def fake_process(**kwargs):
    time.sleep(5)
    outputs = {k:v for k,v in kwargs.items()}
    return outputs


def check_inputs(**inputs):
    result = True
    message = None
    for k,i in inputs.items():
        if i is None or i == '':
            result = False
            message = f"Please provide correct '{k}'"
            break
    return result, message



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
            c = fake_process(**inputs)
        st.header('Result')
        st.write(c)
    else:
        st.write(message)