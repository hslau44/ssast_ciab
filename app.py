import numpy as np
import streamlit as st
import time
from recorder.st_custom_components import st_audiorec
# from utils import write_audio, setup_folder, download_model, CONTENT_FOLDER, main_demo


# def main():
#     st.write('### Upload sound file in .wav format')
#     # data input 
#     modality = 'sentence'
#     participant_id = '0001'
#     # setup
#     audio_folder = setup_folder(participant_id)
#     mdl_folder = download_model(modality, folder=CONTENT_FOLDER)
#     # get and analyse audio 
#     audio = st.file_uploader(' ', type='wav')
#     if audio is not None:  
#         # Write audio
#         audio_file = write_audio(audio, sr=16000, folder=audio_folder)
#         st.write(f'audio_file: {audio_file}')
#         # Play audio
#         st.write('### Play audio')
#         audio_bytes = audio.read()
#         st.audio(audio_bytes, format='audio/wav')
#         # Prediction
#         results = main_demo(mdl_folder, audio_folder, audio_file, name='participant')
#         if st.button('Predict'):
#             if np.mean(results['prediction']) > 0.5:
#                 st.write('Prediction result: Positive')
#             else:
#                 st.write('Prediction result: Negative}')
#     else:
#         st.write('The file has not been uploaded yet')
#     return


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