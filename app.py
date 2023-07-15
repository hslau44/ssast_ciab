import numpy as np
import streamlit as st
from utils import write_audio, setup_folder, download_model, CONTENT_FOLDER, main_demo


def main():
    st.write('### Upload sound file in .wav format')
    # data input 
    modality = 'sentence'
    participant_id = '0001'
    # setup
    audio_folder = setup_folder(participant_id)
    mdl_folder = download_model(modality, folder=CONTENT_FOLDER)
    # get and analyse audio 
    audio = st.file_uploader(' ', type='wav')
    if audio is not None:  
        # Write audio
        audio_file = write_audio(audio, sr=16000, folder=audio_folder)
        st.write(f'audio_file: {audio_file}')
        # Play audio
        st.write('### Play audio')
        audio_bytes = audio.read()
        st.audio(audio_bytes, format='audio/wav')
        # Prediction
        results = main_demo(mdl_folder, audio_folder, audio_file, name='participant')
        if st.button('Predict'):
            if np.mean(results['prediction']) > 0.5:
                st.write('Prediction result: Positive')
            else:
                st.write('Prediction result: Negative}')
    else:
        st.write('The file has not been uploaded yet')
    return


st.title("COVID 19 Sound Detect")