import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write
import util_functions as ufs
import time


st.title('Noise-Suppressor')
st.subheader('Removes background-noise from audio samples')


nav_choice = st.sidebar.radio('Navigation', ('Home', 'Stats'), index=0)

_param_dict = {'audio_sample': None, 'predicted_outcome': None}
_path_to_model = 'utils/models/auto_encoders_for_noise_removal_production.h5'
_targe_file = 'utils/outputs/preds.wav'

if nav_choice == 'Home':
    st.image('utils/images/header.jpg', width=450, height=500)

    st.info('Upload your audio sample below')
    audio_sample = st.file_uploader('Audio Sample', ['wav'])
    if audio_sample:
        try:
            print(type(audio_sample))
            model = ufs.load_model(_path_to_model)
            audio = tf.audio.decode_wav(audio_sample.read(), desired_channels=1)
            print(audio.audio.shape)
            _param_dict['audio_sample'] = audio.audio
            with st.spinner('Wait for it...'):
                time.sleep(1)
                prog1 = st.progress(0)
                preds = model.predict(tf.expand_dims(audio.audio, 0))
                preds = tf.reshape(preds, (-1, 1))
                print(preds.shape, 'preds shape')
                _param_dict['predicted_outcome'] = preds
                preds = np.array(preds)
                write(_targe_file, 44100, preds)
            st.success('Audio after noise removal')
            st.audio(_targe_file)
        except Exception as e:
            print(e, type(e))








