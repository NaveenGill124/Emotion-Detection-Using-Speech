import os
import numpy as np
import tensorflow as tf
import librosa
import streamlit as st
import matplotlib.pyplot as plt
import json
import requests
from streamlit_lottie import st_lottie
from tensorflow import keras

# -------------------------------
# Custom LSTM to fix model loading (if needed)
# -------------------------------
def custom_lstm(*args, **kwargs):
    if 'time_major' in kwargs:
        kwargs.pop('time_major')
    return tf.keras.layers.LSTM(*args, **kwargs)

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.utils import register_keras_serializable

import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)  # Remove unsupported argument if present
        super(CustomLSTM, self).__init__(*args, **kwargs)

# -------------------------------
# Load the model and label encoder
# -------------------------------
model_path = r"C:\Users\CITRA\Desktop\Naveen Gill\emotion_detection\best_model_new.h5"
label_encoder_path = r"C:\Users\CITRA\Desktop\Naveen Gill\emotion_detection\label_encoder.npy"

# model = tf.keras.models.load_model(model_path, custom_objects={"LSTM": custom_lstm})

# model = tf.keras.models.load_model("emotion_model_updated.h5", custom_objects={"CustomLSTM": CustomLSTM})

model = tf.keras.models.load_model("emotion_model_updated.h5", custom_objects={"LSTM": CustomLSTM})



le = np.load(label_encoder_path, allow_pickle=True)

# -------------------------------
# Define movie recommendations for each emotion (3 per emotion)
# -------------------------------
MOVIE_DB = {
    'angry': ['Mad Max: Fury Road', 'John Wick', 'Gladiator'],
    #'calm': ['The Secret Life of Walter Mitty', 'Lost in Translation', 'The Grand Budapest Hotel'],
    'disgust': ['Requiem for a Dream', 'Trainspotting', 'American Beauty'],
    'fearful': ['The Conjuring', 'Insidious', 'The Babadook'],
    'happy': ['The Intouchables', 'Amélie', 'La La Land'],
    'neutral': ['Inception', 'The Social Network', 'Interstellar'],
    'sad': ['Manchester by the Sea', 'A Star is Born', 'The Notebook'],
    #'surprised': ['The Prestige', 'Memento', 'Shutter Island']
}

# -------------------------------
# Function to process audio: extract MFCC features and prepare input for model
# -------------------------------
def process_audio(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    # Pad or truncate MFCCs to fixed length (130 time steps)
    if mfccs.shape[1] < 130:
        mfccs = np.pad(mfccs, ((0, 0), (0, 130 - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :130]
    # Transpose so that shape is (time_steps, n_mfcc) and add a batch dimension
    return mfccs.T[np.newaxis, ...]

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="Speech Emotion Recognition", layout="wide")
st.title("Speech Emotion Recognition & Movie Recommendation 🎤🍿")
st.markdown(
    """
    Welcome! Upload or record an audio file, and our model will detect your emotion 
    and recommend movies accordingly.
    """
)

tab1, tab2 = st.tabs(["📁 Upload Audio", "🎙️ Record Audio"])
# audio_file = st.audio_input("Record a voice message")

with tab1:
    # Use Streamlit's file uploader for an audio file
    audio_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])

with tab2:
    # If you use a recorder component or any plugin, you can integrate it here
    audio_file = st.audio_input("Record a voice message")

if audio_file:
    # Save the uploaded file to a temporary location
    temp_audio_path = "temp.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_file.read())
    
    # Optionally, display the audio player
    st.audio(temp_audio_path)
    
    # Process the audio to get MFCC features
    mfcc_input = process_audio(temp_audio_path)
    
    # Predict emotion from the processed audio
    pred = model.predict(mfcc_input)
    predicted_emotion = le[np.argmax(pred)]

    # Remap 'calm' to 'neutral'
    if predicted_emotion == "calm":
        predicted_emotion = "neutral"
    
    # Display a fun animation above the result (optional)
    # Grab a Lottie animation from a URL
    url_anim_before = requests.get("https://assets2.lottiefiles.com/packages/lf20_q8mzv4.json")  # example
    if url_anim_before.status_code == 200:
        anim_before_json = url_anim_before.json()
        st_lottie(anim_before_json, height=200)
    
    # Display the detected emotion in a more conversational style
    st.markdown(
        f"### I think you are feeling **{predicted_emotion.capitalize()}** right now!"
    )
    st.markdown(
        "No worries—I've got your back. "
        "Here are some movie suggestions that might lift your mood. "
        "Watch them and thank me later! 😉"
    )
    
    # Retrieve the list of movies for the detected emotion
    recommendations = MOVIE_DB.get(predicted_emotion, [])
    
    # Use Streamlit columns to display recommendations nicely
    cols = st.columns(3)
    for i, movie in enumerate(recommendations):
        cols[i % 3].markdown(f"- 🎬 **{movie}**")
    
    # Optionally, display another animation after the recommendations
    url_anim_after = requests.get("https://assets8.lottiefiles.com/packages/lf20_2sz3v2sf.json")  # example
    if url_anim_after.status_code == 200:
        anim_after_json = url_anim_after.json()
        st_lottie(anim_after_json, height=200)

    # Clean up the temporary file
    os.remove(temp_audio_path)

# Display a final Lottie animation or any other decorative element at the bottom
url1 = requests.get("https://lottie.host/74af8bbc-2a50-4f1b-a8bd-beb53239bd92/vzcZEXtLSc.json")
if url1.status_code == 200:
    url_json1 = url1.json()
    st_lottie(url_json1)
