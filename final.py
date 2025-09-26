import os
import numpy as np
import librosa
import tensorflow as tf
import streamlit as st
import requests
from streamlit_lottie import st_lottie

# First things first: my model was trained with an LSTM layer, but while loading,
# TensorFlow sometimes throws issues with extra arguments (like `time_major`).
# To fix that, I made a custom LSTM class that just ignores that argument.
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)  # just removing the problematic arg
        super().__init__(*args, **kwargs)

# Loading the trained model and the label encoder (which maps numbers → emotions).
# I saved them earlier when training the emotion recognition model.
MODEL_PATH = "emotion_model.h5"
ENCODER_PATH = "label_encoder.npy"

model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"LSTM": CustomLSTM})
label_encoder = np.load(ENCODER_PATH, allow_pickle=True)

# I wanted this app to be fun, so for every detected emotion I prepared a short
# list of movies that go well with that mood.
MOVIE_DB = {
    "angry": ["Mad Max: Fury Road", "John Wick", "Gladiator"],
    "disgust": ["Requiem for a Dream", "Trainspotting", "American Beauty"],
    "fearful": ["The Conjuring", "Insidious", "The Babadook"],
    "happy": ["The Intouchables", "Amélie", "La La Land"],
    "neutral": ["Inception", "The Social Network", "Interstellar"],
    "sad": ["Manchester by the Sea", "A Star is Born", "The Notebook"]
}

# Audio can be messy, so here I’m extracting MFCC (Mel Frequency Cepstral Coefficients).
# Basically, it converts raw audio into numbers that represent features like tone/pitch.
# I also make sure the shape is fixed (130 time steps), otherwise the model won’t accept it.
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < 130:
        mfcc = np.pad(mfcc, ((0, 0), (0, 130 - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :130]

    return mfcc.T[np.newaxis, ...]

# Now the Streamlit UI part (the fun part!).
# Setting up the page and adding a title.
st.set_page_config(page_title="Speech Emotion Recognition", layout="wide")
st.title("🎤 Emotion Detection + Movie Recs 🍿")

st.markdown("Upload or record your voice, and I’ll guess your mood + suggest movies!")

# I’ve made two tabs: one for uploading an existing file, and one for directly recording audio.
tab_upload, tab_record = st.tabs(["📁 Upload Audio", "🎙️ Record Audio"])

with tab_upload:
    uploaded_audio = st.file_uploader("Upload a WAV file", type=["wav"])

with tab_record:
    uploaded_audio = st.audio_input("Record your voice here")

# Once I have the audio, I save it temporarily so librosa can process it.
if uploaded_audio:
    temp_file = "temp.wav"
    with open(temp_file, "wb") as f:
        f.write(uploaded_audio.read())

    # This lets the user actually play back the file they just uploaded/recorded.
    st.audio(temp_file)

    # Extract MFCC features and run it through the model.
    features = extract_features(temp_file)
    prediction = model.predict(features)
    detected_emotion = label_encoder[np.argmax(prediction)]

    # Some datasets mark "calm" separately, but I merged it with "neutral".
    if detected_emotion == "calm":
        detected_emotion = "neutral"

    # A little animation before showing results (makes the app feel alive).
    anim1 = requests.get("https://assets2.lottiefiles.com/packages/lf20_q8mzv4.json")
    if anim1.status_code == 200:
        st_lottie(anim1.json(), height=200)

    # Show the detected emotion in a friendly style.
    st.subheader(f"You sound **{detected_emotion.capitalize()}** 🎧")
    st.write("Here are some movies you might enjoy:")

    # Display movie recommendations nicely in 3 columns.
    recs = MOVIE_DB.get(detected_emotion, [])
    cols = st.columns(3)
    for i, m in enumerate(recs):
        cols[i % 3].markdown(f"- 🎬 **{m}**")

    # Another animation after showing movies.
    anim2 = requests.get("https://assets8.lottiefiles.com/packages/lf20_2sz3v2sf.json")
    if anim2.status_code == 200:
        st_lottie(anim2.json(), height=200)

    # Clean up the temporary file.
    os.remove(temp_file)

# And finally, I added one last animation at the bottom as a finishing touch.
final_anim = requests.get("https://lottie.host/74af8bbc-2a50-4f1b-a8bd-beb53239bd92/vzcZEXtLSc.json")
if final_anim.status_code == 200:
    st_lottie(final_anim.json())
