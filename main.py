import streamlit as st
import requests
import json
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
from googletrans import Translator

# Hugging Face API details for MinerLex AI Chatbot
API_URL = "https://api-inference.huggingface.co/models/minerlex/mining_law_chatbot"
HEADERS = {"Authorization": "Bearer YOUR_HUGGINGFACE_TOKEN"}

translator = Translator()


def query(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()


def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language='en')  # Default language English
        return text
    except sr.UnknownValueError:
        return "Sorry, could not understand the audio."
    except sr.RequestError:
        return "Could not request results, please check your connection."


# Streamlit UI setup
st.title("⚖ MinerLex AI - Mining Law Chatbot")
st.write("A chatbot designed to provide legal insights and answer queries related to mining laws.")

# Language selection
languages = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Malayalam": "ml",
    "Kannada": "kn",
    "Odia": "or",
    "Urdu": "ur"
}
selected_lang = st.selectbox("Select Language", list(languages.keys()))


def translate_text(text, target_lang):
    return translator.translate(text, dest=target_lang).text


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.text_input("Type your message here")
if st.button("🎤 Speak"):
    user_input = recognize_speech()
    st.write("Recognized Text: ", user_input)

if user_input:
    user_input_translated = translate_text(user_input, "en")
    st.session_state.messages.append({"role": "user", "content": user_input_translated})

    # Generate response using Hugging Face API
    response = query({"inputs": user_input_translated})
    bot_response = response[0]["generated_text"] if isinstance(response, list) else "Sorry, I couldn't process that."
    bot_response_translated = translate_text(bot_response, languages[selected_lang])

    # Append chatbot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response_translated})

    # Display chatbot response
    with st.chat_message("assistant"):
        st.markdown(bot_response_translated)

    # Convert response to speech
    tts = gTTS(bot_response_translated, lang=languages[selected_lang])
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts.save(fp.name)
        st.audio(fp.name, format='audio/mp3')