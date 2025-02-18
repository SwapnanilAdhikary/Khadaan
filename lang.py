import streamlit as st
import requests
import json
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
from googletrans import Translator
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def translate_to_hindi(text):
    # Load model and tokenizer
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer_name = "facebook/mbart-large-50-one-to-many-mmt"
    
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(tokenizer_name, src_lang="en_XX")
    
    # Tokenize input text
    model_inputs = tokenizer(text, return_tensors="pt")
    
    # Generate translation
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
    )
    
    # Decode output
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translation[0]

# Streamlit UI setup
st.title("⚖ MinerLex AI - Mining Law Chatbot")
st.write("A chatbot designed to provide legal insights and answer queries related to mining laws.")

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
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        audio = recognizer.listen(source)
    try:
        user_input = recognizer.recognize_google(audio, language='en')
        st.write("Recognized Text: ", user_input)
    except sr.UnknownValueError:
        user_input = "Sorry, I could not understand the audio."
    except sr.RequestError:
        user_input = "Speech recognition error. Check your internet connection."

if user_input:
    # Translate input to Hindi
    bot_response_translated = translate_to_hindi(user_input)
    st.session_state.messages.append({"role": "assistant", "content": bot_response_translated})

    # Display chatbot response
    with st.chat_message("assistant"):
        st.markdown(bot_response_translated)

    # Convert response to speech
    tts = gTTS(bot_response_translated, lang="hi")
    with tempfile.TemporaryDirectory() as tempdir:
        temp_audio_path = os.path.join(tempdir, "response.mp3")
        tts.save(temp_audio_path)
        st.audio(temp_audio_path, format="audio/mp3")
