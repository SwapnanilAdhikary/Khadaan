import os
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("🔴 Error: Gemini API key is missing. Set it in a .env file.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Streamlit Page Setup
st.set_page_config(page_title="MinerlexAI", page_icon="🤖", layout="wide")

# Gemini Model Configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=generation_config,
)

# Language Mappings
LANGUAGE_CODES = {
    "English": "en_XX",
    "Hindi (हिन्दी)": "hi_IN",
    "Bengali (বাংলা)": "bn_IN",
    "Tamil (தமிழ்)": "ta_IN",
    "Telugu (తెలుగు)": "te_IN",
    "Marathi (मराठी)": "mr_IN",
    "Gujarati (ગુજરાતી)": "gu_IN",
    "Malayalam (മലയാളം)": "ml_IN",
    "Kannada (ಕನ್ನಡ)": "kn_IN",
    "Odia (ଓଡ଼ିଆ)": "or_IN",
    "Urdu (اردو)": "ur_PK",
    "Assamese (অসমীয়া)": "as_IN",
}

GTTS_LANGUAGE_CODES = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "Bengali (বাংলা)": "bn",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Marathi (मराठी)": "mr",
    "Gujarati (ગુજરાતी)": "gu",
    "Malayalam (മലയാളം)": "ml",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Urdu (اردو)": "ur",
    "Assamese (অসমীয়া)": "as",
}
# Load MBart Model for Translation
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
translator_model = MBartForConditionalGeneration.from_pretrained(model_name)


def translate_text(text, target_language):
    """Translates text to the selected language using MBart model."""
    if target_language == "English":
        return text

    target_lang_code = LANGUAGE_CODES.get(target_language, "en_XX")

    try:
        model_inputs = tokenizer(text, return_tensors="pt")
        generated_tokens = translator_model.generate(
            **model_inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang_code]
        )
        translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text


def generate_audio(text, language):
    """Converts translated text to speech using gTTS."""
    gtts_lang = GTTS_LANGUAGE_CODES.get(language, "en")

    try:
        tts = gTTS(text, lang=gtts_lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio_path = temp_audio.name
            tts.save(temp_audio_path)

        return temp_audio_path
    except Exception as e:
        st.error(f"Audio generation error: {e}")
        return None


# UI Layout
st.markdown("""
    <style>
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }

        .title-container {
            text-align: center;
            animation: fadeIn 1.5s ease-in-out;
            padding: 20px;
        }

        .title-text {
            font-size: 50px;
            font-weight: bold;
            background: -webkit-linear-gradient(45deg, #FF4B4B, #FF6B6B, #FF8B8B);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle-text {
            font-size: 22px;
            color: #BBBBBB;
            margin-top: -10px;
            font-style: italic;
        }
    </style>

    <div class="title-container">
        <div class="title-text">🚀 MinerlexAI</div>
        <div class="subtitle-text">Revolutionizing Mining Law with AI-Powered Precision</div>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

# Sidebar for File Upload
with col1:
    st.sidebar.header("📂 Upload Your PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF", type="pdf")

# Centered Chat Interface
with col2:
    st.markdown('<div style="text-align:center;">', unsafe_allow_html=True)

    # Language Selection Dropdown
    selected_language = st.selectbox("🌍 Choose Language:", list(LANGUAGE_CODES.keys()))

    # Chat Input & Send Button
    user_input = st.text_input("💬 Type your message:", key="chat_input")

    # Create a row layout for buttons
    colA, colB = st.columns([1, 1])

    # Adjust button styling with CSS
    button_style = """
        <style>
            div.stButton > button {
                width: 100%;
                height: 50px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 10px;
                background-color: #007BFF;
                color: white;
                border: none;
            }
        </style>
    """

    st.markdown(button_style, unsafe_allow_html=True)

    # Buttons with equal size
    with colA:
        send_button = st.button("📩 Send")

    with colB:
        speech_button = st.button("🎤 Speak")

    # Speech Recognition Setup
    recognizer = sr.Recognizer()
    if speech_button:
        with sr.Microphone() as source:
            st.info("🎙️ Listening...")
            try:
                audio = recognizer.listen(source)
                user_input = recognizer.recognize_google(audio, language="en")
                st.success(f"🎙️ Recognized: {user_input}")
            except sr.UnknownValueError:
                st.error("⚠️ Could not understand the audio.")
            except sr.RequestError:
                st.error("⚠️ Speech recognition error. Check your internet connection.")

    # Chat Processing
    if send_button or user_input:
        if user_input:
            with st.spinner("🔍 Searching for answers..."):
                chat_session = model.start_chat(history=[])
                answer = chat_session.send_message(user_input).text
                translated_text = translate_text(answer, selected_language)

                st.subheader("💡 AI Response:")
                st.success(answer)

                st.subheader(f"🌐 {selected_language} Translation:")
                st.write(translated_text)

                # Convert Translated Text to Speech
                audio_path = generate_audio(translated_text, selected_language)
                if audio_path:
                    st.audio(audio_path, format="audio/mp3")

    st.markdown('</div>', unsafe_allow_html=True)
