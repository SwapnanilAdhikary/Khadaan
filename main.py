import os
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import tempfile
import google.generativeai as genai
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Configure Gemini API
genai.configure(api_key="AIzaSyC3d0e-4_YonHIwrrFNbMrCE6_nJahsagU")

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash-8b",
  generation_config=generation_config,
)

# RAG Setup
pdf_directory = "./"
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_store = InMemoryVectorStore(embeddings)
rag_model = OllamaLLM(model="deepseek-r1:1.5b")

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

def translate_to_hindi(text):
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer_name = "facebook/mbart-large-50-one-to-many-mmt"
    
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(tokenizer_name, src_lang="en_XX")
    
    model_inputs = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
    )
    
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translation[0]

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def index_docs(documents):
    vector_store.add_documents(documents)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)    
    chain = prompt | rag_model
    return chain.invoke({"question": question, "context": context})

# Streamlit UI
st.title("Minerlex ")

uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)
if uploaded_file:
    file_path = os.path.join(pdf_directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    documents = load_pdf(file_path)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)

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
    related_documents = retrieve_docs(user_input)
    if related_documents:
        answer = answer_question(user_input, related_documents)
    else:
        chat_session = model.start_chat(history=[])
        answer = chat_session.send_message(user_input).text
    
    translated_text = translate_to_hindi(answer)
    st.write("Response in Hindi: ", translated_text)
    
    tts = gTTS(translated_text, lang="hi")
    with tempfile.TemporaryDirectory() as tempdir:
        temp_audio_path = os.path.join(tempdir, "response.mp3")
        tts.save(temp_audio_path)
        st.audio(temp_audio_path, format="audio/mp3")
