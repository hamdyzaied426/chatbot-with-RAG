import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --- API Key Setup ---
# Try to obtain the API key from st.secrets first; if not available, use the environment variable.
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    # Use os.getenv with a fallback default (your provided API key)
    api_key = os.getenv("GOOGLE_API_KEY", "B5ACE85B-B14D-4968-822A-C2740BC6A061/20250209200343")

if not api_key:
    st.error("GOOGLE_API_KEY not found. Please set it in .streamlit/secrets.toml or as an environment variable.")
else:
    genai.configure(api_key=api_key)

# --- Model and Chat Setup ---
model = 'models/embedding-001'
chat_model = genai.GenerativeModel('gemini-1.5-flash')

# --- Functions for PDF Extraction and Vector DB Creation ---
def extract_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def google_pdf_gemini_embedding(task_type):
    embedding = GoogleGenerativeAIEmbeddings(model=model, task_type=task_type)
    return embedding

def create_vector_db(texts):
    v_db = FAISS.from_texts(texts, google_pdf_gemini_embedding("SEMANTIC_SIMILARITY"))
    return v_db

def get_similar_context(v_db, query, n):
    if v_db:
        docs = v_db.similarity_search(query, k=n)
        return docs
    return []

def get_response(query):
    response_generator = chat_model.generate_content(query, stream=True)
    full_response = ""
    for res in response_generator:
        if res.text:
            full_response += res.text
            yield res.text

# --- Initialize Text Splitter ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# --- Title and CSS Styling ---
title_html = """
    <style>
    .title {
        font-size: 70px;
        font-weight: 800;
        color: #c13584;
        background: -webkit-linear-gradient(#4c68d7, #ff6464);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 50px;
        font-weight: 400;
        color: #333337;
    }
    </style>
    <div class="title">Hello,</div>
    <div class="subtitle">How can I help you today?</div>
    """
st.markdown(title_html, unsafe_allow_html=True)

# --- Initialize Session State ---
if "pdf" not in st.session_state:
    st.session_state.pdf = None
if "v_db" not in st.session_state:
    st.session_state.v_db = None
if "texts" not in st.session_state:
    st.session_state.texts = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: PDF Upload and Vector DB Management ---
with st.sidebar:
    st.title("Chatbot Controls")
    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf and st.button("Create Vector Database"):
        with st.spinner("Creating vector database..."):
            texts = text_splitter.split_text(extract_from_pdf(pdf))
            st.session_state.v_db = create_vector_db(texts)
            st.session_state.pdf = pdf
            st.session_state.texts = texts
            st.success("Vector database created successfully!")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.success("Chat history cleared!")
    if st.button("Delete Vector Database"):
        st.session_state.v_db = None
        st.session_state.pdf = None
        st.session_state.texts = None
        st.success("Vector database deleted!")

# --- Display Previous Chat Messages ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Chat Input and Response Generation ---
user_input = st.chat_input("Enter your message:")

if user_input:
    st.chat_message("user").write(user_input)
    placeholder = st.chat_message("AI").empty()
    
    # Base context text for the chatbot
    similar_text = "You are a Multi Task AI Agent. "
    
    # Append the new user message to the conversation history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # If a vector database exists, add similar context from the PDF
    if st.session_state.v_db:
        similar_context = get_similar_context(st.session_state.v_db, user_input, 5)
        for doc in similar_context:
            similar_text += doc.page_content + "\n"
    
    # To avoid overly long input, limit conversation history to the last 3 messages
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-3:]])
    
    # Combine the conversation history, new query, and similar context
    combined_input = f"{conversation_history}\nuser: {user_input}\nAI: {similar_text}"
    
    stream_res = ""
    with st.spinner("Thinking..."):
        # Use the minimal prompt for testing if needed:
        # test_input = "Hello, how are you?"
        # for response in get_response(test_input):
        #     stream_res += response
        #     placeholder.markdown(stream_res)
        
        # Use the combined input
        for response in get_response(combined_input):
            stream_res += response
            placeholder.markdown(stream_res)
    
    st.session_state.messages.append({"role": "AI", "content": stream_res})
