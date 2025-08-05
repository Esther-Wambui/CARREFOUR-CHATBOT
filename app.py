import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="ASK-Carrefour", page_icon="🛒", layout="centered")

# Robust Carrefour-inspired styling
st.markdown("""
    <style>
    body, .main, .stApp {
        background-color: #f5faff !important;
    }
    .carrefour-header {
        background: #0056b3;
        padding: 1.2em 0 1em 0;
        border-radius: 0 0 18px 18px;
        margin-bottom: 1.5em;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px #0056b340;
    }
    .basket-icon {
        font-size: 2.5rem;
        margin-right: 0.7em;
        color: #fff;
    }
    .ask-title {
        color: #fff;
        font-size: 2.3rem;
        font-weight: bold;
        font-family: 'Segoe UI', Arial, sans-serif;
        letter-spacing: 1px;
        margin-bottom: 0;
    }
    .ask-subtitle {
        color: #0056b3;
        font-size: 1.1rem;
        text-align: center;
        font-family: 'Segoe UI', Arial, sans-serif;
        margin-bottom: 2em;
        margin-top: -1em;
    }
    .user-bubble {
        background-color: #e6f0ff;
        color: #0056b3;
        border-radius: 16px;
        padding: 8px 16px;
        margin: 6px 0;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 1rem;
        text-align: right;
        width: fit-content;
        max-width: 70%;
        float: right;
        clear: both;
        box-shadow: 0 2px 8px #0056b320;
    }
    .bot-bubble {
        background-color: #d0e7ff;
        color: #0056b3;
        border-radius: 16px;
        padding: 8px 16px;
        margin: 6px 0;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 1rem;
        text-align: left;
        width: fit-content;
        max-width: 70%;
        float: left;
        clear: both;
        border: 1px solid #0056b3;
        box-shadow: 0 2px 8px #0056b320;
    }
    .stTextInput>div>input {
        background-color: #e6f0ff;
        color: #0056b3;
        font-size: 1rem;
        border-radius: 8px;
        border: 1px solid #0056b3;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    .stButton>button {
        background-color: #0056b3;
        color: #fff;
        font-size: 1rem;
        border-radius: 8px;
        border: none;
        font-family: 'Segoe UI', Arial, sans-serif;
        padding: 0.5em 2em;
        margin-top: 0.5em;
    }
    </style>
""", unsafe_allow_html=True)

df = pd.read_csv("carrefour_cleaned_qa_dataset_v2.csv")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

@st.cache_data
def get_embeddings(questions):
    return model.encode(questions, convert_to_tensor=True)

question_embeddings = get_embeddings(df['cleaned_question'].tolist())

def get_answer_bert(user_query):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, question_embeddings)[0]
    idx = similarities.argmax().item()
    return df.iloc[idx]['cleaned_answer']

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Robust header with basket icon and ASK-Carrefour name
st.markdown(
    """
    <div class="carrefour-header">
        <span class="basket-icon">🛒</span>
        <span class="ask-title">ASK-Carrefour</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="ask-subtitle">Your smart assistant for Carrefour supermarket. Type your question below!</div>', unsafe_allow_html=True)

with st.form(key="chat_form"):
    user_input = st.text_input("Type your question:", key="user_input", label_visibility="collapsed")
    submit = st.form_submit_button("Send", use_container_width=True)

if submit and user_input:
    answer = get_answer_bert(user_input)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", answer))

# Display chat flow with improved bubbles
for sender, message in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f'<div class="user-bubble">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{message}</div>', unsafe_allow_html=True)

st.markdown(
    "<hr style='border-top: 1px solid #0056b3;'>"
    "<div style='text-align:center; color:#0056b3; font-size:0.9rem;'>"
    "Powered by BERT & Streamlit | Carrefour Kenya"
    "</div>",
    unsafe_allow_html=True
)