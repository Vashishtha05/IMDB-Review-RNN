# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb # type: ignore
from tensorflow.keras.preprocessing import sequence # type: ignore
from tensorflow.keras.models import load_model # type: ignore

import streamlit as st

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------
# Styling (Professional Dashboard)
# ---------------------------------------------------
st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(59,130,246,0.12), transparent 40%),
        radial-gradient(circle at 85% 80%, rgba(168,85,247,0.12), transparent 40%),
        linear-gradient(120deg,#f8fafc,#eef2f7);
}

.title {
    font-size:38px;
    font-weight:700;
    text-align:center;
    color:#0f172a;
}

.subtitle {
    text-align:center;
    color:#475569;
    margin-bottom:30px;
}

.card {
    background:white;
    padding:25px;
    border-radius:14px;
    box-shadow:0 10px 20px rgba(0,0,0,0.05);
}

.result-positive {
    background:#dcfce7;
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-size:22px;
    color:#166534;
    font-weight:600;
}

.result-negative {
    background:#fee2e2;
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-size:22px;
    color:#991b1b;
    font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Load Dataset Index + Model
# ---------------------------------------------------
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
model = load_model('simple_rnn_imdb.h5')

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.markdown('<div class="title">🎬 IMDB Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning RNN Model for Movie Review Classification</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# Layout
# ---------------------------------------------------
col1, col2 = st.columns([1,1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    user_input = st.text_area(
        "Enter Movie Review",
        height=200,
        placeholder="Type your review here..."
    )

    predict_btn = st.button("Classify Sentiment", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# Prediction Section
# ---------------------------------------------------
with col2:

    if predict_btn and user_input.strip() != "":

        preprocessed_input = preprocess_text(user_input)

        prediction = model.predict(preprocessed_input)
        score = float(prediction[0][0])

        sentiment = "Positive" if score > 0.5 else "Negative"

        st.progress(score)

        if sentiment == "Positive":
            st.markdown(
                f'<div class="result-positive">😊 Positive Review<br>Confidence: {score:.2f}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-negative">😔 Negative Review<br>Confidence: {score:.2f}</div>',
                unsafe_allow_html=True
            )

    elif predict_btn:
        st.warning("Please enter a movie review first.")

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")
st.caption("RNN Sentiment Analysis • TensorFlow • Streamlit UI")
