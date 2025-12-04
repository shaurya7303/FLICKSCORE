import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
import streamlit as st
import time

wordindex = imdb.get_word_index()
reverse_wordindex = {value: key for (key, value) in wordindex.items()}

model = load_model("flick_score.h5")


def preprocess_text(text):
    words = text.lower().split()
    encodedreview = [wordindex.get(word, 2) + 3 for word in words]
    paddedreview = sequence.pad_sequences([encodedreview], maxlen=500)
    return paddedreview


def decode_review(encodedreview):
    return " ".join([reverse_wordindex.get(i - 3, "?") for i in encodedreview])


def predict_sentiment(review):
    preprocessed_review = preprocess_text(review)
    prediction = model.predict(preprocessed_review, verbose=0)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    return sentiment, float(prediction[0][0])



st.set_page_config(
    page_title="FlickScore",
    layout="centered",
)



st.markdown(
    """
    <style>
    /* Global background */
    .stApp {
        background: radial-gradient(circle at top left, #1f2933 0, #111827 45%, #020617 100%);
        color: #e5e7eb;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Reduce default top padding */
    div.block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Main card */
    .main-card {
        background: rgba(15, 23, 42, 0.92);
        padding: 2.5rem 2rem;
        border-radius: 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.3);
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.6);
    }

    /* Title styling */
    .app-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-align: center;
        margin-bottom: 0.3rem;
        background: linear-gradient(120deg, #38bdf8, #a855f7, #f97316);
        -webkit-background-clip: text;
        color: transparent;
    }

    .app-subtitle {
        text-align: center;
        font-size: 0.96rem;
        color: #9ca3af;
        margin-bottom: 1.8rem;
    }

    /* Sentiment badge */
    .sentiment-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .sentiment-positive {
        background: rgba(22, 163, 74, 0.12);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }
    .sentiment-negative {
        background: rgba(220, 38, 38, 0.12);
        color: #fca5a5;
        border: 1px solid rgba(248, 113, 113, 0.35);
    }

    /* Score text */
    .score-text {
        font-size: 0.9rem;
        color: #d1d5db;
        margin-top: 0.3rem;
    }

    /* Text area */
    textarea {
        background-color: rgba(15, 23, 42, 0.9) !important;
        color: #e5e7eb !important;
        border-radius: 0.9rem !important;
        border: 1px solid rgba(148, 163, 184, 0.5) !important;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #38bdf8, #6366f1);
        color: white;
        border-radius: 999px;
        padding: 0.5rem 1.8rem;
        border: none;
        font-weight: 600;
        letter-spacing: 0.04em;
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.45);
        transition: transform 0.08s ease-out, box-shadow 0.08s ease-out, background 0.15s ease-out;
    }
    .stButton>button:hover {
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 18px 40px rgba(37, 99, 235, 0.6);
        background: linear-gradient(135deg, #0ea5e9, #4f46e5);
    }
    .stButton>button:active {
        transform: translateY(1px) scale(0.99);
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.9);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    
    st.markdown('<div class="app-title">FlickScore</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Movie review sentiment analysis powered by deep learning.</div>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([2.1, 1.4])

    
    with left_col:
        st.markdown("####  Try your review")
        user_input = st.text_area(
            "Movie review",
            "An amazing movie with a thrilling plot and stellar performances.",
            height=140,
            label_visibility="collapsed",
        )

        classify_btn = st.button("CLASSIFY REVIEW")

        if classify_btn:
            if not user_input.strip():
                st.warning("Please enter a movie review to analyze.")
            else:
                
                progress_text = "Analyzing sentiment..."
                progress_bar = st.progress(0, text=progress_text)
                for pct in range(0, 101, 8):
                    time.sleep(0.03)
                    progress_bar.progress(pct, text=progress_text)
                time.sleep(0.15)
                progress_bar.empty()

                sentiment, score = predict_sentiment(user_input)

                
                if sentiment == "Positive":
                    badge_class = "sentiment-badge sentiment-positive"
                else:
                    badge_class = "sentiment-badge sentiment-negative"

                st.markdown("---")
                st.markdown("#### 🔍 Result")

                st.markdown(
                    f"""
                    <div class="{badge_class}">
                        <span>{sentiment} review</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="score-text">Prediction score: <b>{score:.3f}</b> (closer to 1 → more confident positive, closer to 0 → more confident negative)</div>',
                    unsafe_allow_html=True,
                )

  
    with right_col:
        st.markdown("####  How it works")
        st.write(
            "- Type any movie review in the box.\n"
            "- Click on **CLASSIFY REVIEW**.\n"
            "- Wait for the analysis and check the predicted sentiment."
        )

        st.markdown("####  Example ideas")
        st.write(
            "- *“The plot was confusing and the acting was terrible.”*\n"
            "- *“A masterpiece with beautiful storytelling and visuals.”*"
        )

    st.markdown("</div>", unsafe_allow_html=True)
