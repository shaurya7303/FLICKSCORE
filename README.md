FlickScore 🎬 the website is { https://flickscore-mzzedumz2zeterxbuzvala.streamlit.app/ }
Movie Review Sentiment Analysis Web App

FlickScore is a Streamlit-based web application that predicts the sentiment (Positive/Negative) of a movie review using a deep learning model trained on the IMDB dataset. It provides a modern UI with a card layout, gradient background, sentiment badges, and a smooth progress animation.

Features
🌐 Web interface built with Streamlit

🧠 Deep learning model loaded from flick_score.h5

📝 Text preprocessing using the IMDB word index

✅ Binary sentiment classification: Positive / Negative

📊 Prediction score (0–1) to indicate confidence

🎨 Custom CSS for:

Radial gradient background

Glassmorphism-style main card

Styled buttons and text area

Colored sentiment badges

⏱ Progress animation while analyzing the review

💡 Right-hand info panel with usage instructions and example reviews

Project Structure
bash
.
├── app.py            # Main Streamlit app
├── flick_score.h5    # Trained Keras/TensorFlow sentiment model
└── README.md         # Project documentation
Requirements
Make sure the following are installed in your environment:

Python 3.8+

TensorFlow / Keras

NumPy

Streamlit

Example requirements.txt:

text
streamlit
numpy
tensorflow
If your model was trained with a specific TensorFlow version, pin that version accordingly.

Getting Started
1. Clone or download the project
bash
git clone <your-repo-url>
cd <your-project-folder>
2. Create and activate a virtual environment (optional but recommended)
bash
python -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate
3. Install dependencies
bash
pip install -r requirements.txt
4. Ensure the model file is present
Place flick_score.h5 in the same directory as app.py.
If the name or path is different, update:

python
model = load_model("flick_score.h5")
accordingly.

5. Run the app
bash
streamlit run app.py
Then open the URL shown in the terminal (usually http://localhost:8501).

How It Works
The app loads the IMDB word index using:

python
from tensorflow.keras.datasets import imdb
wordindex = imdb.get_word_index()
The preprocess_text function:

Lowercases and splits the review into words

Converts each word to its index (wordindex.get(word, 2) + 3)

Pads the sequence to a fixed length (maxlen=500)

The encoded and padded review is passed to the loaded model:

python
prediction = model.predict(preprocessed_review, verbose=0)
Sentiment is decided by a threshold of 0.5:

>= 0.5 → Positive

< 0.5 → Negative

The UI:

Left column: input text area, classify button, result section

Right column: “How it works” and “Example ideas”

UI / UX Details
Custom CSS is injected via st.markdown(..., unsafe_allow_html=True) to:

Style the background and main container

Customize button hover/active states

Style the text area and sentiment badges

A progress bar gives feedback during inference:

python
progress_bar = st.progress(0, text="Analyzing sentiment...")
for pct in range(0, 101, 8):
    time.sleep(0.03)
    progress_bar.progress(pct, text="Analyzing sentiment...")
progress_bar.empty()
The result is shown with a colored badge and a short explanation of the score.

Customization
You can easily customize:

Theme colors: edit the gradient and colors in the <style> block.

Threshold: change the 0.5 cutoff if your model calibration suggests another value.

Model: replace flick_score.h5 with another Keras model (e.g., using embeddings, LSTMs, or transformers) as long as the input shape matches.

Copy: update headings, subtitle, and helper text to match your branding.

Known Limitations
Uses simple word-level tokenization (str.split()), which may not handle punctuation or out-of-vocabulary words as robustly as modern tokenizers.

Assumes the model was trained with the same IMDB word index and padding strategy as used in preprocess_text.

License
Add your chosen license information here (e.g., MIT, Apache 2.0, or proprietary), depending on how you want others to use this project
