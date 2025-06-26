from flask import Flask, request, jsonify
import joblib
import numpy as np
from xgboost import Booster, DMatrix
from scipy.sparse import hstack
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from flask_cors import CORS
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Download required NLTK data
nltk.download('punkt')

# Load models and vectorizers at startup
try:
    xgb_model = Booster()
    xgb_model.load_model("Models/xgb_model.json")
    vectorizer = joblib.load("Models/tfidf_vectorizer.pkl")
    lbl_enc = joblib.load("Models/label_encoder.pkl")
    logging.info("Model and vectorizers loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model/vectorizer: {e}")
    raise

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove markdown links
    text = re.sub(r'@\w+', '', text)            # Remove mentions
    text = re.sub(r'[^\w\s]', '', text)         # Remove punctuation
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    stemmed_tokens = ' '.join(stemmer.stem(token) for token in tokens)
    return stemmed_tokens

# Prediction function
def predict_mental_health_status(text):
    preprocessed_text = preprocess_text(text)
    tfidf_features = vectorizer.transform([preprocessed_text])
    num_characters = len(text)
    num_sentences = len(sent_tokenize(text))
    additional_features = np.array([[num_characters, num_sentences]])
    combined_features = hstack([tfidf_features, additional_features])

    dmatrix = DMatrix(combined_features)
    preds = xgb_model.predict(dmatrix)

    # Handle binary or multiclass
    predicted_class = int(round(preds[0])) if preds.ndim == 1 else np.argmax(preds[0])
    predicted_label = lbl_enc.inverse_transform([predicted_class])[0]
    return predicted_label

# API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" in request'}), 400

    user_text = data['text']
    try:
        result = predict_mental_health_status(user_text)
        return jsonify({'user_text': user_text, 'prediction': result})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# Run server
if __name__ == '__main__':
    app.run(debug=True)
