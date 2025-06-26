# 🧠 Mental Health Text Classification API

This project is a **Natural Language Processing (NLP)**-powered API for predicting mental health conditions from user-submitted text using **machine learning**.

## 📌 Description

The system analyzes user statements and classifies their mental health status into one of seven categories:

* **Normal**
* **Depression**
* **Suicidal**
* **Anxiety**
* **Bipolar**
* **Stress**
* **Personality Disorder**

The backend is powered by:

* **TF-IDF vectorization** for text features
* **XGBoost classifier** for prediction (Accuracy \~81%)
* **Flask** to serve predictions via a RESTful API

The `/predict` endpoint accepts POST requests with a JSON payload containing user text and returns the predicted mental health label.

## 🛠 Features

* Full preprocessing pipeline: cleaning, tokenization, stemming
* Hybrid features: text + numerical (character & sentence count)
* Class balancing with RandomOverSampler
* Model and vectorizer loading at startup
* Cross-Origin support for web integrations (CORS)
* Robust logging and error handling

## 🚀 Deployment

Models are saved in:

* `xgb_model.json`
* `tfidf_vectorizer.pkl`
* `label_encoder.pkl`

Run the Flask app to start serving predictions:

```bash
python app.py
```

## 📊 Dataset

* **Source**: [Kaggle - Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data)
* **Size**: \~52,000 cleaned entries
* **Classes**: 7 mental health conditions

## 📦 Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
python app.py
```

## 📁 API Usage

### Endpoint

```http
POST /predict
```

### Request Body (JSON)

```json
{
  "text": "I feel hopeless and anxious every day."
}
```

### Response

```json
{
  "user_text": "I feel hopeless and anxious every day.",
  "prediction": "Anxiety"
}
```

---

## ✅ Highlights

* Combines NLP with traditional machine learning
* Includes both textual and numerical features
* Supports real-time classification via Flask
* Can be used in mental health monitoring tools, digital therapeutics, or clinical decision support systems
