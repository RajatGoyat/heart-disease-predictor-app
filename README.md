# 🫀 Heart Disease Prediction App

A simple, interactive Machine Learning web app built with **Streamlit** that predicts the likelihood of heart disease based on user input. The model is trained using the [UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).

## 🚀 Live Demo

🔗 [Click here to use the app](https://heart-disease-predictor-app-kejwjwkbvqjsuk269x226h.streamlit.app)

---

## 📌 Features

- Easy-to-use web interface
- Predicts risk of heart disease using a logistic regression model
- Built with `Streamlit`, `scikit-learn`, and `pandas`
- Takes input features like:
  - Age
  - Cholesterol
  - Resting blood pressure
  - Max heart rate
  - ST depression (oldpeak)
  - Chest pain type, etc.

---

## 🧠 ML Model

- Trained using `LogisticRegression` from `scikit-learn`
- Input features were selected based on clinical relevance
- Model is saved using `joblib` for quick inference

---

## 📁 Folder Structure
📁 ML_App_Deployment/
├── streamlit_app.py
├── model.pkl
├── feature_names.pkl
├── heart_disease_data.csv (optional, if you reference it in app)
├── requirements.txt
└── README.md (optional, but recommended)

