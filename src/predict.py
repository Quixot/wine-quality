# src/predict.py

# Загружает модель и scaler
# Получает на вход словарь признаков
# Возвращает результат: "Хорошее 🍷" или "Обычное 🧪"

import numpy as np
import joblib
import pandas as pd

# Загрузка модели и scaler
model = joblib.load("models/rf_model.pkl")  # или rf_model.pkl, если лучшая - Random Forest
scaler = joblib.load("models/scaler.pkl")

# Порядок признаков (из features.py)
FEATURE_ORDER = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'acid_ratio'
]

def predict_single(input_dict: dict) -> str:
    """
    Принимает словарь признаков, возвращает 'Хорошее' или 'Обычное'.
    """
    values = np.array([input_dict[feat] for feat in FEATURE_ORDER]).reshape(1, -1)
    values_scaled = scaler.transform(values)
    prediction = model.predict(values_scaled)[0]
    return "Хорошее 🍷" if prediction == 1 else "Обычное 🧪"

if __name__ == "__main__":
    sample_input = {
        'fixed acidity': 7.4,
        'volatile acidity': 0.7,
        'citric acid': 0.0,
        'residual sugar': 1.9,
        'chlorides': 0.076,
        'free sulfur dioxide': 11.0,
        'total sulfur dioxide': 34.0,
        'density': 0.9978,
        'pH': 3.51,
        'sulphates': 0.56,
        'alcohol': 9.4,
        'acid_ratio': 7.4 / (0.7 + 1e-5)
    }

    result = predict_single(sample_input)
    print("Результат предсказания:", result)
