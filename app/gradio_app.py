# app/gradio_app.py

# Строит веб-интерфейс с вводом 12 признаков
# Показывает результат: хорошее или обычное вино
# Запускается через python app/gradio_app.py

import gradio as gr
import joblib
import numpy as np

# Загрузка модели и scaler
model = joblib.load("./models/rf_model.pkl")  # заменить на rf_model.pkl при необходимости
scaler = joblib.load("./models/scaler.pkl")

# Порядок признаков
FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'acid_ratio'
]

# Интерфейсная функция
def predict_wine(*args):
    values = np.array(args).reshape(1, -1)
    values_scaled = scaler.transform(values)
    prediction = model.predict(values_scaled)[0]
    return "Хорошее 🍷" if prediction == 1 else "Обычное 🧪"

# Gradio интерфейс
inputs = [gr.Number(label=feature) for feature in FEATURES]
demo = gr.Interface(fn=predict_wine, inputs=inputs, outputs="text", title="Качество вина")

if __name__ == "__main__":
    demo.launch()
