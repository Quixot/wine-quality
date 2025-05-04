# 🍷 Wine Quality Prediction

Простой проект машинного обучения для предсказания качества вина на основе химических признаков.

## 📁 Структура проекта

```
wine_quality_project/
├── data/                  # CSV-файл с данными (winequality-red.csv)
├── models/                # Сохранённые модель и scaler
├── notebooks/             # Ноутбуки для анализа и EDA (опционально)
├── src/                   # Исходный код (загрузка, обработка, обучение)
│   ├── data_loader.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
├── app/                   # Веб-интерфейс на Gradio
│   └── gradio_app.py
├── README.md              # Инструкция по проекту
├── requirements.txt       # Зависимости
```

## 🚀 Запуск проекта

### 1. Установите зависимости
```bash
pip install -r requirements.txt
```

### 2. Скачайте датасет
Скачайте `winequality-red.csv` отсюда:
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

Сохраните файл в папку `data/`

### 3. Обучите модель
```bash
python src/train.py
```

После обучения в папке `models/` появятся `.pkl` файлы модели и scaler.

### 4. Запустите веб-интерфейс
```bash
python app/gradio_app.py
```

Интерфейс откроется в браузере. Введите параметры вина и получите результат 🧪 или 🍷.

## 🧠 Используемые алгоритмы
- Logistic Regression
- Random Forest
- Gradient Boosting (лучший по F1-score для "хорошего" вина)

## 📚 Использованные библиотеки
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- joblib
- gradio

---

Проект разработан для учебных целей — понять весь pipeline: от EDA до деплоя модели.