# src/train.py

# Обучает все 3 модели
# Сравнивает по F1-score для класса 1 (хорошее вино)
# Сохраняет лучшую модель и scaler в models/

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from features import engineer_features, prepare_data
from data_loader import load_wine_data


def train_and_evaluate():
    df = load_wine_data()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "gb": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    best_model_name = None
    best_f1 = 0
    best_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = report['1']['f1-score']
        print(f"=== {name} ===\nF1 (class 1): {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    print(f"\n✅ Лучший: {best_model_name} (F1-score = {best_f1:.3f})")
    joblib.dump(best_model, f"models/{best_model_name}_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")


if __name__ == "__main__":
    train_and_evaluate()
