# src/features.py

# бинаризация quality → good
# добавление acid_ratio
# масштабирование (StandardScaler)

# Функция engineer_features() — добавляет good и acid_ratio
# Функция prepare_data() — делит на train/test, масштабирует и возвращает данные

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет признаки: бинаризует quality и создаёт acid_ratio.
    """
    df = df.copy()
    df['good'] = (df['quality'] >= 7).astype(int)
    df['acid_ratio'] = df['fixed acidity'] / (df['volatile acidity'] + 1e-5)
    return df

def prepare_data(df: pd.DataFrame):
    """
    Делит данные на X, y, затем train/test, и масштабирует признаки.
    Возвращает X_train_scaled, X_test_scaled, y_train, y_test, scaler.
    """
    X = df.drop(columns=['quality', 'good'])
    y = df['good']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    from data_loader import load_wine_data
    df = load_wine_data()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    print(f"✅ X_train: {X_train.shape}, X_test: {X_test.shape}")
