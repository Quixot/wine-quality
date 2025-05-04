# src/data_loader.py
import pandas as pd

def load_wine_data(path: str = "data/winequality-red.csv") -> pd.DataFrame:
    """
    Загружает датасет вина из CSV и возвращает DataFrame.
    """
    df = pd.read_csv(path, sep=';')
    return df

def check_missing(df: pd.DataFrame) -> pd.Series:
    """
    Возвращает количество пропущенных значений по каждому столбцу.
    """
    return df.isnull().sum()

if __name__ == "__main__":
    df = load_wine_data()
    print(df.head())
    print("\n🔍 Пропуски:\n", check_missing(df))
