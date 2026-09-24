import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def rmse(log_true: np.ndarray, log_pred: np.ndarray) -> float:
    """Среднеквадратичная ошибка между двумя массивами логарифмов цены"""
    return float(np.sqrt(np.mean((log_true - log_pred) ** 2)))


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42) -> np.ndarray:
    """Считает RMSE на логарифме цены по KFold, возвращает массив скоров по фолдам

    Модель обучается на log1p цены, препроцессинг внутри модели подгоняется заново в каждом фолде
    """
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    log_y = np.log1p(y)
    scores = []

    for fit_idx, valid_idx in folds.split(X):
        fold_model = copy.deepcopy(model)
        fold_model.fit(X.iloc[fit_idx], log_y.iloc[fit_idx])
        prediction = fold_model.predict(X.iloc[valid_idx])
        scores.append(rmse(log_y.iloc[valid_idx].values, prediction))

    return np.array(scores)


def fit_and_predict(model, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    """Обучает модель на всём трейне и возвращает предсказанные цены в долларах для теста"""
    model.fit(X, np.log1p(y))
    return np.expm1(model.predict(X_test))
