from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "SalePrice"
ID_COLUMN = "Id"

GARAGE_COLUMNS = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
BASEMENT_COLUMNS = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
NO_OBJECT_COLUMNS = ["Alley", "Fence", "PoolQC", "FireplaceQu", "MiscFeature"] + GARAGE_COLUMNS + BASEMENT_COLUMNS
NO_ZERO_AREA_COLUMNS = ["LotFrontage", "LotArea", "1stFlrSF", "GrLivArea"]
DROPPED_COLUMNS = [
    "GarageArea", "GarageYrBlt", "TotRmsAbvGrd", "1stFlrSF",
    "Utilities", "Street", "Condition2", "RoofMatl", "Heating",
]
ORDINAL_QUALITY_COLUMNS = ["ExterQual", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", "FireplaceQu"]
QUALITY_ORDER = {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
OUTLIER_IDS = [524, 1299]


def apply_accepted_preprocessing(df: pd.DataFrame, masvnrtype_path: Path) -> pd.DataFrame:
    """Повторяет предобработку, принятую в baseline_and_preprocessing.ipynb и feature_engineering.ipynb

    masvnrtype_path - csv, из которого взят df, нужен, чтобы отличить категорию None у MasVnrType
    от настоящих пропусков: обычное чтение pandas превращает строку None в NaN
    """
    result = df.copy()
    result[NO_OBJECT_COLUMNS] = result[NO_OBJECT_COLUMNS].fillna("NA")
    result["GarageYrBlt"] = result["GarageYrBlt"].fillna(0)

    raw_masvnrtype = pd.read_csv(masvnrtype_path, keep_default_na=False)["MasVnrType"]
    result["MasVnrType"] = raw_masvnrtype.replace({"NA": np.nan}).values

    result[NO_ZERO_AREA_COLUMNS] = np.log1p(result[NO_ZERO_AREA_COLUMNS])
    result = result.drop(columns=DROPPED_COLUMNS)

    for column in ORDINAL_QUALITY_COLUMNS:
        result[column] = result[column].map(QUALITY_ORDER)

    result["HasPool"] = (result["PoolArea"] > 0).astype(int)
    result["HasMiscFeature"] = (result["MiscFeature"] != "NA").astype(int)
    result = result.drop(columns=["PoolArea", "PoolQC", "MiscFeature", "MiscVal"])

    result["IsRemodeled"] = ((result["YearRemodAdd"] > result["YearBuilt"]) & (result["YearRemodAdd"] > 1950)).astype(
        int
    )
    result["HouseAge"] = result["YrSold"] - result["YearBuilt"]
    return result.drop(columns=["YearBuilt"])


def load_features(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Читает данные и применяет принятую предобработку, из трейна убирает два выброса

    Возвращает признаки трейна, цены трейна, признаки теста и Id теста
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = apply_accepted_preprocessing(train.drop(columns=[TARGET, ID_COLUMN]), Path(train_path))
    X_test = apply_accepted_preprocessing(test.drop(columns=[ID_COLUMN]), Path(test_path))

    keep = ~train[ID_COLUMN].isin(OUTLIER_IDS)
    return (
        X_train[keep].reset_index(drop=True),
        train.loc[keep, TARGET].reset_index(drop=True),
        X_test,
        test[ID_COLUMN],
    )


def build_preprocessor(scale: bool, dense: bool = False) -> ColumnTransformer:
    """Медиана для чисел и мода с one-hot для категорий

    scale добавляет стандартизацию чисел, dense отдаёт one-hot плотным массивом, а не разреженным
    """
    numeric_steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
    return ColumnTransformer(
        [
            ("numeric", make_pipeline(*numeric_steps), make_column_selector(dtype_include="number")),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=not dense),
                ),
                make_column_selector(dtype_exclude="number"),
            ),
        ]
    )
