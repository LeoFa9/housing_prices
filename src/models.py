import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from xgboost import XGBRegressor

from src.dnn import NeuralNetRegressor
from src.preprocessing import build_preprocessor

MODEL_CLASSES = {
    "lasso": Lasso,
    "ridge": Ridge,
    "elasticnet": ElasticNet,
    "random_forest": RandomForestRegressor,
    "xgboost": XGBRegressor,
    "lightgbm": LGBMRegressor,
    "catboost": CatBoostRegressor,
    "dnn": NeuralNetRegressor,
}

FINAL_ESTIMATOR_CLASSES = {
    "linear_regression": LinearRegression,
    "ridge_cv": RidgeCV,
}


def build_base_model(name: str, cfg: DictConfig) -> Pipeline:
    """Строит одну базовую модель по имени из cfg.models вместе с её препроцессингом"""
    model_cfg = cfg.models[name]
    params = OmegaConf.to_container(model_cfg.params, resolve=True)
    preprocessor = build_preprocessor(scale=model_cfg.scale, dense=model_cfg.dense)
    return make_pipeline(preprocessor, MODEL_CLASSES[name](**params))


def build_final_estimator(name: str, cfg: DictConfig):
    """Строит мета-модель для stacking по имени из cfg.final_estimators

    Для ridge_cv сетка alpha задаётся тремя числами alphas_logspace, как аргументы np.logspace
    """
    params = OmegaConf.to_container(cfg.final_estimators[name].params, resolve=True)
    if "alphas_logspace" in params:
        params["alphas"] = np.logspace(*params.pop("alphas_logspace"))
    return FINAL_ESTIMATOR_CLASSES[name](**params)


def build_model(name: str, cfg: DictConfig):
    """Строит модель или ансамбль по имени

    Сначала ищет одиночную модель в cfg.models, иначе собирает ансамбль из cfg.ensembles
    по описанным в нём base_models
    """
    if name in cfg.models:
        return build_base_model(name, cfg)

    ensemble_cfg = cfg.ensembles[name]
    base_estimators = [(base_name, build_base_model(base_name, cfg)) for base_name in ensemble_cfg.base_models]

    if ensemble_cfg.type == "averaging":
        return VotingRegressor(estimators=base_estimators)

    if ensemble_cfg.type == "stacking":
        inner_folds = KFold(n_splits=ensemble_cfg.cv, shuffle=True, random_state=cfg.validation.random_state)
        return StackingRegressor(
            estimators=base_estimators,
            final_estimator=build_final_estimator(ensemble_cfg.final_estimator, cfg),
            cv=inner_folds,
        )

    raise ValueError(f"Неизвестный тип ансамбля: {ensemble_cfg.type}")
