import argparse
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from models import build_model
from preprocessing import load_features
from validation import cross_validate_model, fit_and_predict


def evaluate_all(cfg: DictConfig, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """model: all - прогоняет по CV каждую модель и каждый ансамбль из конфига и печатает
    сравнительную таблицу, аналог итоговых таблиц из ноутбуков. Без сабмита, для сабмита
    нужно выбрать одну конкретную модель в model"""
    scores_by_name = {}
    for name in list(cfg.models.keys()) + list(cfg.ensembles.keys()):
        scores = cross_validate_model(
            build_model(name, cfg),
            X_train,
            y_train,
            n_splits=cfg.validation.n_splits,
            random_state=cfg.validation.random_state,
        )
        scores_by_name[name] = scores
        print(f"{name}: RMSE {scores.mean():.4f} +- {scores.std():.4f}", flush=True)

    summary = pd.DataFrame(
        {
            "rmse": {name: scores.mean() for name, scores in scores_by_name.items()},
            "std": {name: scores.std() for name, scores in scores_by_name.items()},
        }
    ).sort_values("rmse")
    print("\n" + summary.round(4).to_string())


def main(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)

    X_train, y_train, X_test, test_ids = load_features(cfg.data.train_path, cfg.data.test_path)

    if cfg.model == "all":
        evaluate_all(cfg, X_train, y_train)
        return

    scores = cross_validate_model(
        build_model(cfg.model, cfg),
        X_train,
        y_train,
        n_splits=cfg.validation.n_splits,
        random_state=cfg.validation.random_state,
    )
    print(f"{cfg.model}: CV RMSE {scores.mean():.4f} +- {scores.std():.4f}")

    prices = fit_and_predict(build_model(cfg.model, cfg), X_train, y_train, X_test)
    submission_dir = Path(cfg.data.submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)
    submission_path = submission_dir / f"{cfg.model}_submission.csv"
    pd.DataFrame({"Id": test_ids, "SalePrice": prices}).to_csv(submission_path, index=False)
    print(f"Сабмит сохранён в {submission_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
