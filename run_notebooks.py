import subprocess
import sys
import time

NOTEBOOKS = [
    "notebooks/eda_and_hypotheses.ipynb",
    "notebooks/baseline_and_preprocessing.ipynb",
    "notebooks/feature_engineering.ipynb",
    "notebooks/classical_models.ipynb",
    "notebooks/dnn.ipynb",
    "notebooks/ensembles.ipynb",
    "notebooks/optuna_tuning.ipynb",
]


def run_notebook(path: str) -> bool:
    """Выполняет один ноутбук на месте тем же интерпретатором, что запустил скрипт"""
    result = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr[-3000:])
    return result.returncode == 0


def main() -> None:
    failed = []
    for path in NOTEBOOKS:
        print(f"Запускаю {path}...", flush=True)
        start = time.time()
        ok = run_notebook(path)
        elapsed = time.time() - start
        print(f"{path}: {'OK' if ok else 'FAILED'} ({elapsed:.0f}s)", flush=True)
        if not ok:
            failed.append(path)

    if failed:
        print(f"\nНе выполнились: {', '.join(failed)}")
        sys.exit(1)
    print("\nВсе ноутбуки выполнены успешно")


if __name__ == "__main__":
    main()
