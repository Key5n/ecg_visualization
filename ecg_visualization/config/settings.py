import logging
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
ENV_PATH: Path = REPO_ROOT / ".env"

load_dotenv(ENV_PATH)


def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_int_env(name: str) -> int:
    return int(get_env(name))


def get_float_env(name: str) -> float:
    return float(get_env(name))


DATASET_ROOT: Path = Path(get_env("DATASET_ROOT"))

MIN_NORMAL_RR_INTERVAL_SEC: float = get_float_env("MIN_NORMAL_RR_INTERVAL_SEC")
MAX_NORMAL_RR_INTERVAL_SEC: float = get_float_env("MAX_NORMAL_RR_INTERVAL_SEC")
NORMAL_SEGMENT_DURATION_SEC: float = get_float_env("NORMAL_SEGMENT_DURATION_SEC")

DEFAULT_LOG_LEVEL: int = logging.WARNING
LOG_LEVEL: str = get_env("LOG_LEVEL")

ECG_VISUALIZE_WORKERS: int = get_int_env("ECG_VISUALIZE_WORKERS")

OPTUNA_DB_DRIVER: str = get_env("OPTUNA_DB_DRIVER")
OPTUNA_DB_USER: str = get_env("OPTUNA_DB_USER")
OPTUNA_DB_PASSWORD: str = get_env("OPTUNA_DB_PASSWORD")
OPTUNA_DB_HOST: str = get_env("OPTUNA_DB_HOST")
OPTUNA_DB_PORT: str = get_float_env("OPTUNA_DB_PORT")
OPTUNA_DB_NAME: str = get_env("OPTUNA_DB_NAME")
