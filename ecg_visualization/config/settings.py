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


def get_optional_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if not value:
        return default
    return value


def get_optional_int_env(name: str, default: int) -> int:
    return int(get_optional_env(name, str(default)))


def get_optional_float_env(name: str, default: float) -> float:
    return float(get_optional_env(name, str(default)))


DATASET_ROOT: Path = Path(get_optional_env("DATASET_ROOT", "data/raw-datasets"))

MIN_NORMAL_RR_INTERVAL_SEC: float = get_optional_float_env(
    "MIN_NORMAL_RR_INTERVAL_SEC",
    0.6,
)
MAX_NORMAL_RR_INTERVAL_SEC: float = get_optional_float_env(
    "MAX_NORMAL_RR_INTERVAL_SEC",
    1.0,
)
NORMAL_SEGMENT_DURATION_SEC: float = get_optional_float_env(
    "NORMAL_SEGMENT_DURATION_SEC",
    5 * 60,
)
