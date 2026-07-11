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


DATASET_ROOT: Path = Path(get_env("DATASET_ROOT"))

MIN_NORMAL_RR_INTERVAL_SEC: float = float(get_env("MIN_NORMAL_RR_INTERVAL_SEC"))
MAX_NORMAL_RR_INTERVAL_SEC: float = float(get_env("MAX_NORMAL_RR_INTERVAL_SEC"))
NORMAL_SEGMENT_DURATION_SEC: float = float(get_env("NORMAL_SEGMENT_DURATION_SEC"))
