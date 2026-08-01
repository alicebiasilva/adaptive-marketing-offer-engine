from pathlib import Path
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / "config" / "settings.toml", "rb") as f:
    settings = tomllib.load(f)

RAW_DATA_PATH = PROJECT_ROOT / settings["paths"]["raw_data"]
PROCESSED_DATA_PATH = PROJECT_ROOT / settings["paths"]["processed_data"]