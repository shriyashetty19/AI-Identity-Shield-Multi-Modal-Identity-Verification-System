"""Central, relative path config. Nothing here should ever be an absolute
local path - everything is derived from the repo root so the project runs
the same on Colab, this machine, or anyone else's."""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

MIDV2020_DIR = DATA_DIR / "midv2020"
MIDV2020_TAMPER_DIR = DATA_DIR / "midv2020_tamper"
FUNSD_DIR = DATA_DIR / "funsd"
LFW_DIR = DATA_DIR / "lfw"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
