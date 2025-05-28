# settings.py (en la raíz del proyecto)
from pathlib import Path

# BASE_DIR → carpeta raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent

# rutas comunes
DATA_DIR    = BASE_DIR / "data"
LOG_DIR     = BASE_DIR / "logs"
CONFIG_DIR  = BASE_DIR / "configs"
MODEL_DIR   = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# crea directorios si no existen
for d in (DATA_DIR, LOG_DIR, CONFIG_DIR, MODEL_DIR, RESULTS_DIR):
    d.mkdir(exist_ok=True)
