import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 1. Ruta al directorio de logs (../../logs respecto a banana_net/utils)
LOG_DIR = Path(__file__).resolve().parents[3] / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 2. Nombre del archivo de log
LOG_FILE = LOG_DIR / "project.log"

# 3. Formato estándar
LOG_FORMAT = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"

# 4. Nivel por defecto (puede sobreescribirse con la variable de entorno LOG_LEVEL)
DEFAULT_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Crea y retorna un logger con:
      - Handler rotatorio en archivo (max 5 archivos de 5 MB cada uno)
      - Handler a consola
      - Formateo unificado
    """
    logger = logging.getLogger(name)
    logger.setLevel(DEFAULT_LEVEL)
    logger.propagate = False  # evita doble registro

    # Si ya tenía handlers, no volver a añadir
    if logger.handlers:
        return logger

    # File handler rotatorio
    fh = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(DEFAULT_LEVEL)
    fh.setFormatter(logging.Formatter(LOG_FORMAT))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(DEFAULT_LEVEL)
    ch.setFormatter(logging.Formatter(LOG_FORMAT))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# Logger global que pueden importar todos los módulos
logger = setup_logger("banana_net")