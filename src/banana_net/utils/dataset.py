# src/banana_net/utils/dataset.py
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from src.banana_net.models.annotation import Annotation
from src.banana_net.models.enums import Specie
from src.banana_net.utils.logger import logger


class AudioYOLODataset(Dataset):
    """
    Dataset para modelos YOLO adaptados al análisis de audio.
    Procesa archivos de audio y sus anotaciones para crear espectrogramas
    y tensores objetivo compatibles con la arquitectura YOLO.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: str,
        global_absolute_anchors: np.ndarray,  # Anchors in (duration_s, bandwidth_hz)
        sample_rate: int,
        S: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_classes: int = len(Specie),
    ):
        self.df = df
        self.audio_dir = audio_dir
        self.global_absolute_anchors = global_absolute_anchors
        self.sample_rate = sample_rate
        self.S = S
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_classes = num_classes

        if self.df.empty:
            self.unique_audio_files = []
        else:
            self.unique_audio_files = self.df["recording_file"].unique()