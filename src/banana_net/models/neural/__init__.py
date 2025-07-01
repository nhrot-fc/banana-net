"""
Bloques básicos para construir modelos neuronales.
"""
from .conv_block import ConvBlock
from .res_block import ResBlock
from .yolo_head import YOLOHead

# Componentes disponibles
__all__ = ['ConvBlock', 'ResBlock', 'YOLOHead']
