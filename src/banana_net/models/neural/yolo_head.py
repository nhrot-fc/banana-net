"""
Cabeza de detección tipo YOLO para redes neuronales.
"""
import torch
import torch.nn as nn
from typing import Optional


class YOLOHead(nn.Module):
    """
    A simple YOLO-style detection head.

    This head takes feature maps from a backbone and produces predictions for
    bounding boxes (x, y, w, h), objectness score, and class probabilities
    for each anchor at each grid cell.

    Args:
        in_channels (int): Number of channels in the input feature map from the backbone.
        num_anchors (int): Number of anchors to predict per grid cell.
        num_classes (int): Number of classes to predict.
        intermediate_channels (int, optional): Number of channels for an intermediate
                                               convolutional layer before the final prediction layer.
                                               If None, no intermediate layer is used. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        intermediate_channels: Optional[int] = None,
    ):
        super().__init__()
        
        # Number of outputs per anchor: 4 box coords + 1 objectness + num_classes
        num_outputs_per_anchor = 5 + num_classes
        
        layers = []
        
        # Optional intermediate layer to reduce the number of channels
        if intermediate_channels is not None:
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=intermediate_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            layers.append(nn.BatchNorm2d(intermediate_channels))
            layers.append(nn.LeakyReLU(0.1))
            in_channels = intermediate_channels
        
        # Final prediction layer
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_anchors * num_outputs_per_anchor,
                kernel_size=1,  # 1x1 convolution for final predictions
                stride=1,
                padding=0,
            )
        )
        
        self.head = nn.Sequential(*layers)
        self.num_anchors = num_anchors
        self.num_outputs_per_anchor = num_outputs_per_anchor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the YOLO head.
        
        Args:
            x (torch.Tensor): Input feature map from the backbone,
                             shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor containing predictions for each anchor at each grid cell,
                         shape (batch_size, num_anchors * (5 + num_classes), height, width).
                         This can be reshaped to (batch_size, num_anchors, 5 + num_classes, height, width)
                         for easier access to individual predictions.
        """
        batch_size = x.shape[0]
        output = self.head(x)
        
        # Reshape output for easier access to individual predictions
        # From: (batch_size, num_anchors * (5 + num_classes), height, width)
        # To: (batch_size, height, width, num_anchors, 5 + num_classes)
        height, width = output.shape[2], output.shape[3]
        output = output.view(batch_size, self.num_anchors, self.num_outputs_per_anchor, height, width)
        output = output.permute(0, 3, 4, 1, 2)
        
        return output
