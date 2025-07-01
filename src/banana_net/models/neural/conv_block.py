"""
Bloque convolucional básico.
"""
import torch
import torch.nn as nn
from typing import Type, Union, List, Tuple, Optional


class ConvBlock(nn.Module):
    """
    A standard convolutional block consisting of a Convolutional layer,
    Batch Normalization (optional), and an activation function (optional).

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolving kernel.
        stride (Union[int, Tuple[int, int]], optional): Stride of the convolution. Defaults to 1.
        padding (Union[int, Tuple[int, int]], optional): Zero-padding added to both sides of the input. Defaults to 0.
        dilation (Union[int, Tuple[int, int]], optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        use_batch_norm (bool, optional): If True, adds a BatchNorm2d layer. Defaults to True.
        activation_fn (Optional[Type[nn.Module]], optional): Type of activation function to use (e.g., nn.ReLU, nn.LeakyReLU).
                                                             If None, no activation is applied. Defaults to nn.ReLU.
        activation_params (Optional[dict], optional): Parameters to pass to the activation function constructor.
                                                      Defaults to None (e.g., {'negative_slope': 0.1} for LeakyReLU).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        use_batch_norm: bool = True,
        activation_fn: Optional[Type[nn.Module]] = nn.ReLU,
        activation_params: Optional[dict] = None,
    ):
        super().__init__()
        layers: List[nn.Module] = []

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=(
                    bias if not use_batch_norm else False
                ),  # Bias is redundant if BatchNorm is used
            )
        )

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        if activation_fn:
            if activation_params:
                layers.append(activation_fn(**activation_params))
            else:
                layers.append(activation_fn())

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolutional block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out).
        """
        return self.block(x)
