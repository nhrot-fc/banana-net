"""
Bloque residual para redes neuronales.
"""
import torch
import torch.nn as nn
from typing import Type, Union, Tuple, Optional
from .conv_block import ConvBlock


class ResBlock(nn.Module):
    """
    A Residual Block, typically consisting of two convolutional blocks and a skip connection.
    The skip connection is a 1x1 convolution if the number of channels or spatial dimensions change.

    Args:
        in_channels (int): Number of input channels.
        intermediate_channels (int): Number of channels in the intermediate convolutional layers.
        stride (Union[int, Tuple[int, int]], optional): Stride for the first convolutional block.
                                                       This affects the output spatial dimensions. Defaults to 1.
        use_batch_norm (bool, optional): If True, ConvBlocks will use BatchNorm. Defaults to True.
        activation_fn (Optional[Type[nn.Module]], optional): Activation function for ConvBlocks. Defaults to nn.ReLU.
        activation_params (Optional[dict], optional): Parameters for the activation function. Defaults to None.
        conv_block_type (Type[ConvBlock], optional): The type of ConvBlock to use. Defaults to ConvBlock.
    """

    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        stride: Union[int, Tuple[int, int]] = 1,
        use_batch_norm: bool = True,
        activation_fn: Optional[Type[nn.Module]] = nn.ReLU,
        activation_params: Optional[dict] = None,
        conv_block_type: Type[ConvBlock] = ConvBlock,
    ):
        super().__init__()

        # The output channels of the ResBlock will be intermediate_channels * expansion factor (if any)
        # For a simple ResBlock, out_channels = intermediate_channels
        out_channels = intermediate_channels

        self.conv1 = conv_block_type(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=3,
            stride=stride,  # Apply stride here if downsampling
            padding=1,
            bias=False,
            use_batch_norm=use_batch_norm,
            activation_fn=activation_fn,
            activation_params=activation_params,
        )
        self.conv2 = conv_block_type(
            in_channels=intermediate_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            use_batch_norm=use_batch_norm,
            activation_fn=None,  # Activation is applied after skip connection
        )

        self.skip_connection: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = conv_block_type(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,  # Match stride for downsampling
                padding=0,
                bias=False,
                use_batch_norm=use_batch_norm,
                activation_fn=None,  # No activation on skip connection's ConvBlock
            )

        self.activation: nn.Module = nn.Identity()
        if activation_fn:
            if activation_params:
                self.activation = activation_fn(**activation_params)
            else:
                self.activation = activation_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Residual Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying residual connection and activation.
        """
        identity = self.skip_connection(x)

        out = self.conv1(x)
        out = self.conv2(out)

        out += identity
        out = self.activation(out)
        return out
