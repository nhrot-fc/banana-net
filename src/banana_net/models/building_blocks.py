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
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Each anchor predicts: 4 bbox coords (tx, ty, tw, th) + 1 objectness score + num_classes probabilities
        self.num_outputs_per_anchor = 4 + 1 + num_classes

        prediction_channels = self.num_anchors * self.num_outputs_per_anchor

        if intermediate_channels:
            self.head_conv = nn.Sequential(
                ConvBlock(
                    in_channels,
                    intermediate_channels,
                    kernel_size=3,
                    padding=1,
                    activation_fn=nn.LeakyReLU,
                    activation_params={"negative_slope": 0.1},
                ),
                nn.Conv2d(intermediate_channels, prediction_channels, kernel_size=1),
            )
        else:
            self.head_conv = nn.Conv2d(in_channels, prediction_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the YOLO head.

        Args:
            x (torch.Tensor): Input feature map of shape (N, C_in, H_grid, W_grid).

        Returns:
            torch.Tensor: Output tensor of shape (N, H_grid, W_grid, num_anchors, 5 + num_classes).
                          The last dimension contains (tx, ty, tw, th, objectness, class_probs...).
        """
        out = self.head_conv(
            x
        )  # Shape: (N, num_anchors * (5 + num_classes), H_grid, W_grid)

        # Reshape to (N, H_grid, W_grid, num_anchors, 5 + num_classes)
        # This makes it easier to interpret and apply loss later.
        N, _, H_grid, W_grid = out.shape
        out = out.permute(
            0, 2, 3, 1
        )  # (N, H_grid, W_grid, num_anchors * (5 + num_classes))
        out = out.reshape(
            N, H_grid, W_grid, self.num_anchors, self.num_outputs_per_anchor
        )

        return out
