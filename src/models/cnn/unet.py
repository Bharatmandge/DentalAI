import torch.nn as nn
from monai.networks.nets import UNet


def get_3d_unet(
    in_channels: int = 1, 
    out_channels: int = 1, 
    features: tuple = (16, 32, 64, 128, 256),
    dropout: float = 0.2
) -> nn.Module:
    """
    Instantiates a 3D Convolutional Neural Network (UNet) optimized for medical imaging.
    Uses residual units to improve gradient flow during training.
    
    Args:
        in_channels (int): Number of input modalities (1 for standard CT/CBCT).
        out_channels (int): Number of target classes (1 for binary Teeth segmentation).
        features (tuple): The number of convolution filters at each depth layer.
        dropout (float): Dropout probability to prevent overfitting.
        
    Returns:
        nn.Module: The compiled PyTorch UNet model.
    """

    strides = (2,) * (len(features) - 1)

    model = UNet(
        spatial_dims=3, 
        in_channels=in_channels,
        out_channels=out_channels,
        channels=features,
        strides=strides, 
        num_res_units=2,
        norm="batch",
        dropout=dropout
    )
    return model

    