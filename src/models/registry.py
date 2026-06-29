import torch.nn as nn 
from typing import Dict, Any

from src.models.cnn.unet import get_3d_unet

def create_model(model_name: str, config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to instantiate neural network models dynamically based on string names.
    This allows us to swap architectures just by changing the YAML config file.
    
    Args:
        model_name (str): The identifier of the model (e.g., 'monai_unet').
        config (Dict[str, Any]): Dictionary containing model hyperparameters.
        
    Returns:
        nn.Module: The instantiated PyTorch model.
        
    Raises:
        ValueError: If the requested model_name is not supported.
    """
    
    model_name = model_name.lower()
    
    if model_name == "monai_unet":
        return get_3d_unet(
            in_channels=config.get("in_channels", 1),
            out_channels=config.get("out_channels", 1),
            features=config.get("features", (16, 32, 64, 128, 256)),
            dropout=config.get("dropout", 0.2)
        )
        
    # Example placeholder for future extensions
    elif model_name == "segresnet":
        raise NotImplementedError("SegResNet implementation planned for future release.")
        
    else:
        raise ValueError(f"Model architecture '{model_name}' is not recognized in the registry.")

if __name__ == "__main__":
    import torch

    print("Testing Model Registry instantiation")
    dummy_config = {
        "in_channels": 1,
        "out_channels": 1,
        "features": (16, 32, 64, 128, 256),
        "dropout": 0.1
    }

    test_model = create_model("monai_unet", dummy_config)

    dummy_input = torch.randn(1, 1, 96, 96, 96)

    output = test_model(dummy_input)
    
    print(f"Model instantiated successfully: {test_model.__class__.__name__}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") 
    assert output.shape == dummy_input.shape, "Output dimensions must match input dimensions for segmentation!"
