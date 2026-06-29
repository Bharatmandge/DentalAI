import yaml
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import DataLoader

# Import our custom modules
from src.models.registry import create_model
from src.training.trainer import Trainer
from monai.utils import set_determinism

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    config = load_config("configs/train_unet.yaml")
    set_determinism(seed=config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing Datasets and DataLoaders...")
    
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, length):
            self.length = length
        def __len__(self):
            return self.length
        def __getitem__(self, idx):
            return {
                "image": torch.randn(1, 96, 96, 96), 
                "label": (torch.rand(1, 96, 96, 96) > 0.8).float()
            }

    train_loader = DataLoader(DummyDataset(10), batch_size=config["data"]["batch_size"], shuffle=True)
    val_loader = DataLoader(DummyDataset(4), batch_size=config["data"]["batch_size"], shuffle=False)

    model = create_model(config["model"]["name"], config["model"])
    
    loss_fn = DiceLoss(sigmoid=True, squared_pred=True)
    
    # Best practice for binary segmentation
    metric_fn = DiceMetric(include_background=False, reduction="mean")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["training"]["learning_rate"], 
        weight_decay=config["training"]["weight_decay"]
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        device=device,
        config=config
    )
    
    trainer.run()

if __name__ == "__main__":
    main()