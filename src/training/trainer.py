import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        metric_fn: callable,
        device: torch.device,
        config: dict
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.device = device
        
        # Hyperparameters
        self.epochs = config["training"]["epochs"]
        self.patience = config["training"]["patience"]
        self.save_dir = config["training"]["save_dir"]
        
        # Tools
        self.scaler = GradScaler()
        self.writer = SummaryWriter(log_dir=os.path.join(config["training"]["log_dir"], config["experiment_name"]))
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # State
        self.best_val_metric = 0.0
        self.epochs_without_improvement = 0

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
        
        for batch in progress_bar:
            inputs = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return epoch_loss / len(self.train_loader)

    def validate_epoch(self, epoch: int) -> float:
        self.model.eval()
        
        # Reset metric before validation
        if hasattr(self.metric_fn, 'reset'):
            self.metric_fn.reset()
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]")
        
        with torch.no_grad():
            for batch in progress_bar:
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                with autocast():
                    outputs = self.model(inputs)
                    # DiceMetric works best with probabilities or binary masks
                    predictions = (torch.sigmoid(outputs) > 0.5).float()
                    
                    # Accumulate (do not aggregate inside loop)
                    self.metric_fn(y_pred=predictions, y=labels)
        
        # Aggregate after all batches
        val_metric = self.metric_fn.aggregate()
        
        if isinstance(val_metric, torch.Tensor):
            if val_metric.numel() > 1:
                val_metric = val_metric.mean().item()
            else:
                val_metric = val_metric.item()
        else:
            val_metric = float(val_metric)
        
        # Reset for next time
        if hasattr(self.metric_fn, 'reset'):
            self.metric_fn.reset()
        
        return val_metric

    def run(self):
        print(f"--- Starting Training for {self.epochs} Epochs ---")
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_metric = self.validate_epoch(epoch)
            
            # Log to TensorBoard
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Metric/Validation_Dice", val_metric, epoch)
            
            print(f"Epoch {epoch} Summary -> Train Loss: {train_loss:.4f} | Val Dice: {val_metric:.4f}")
            
            # Early Stopping and Model Saving
            if val_metric > self.best_val_metric:
                print(f"Validation metric improved from {self.best_val_metric:.4f} to {val_metric:.4f}. Saving model.")
                self.best_val_metric = val_metric
                self.epochs_without_improvement = 0
                
                save_path = os.path.join(self.save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement} epoch(s).")
                
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}. Training halted.")
                    break
                    
        self.writer.close()
        print("--- Training Complete ---")