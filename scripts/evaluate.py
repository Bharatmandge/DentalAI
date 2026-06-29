import os 
import yaml
import torch 
from torch.utils.data import DataLoader

from src.models.registry import create_model

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


class DummyEvalDataset(torch.utils.data.Dataset):
    def __init__(self, length: int = 5):
        self.length = length

    def __len__(self) -> int:
        return self.length
    def __getitem__(self, idx: int) -> dict:
        return {
            "image": torch.randn(1, 96, 96, 96),
            "label": (torch.rand(1, 96, 96, 96) > 0.85).float()
        } 

def run_evaluation() -> None:
    # 1. Setup Configuration and Hardware Environments
    config = load_config("configs/train_unet.yaml")
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Starting Evaluation on device: {device}")

    # 2. Instantiate Model and Load Weights
    model = create_model(config["model"]["name"], config["model"])
    weight_path = os.path.join(config["training"]["save_dir"], "best_model.pth")
    
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"[Success] Loaded trained weights from: {weight_path}")
    else:
        print(f"[Warning] Checked weights at '{weight_path}' but none were found.")
        print("Proceeding with randomized weights for pipeline architecture testing...")

    model = model.to(device)
    model.eval()

    # 3. Initialize Verification Dataloader
    val_dataset = DummyEvalDataset(length=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Metric accumulators
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0

    # 4. Deterministic Evaluation Loop
    print("\n--- Running Inference Across Validation Scans ---")
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Raw output logits from the model
            outputs = model(inputs)
            
            # Apply binarization thresholds
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Fundamental Voxel-Wise Confusion Matrix Calculations
            tp = (preds * labels).sum()
            fp = (preds * (1.0 - labels)).sum()
            fn = ((1.0 - preds) * labels).sum()
            tn = ((1.0 - preds) * (1.0 - labels)).sum()

            # Avoid division by zero anomalies using an epsilon safety buffer
            epsilon = 1e-6
            
            # Mathematical Metric Implementations
            dice = (2.0 * tp) / (2.0 * tp + fp + fn + epsilon)
            iou = tp / (tp + fp + fn + epsilon)
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)

            # Accumulate scores
            total_dice += dice.item()
            total_iou += iou.item()
            total_precision += precision.item()
            total_recall += recall.item()

            print(f"Scan {idx + 1} -> Dice: {dice.item():.4f} | IoU: {iou.item():.4f} | Precision: {precision.item():.4f} | Recall: {recall.item():.4f}")

    # 5. Compute Final Aggregates
    num_scans = len(val_loader)
    final_metrics = {
        "Mean_Dice": total_dice / num_scans,
        "Mean_IoU": total_iou / num_scans,
        "Mean_Precision": total_precision / num_scans,
        "Mean_Recall": total_recall / num_scans,
    }

    # 6. Generate Text Report Asset
    report_path = os.path.join(config["training"]["log_dir"], "evaluation_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w") as report_file:
        report_file.write("=========================================\n")
        report_file.write("      CLINICAL EVALUATION REPORT         \n")
        report_file.write("=========================================\n")
        for metric_name, value in final_metrics.items():
            report_file.write(f"{metric_name:20}: {value:.4f}\n")
            
    print("\n=========================================")
    print(f"[Success] Evaluation Complete.")
    print(f"Report saved to: {report_path}")
    print("=========================================")


if __name__ == "__main__":
    run_evaluation() 