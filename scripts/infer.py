# --- scripts/infer.py ---
import argparse
import yaml
import sys
import os
# Ensure the project root is in sys.path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.inference.predictor import DentalSegmenter


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(description="Run DentalVision AI Inference on a single NIfTI scan.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .nii.gz file.")
    parser.add_argument("--output", type=str, default="data/processed/prediction.nii.gz", help="Path to save the output mask.")
    # Updated default config path to be relative to project root
    parser.add_argument("--config", type=str, default=os.path.join(project_root, "configs", "train_unet.yaml"), help="Path to configuration file.")
    
    args = parser.parse_args()

    # Initialize Engine
    config = load_config(args.config)
    segmenter = DentalSegmenter(config)

    # Run Prediction
    segmenter.predict_scan(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()