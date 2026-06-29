# --- scripts/visualize.py ---
import argparse
import sys
import os

# Append project root directory to path to ensure clean modular imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.visualizer import generate_3d_mesh_html

def main():
    parser = argparse.ArgumentParser(description="Generate an interactive 3D HTML visualization from a NIfTI mask.")
    parser.add_argument("--mask", type=str, required=True, help="Path to the input .nii.gz mask file.")
    parser.add_argument("--output", type=str, default="data/processed/view_3d.html", help="Path to save the generated HTML.")
    
    args = parser.parse_args()

    try:
        generate_3d_mesh_html(mask_path=args.mask, output_html_path=args.output)
        print(f"\n[Done] Double-click the file at '{args.output}' to view it in your browser!")
    except Exception as e:
        print(f"[Execution Error] {str(e)}")


if __name__ == "__main__":
    main()