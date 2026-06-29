# --- src/visualization/visualizer.py ---
import os
import nibabel as nib
import numpy as np
from skimage import measure
import plotly.graph_objects as go


def generate_3d_mesh_html(mask_path: str, output_html_path: str) -> None:
    """
    Converts a 3D NIfTI binary mask into a smooth, interactive 3D mesh HTML page.
    
    Args:
        mask_path: Path to the input segmentation .nii.gz file.
        output_html_path: Path where the final interactive HTML file will be saved.
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Target mask file not found: {mask_path}")

    print(f"[Visualizer] Loading segmentation mask: {os.path.basename(mask_path)}")
    nifti_data = nib.load(mask_path)
    mask = nifti_data.get_fdata()

    # Safety Check: Ensure the mask actually contains segmented teeth
    if mask.sum() == 0:
        raise ValueError("The provided mask is completely empty. Cannot generate a 3D mesh surface.")

    print("Extracting 3D surface boundaries via Marching Cubes...")
    # level=0.5 finds the exact boundary halfway between background (0) and tooth structure (1)
    verts, faces, normals, values = measure.marching_cubes(mask, level=0.5)

    print("Compiling interactive 3D surface graphics...")
    # Extract structural coordinate vectors for Plotly
    # x, y, z represent spatial coordinates; i, j, k represent triangular vertex connections
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    # Construct the 3D Medical Mesh Object
    mesh3d = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=1.0,
        color="darkturquoise",      # Clean, clinical medical tone
        lighting=dict(
            ambient=0.4,
            diffuse=0.8,
            fresnel=0.2,
            specular=0.6,
            roughness=0.3
        ),
        lightposition=dict(x=100, y=200, z=150)
    )

    # Set up a clean, high-contrast dark room theme for the dentist's interface
    layout = go.Layout(
        title=dict(
            text="DentalVision AI — Interactive 3D Surface Reconstruction",
            font=dict(size=18, color="white"),
            x=0.5
        ),
        paper_bgcolor="rgb(15, 15, 15)",  # Deep slate background
        scene=dict(
            xaxis=dict(visible=False),     # Hide tracking grids for a cleaner look
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data"             # Preserves original physical proportions of the teeth
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig = go.Figure(data=[mesh3d], layout=layout)

    # Export configuration
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    fig.write_html(output_html_path, auto_open=False)
    print(f"[Success] Interactive 3D visualization generated at: {output_html_path}")