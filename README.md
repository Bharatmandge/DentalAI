# 🦷 DentalVision AI
**Enterprise-Grade 3D CBCT Segmentation Pipeline**

DentalVision AI is an end-to-end machine learning platform designed to perform automated binary segmentation (Teeth vs. Background) on 3D Cone Beam Computed Tomography (CBCT) medical scans. 

Built as a production-ready engineering assessment, this project emphasizes robust software architecture, reproducibility, mathematical post-processing, and modern web deployment over simple notebook-based prototyping.

---

## 🚀 Features
* **Deterministic 3D Preprocessing:** Automated Isotropic voxel resampling (1x1x1 mm) and clinical Hounsfield Unit (HU) windowing.
* **Stochastic Augmentation:** On-the-fly 3D rotations, zoom, and Gaussian noise injection via MONAI.
* **Deep Learning Engine:** 3D Residual UNet utilizing Automatic Mixed Precision (AMP) for memory-efficient training.
* **Mathematical Post-Processing:** Morphological closing, 3D hole filling, and small-object noise removal.
* **Sliding Window Inference:** Capable of processing infinitely large NIfTI volumes without GPU memory overflow.
* **FastAPI Backend:** Asynchronous REST API with Pydantic schema validation and singleton model loading.
* **React/Vite Frontend:** Responsive, dark-themed medical dashboard with asynchronous state management.
* **Interactive 3D Visualization:** Marching Cubes algorithm implementation for browser-based 3D mesh rendering.

---

## 🛠️ Technology Stack
* **Machine Learning:** PyTorch, MONAI
* **Medical Image Processing:** SimpleITK, NiBabel, Scikit-Image, SciPy
* **Backend API:** FastAPI, Uvicorn, Pydantic
* **Frontend UI:** React, TypeScript, Vite, Tailwind CSS, Axios
* **Visualization:** Plotly, Marching Cubes

---

## 📂 Architecture & Project Structure
```text
dentalvision_ai/
├── configs/            # YAML hyperparameter configurations
├── data/               # Raw and processed .nii.gz scans (Gitignored)
├── frontend/           # React SPA Dashboard
├── scripts/            # CLI Entry points (train, evaluate, infer, visualize)
├── src/                
│   ├── api/            # FastAPI routes and schemas
│   ├── data/           # Preprocessing and DataLoaders
│   ├── inference/      # Sliding window predictor and morphology cleaning
│   ├── models/         # 3D UNet and Factory Registry
│   ├── training/       # Object-oriented training loop with Early Stopping
│   └── visualization/  # Plotly 3D HTML generators
└── tests/              # System validation