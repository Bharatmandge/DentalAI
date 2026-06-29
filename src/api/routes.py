# --- src/api/routes.py ---
import os
import shutil
import yaml
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, RedirectResponse  # Imported RedirectResponse

from src.api.schemas import HealthResponse, PredictionResponse
from src.inference.predictor import DentalSegmenter

router = APIRouter()

# Load config and initialize model ONCE at startup
with open("configs/train_unet.yaml", "r") as f:
    config = yaml.safe_load(f)

print("Loading AI Model into memory for API...")
try:
    segmenter = DentalSegmenter(config)
    AI_MODEL_LOADED = True
except Exception as e:
    print(f"Failed to load model: {e}")
    AI_MODEL_LOADED = False


@router.get("/")
async def root_redirect():
    """Redirects visitors from the blank root page straight to the interactive API Docs."""
    return RedirectResponse(url="/docs")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Checks if the server is alive and the AI model is ready."""
    return HealthResponse(
        status="healthy",
        ai_model_loaded=AI_MODEL_LOADED,
        device=config.get("device", "cpu")
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_scan(file: UploadFile = File(...)):
    """Receives a NIfTI scan, runs the AI, and returns the download link."""
    if not AI_MODEL_LOADED:
        raise HTTPException(status_code=503, detail="AI Model is not loaded on the server.")
    
  
    temp_input_path = f"data/raw/temp_{file.filename}"
    temp_output_filename = f"mask_{file.filename}"
    temp_output_path = f"data/processed/{temp_output_filename}"

    try:
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    try:
        segmenter.predict_scan(input_path=temp_input_path, output_path=temp_output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Inference failed: {str(e)}")
    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

    return PredictionResponse(
        message="Prediction successful.",
        mask_download_url=f"/download/{temp_output_filename}"
    )


@router.get("/download/{filename}")
async def download_mask(filename: str):
    """Serves the generated mask file back to the user."""
    file_path = f"data/processed/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Requested mask file not found.")
    
    return FileResponse(file_path, media_type="application/gzip", filename=filename)