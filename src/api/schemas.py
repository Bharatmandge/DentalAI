# --- src/api/schemas.py ---
from pydantic import BaseModel

class HealthResponse(BaseModel):
    """Schema for the API health check endpoint."""
    status: str
    ai_model_loaded: bool  # Renamed from model_loaded to prevent Pydantic namespace conflict
    device: str

class PredictionResponse(BaseModel):
    """Schema for a successful prediction return."""
    message: str
    mask_download_url: str