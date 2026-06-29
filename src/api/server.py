# --- src/api/server.py ---
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router

def create_app() -> FastAPI:
    """Factory function to build and configure the FastAPI application."""
    app = FastAPI(
        title="DentalVision AI Backend",
        description="RESTful Medical Image Segmentation API",
        version="1.0.0"
    )

    # CORS Configuration: Allows our future React frontend to communicate with this API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins (change to specific IP in production)
        allow_credentials=True,
        allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
        allow_headers=["*"],
    )

    # Attach our defined endpoints
    app.include_router(router)
    return app

# Instantiate the app
app = create_app()


@app.get("/", tags=["Root"]) 
async def root():
    """Root endpoint returning a simple welcome message."""
    return {"message": "DentalVision AI API is running"}

if __name__ == "__main__":
    print("========================================")
    print("   Starting DentalVision AI API...      ")
    print("========================================")
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)