"""
main.py - PathSnap FastAPI Backend (Role D: Deployment & API)

Run on Laptop 1:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import time
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import pipeline_instance

# Setup live console logging for Laptop 1 demonstration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pathsnap.api")

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Team 11 Map-Matching Inference Service for Evaluator Demo",
    version="1.0.0",
)

# Enable CORS for two-laptop evaluation setup (Laptop 2 UI -> Laptop 1 API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """
    Mandatory Evaluator Feature:
    Live request logging shown on Laptop 1 console during evaluation.
    """
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path
    start_time = time.perf_counter()

    logger.info(f"--> Incoming {method} request from [{client_ip}] to {path}")

    try:
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            f"<-- Completed {method} {path} | Status: {response.status_code} | Latency: {duration_ms:.2f}ms"
        )
        return response
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error(f"<-- FAILED {method} {path} | Error: {exc} | Latency: {duration_ms:.2f}ms")
        raise exc


class TripListResponse(BaseModel):
    trips: List[str]
    count: int
    data_source: str


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "pathsnap-api", "team": "Team 11"}


@app.get("/api/trips", response_model=TripListResponse)
def get_trips():
    """
    Returns the list of 10 evaluation trips available on the server.
    """
    trips = pipeline_instance.list_available_trips()
    logger.info(f"Serving {len(trips)} available trips to client")
    return {
        "trips": trips,
        "count": len(trips),
        "data_source": str(pipeline_instance.data_dir),
    }


@app.post("/api/predict/{trip_id}")
def predict_trip(trip_id: str):
    """
    Main evaluation endpoint:
    1. Loads the selected trip's GPS track and road network.
    2. Runs map-matching inference.
    3. Calculates accuracy & continuity on server (never exposing ground truth).
    4. Returns predicted route and metrics to UI.
    """
    logger.info(f"Starting inference pipeline for Trip ID: [{trip_id}]")
    try:
        result = pipeline_instance.evaluate_trip(trip_id)
        logger.info(
            f"Trip [{trip_id}] evaluated successfully: "
            f"Top-1 Accuracy={result['metrics'].get('top1_accuracy')} | "
            f"Continuity={result['metrics'].get('route_continuity')} | "
            f"Latency={result['metrics'].get('inference_time_ms')}ms"
        )
        return result
    except FileNotFoundError as e:
        logger.warning(f"Trip not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Inference error on trip [{trip_id}]: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
