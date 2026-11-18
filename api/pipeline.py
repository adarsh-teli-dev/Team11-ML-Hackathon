"""
pipeline.py - Map-matching inference pipeline module.

Designed for Role D (API & Deployment).
Provides a clean abstraction to load:
  1. Data (track, nodes, arcs, route)
  2. Trained classifier (from Role B)
  3. HMM / Viterbi Decoder (from Role C)

Includes a robust baseline/dummy fallback so the API & UI work immediately
before Role B and Role C push their production weights and decoder.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

logger = logging.getLogger("pathsnap.pipeline")

# Default paths - can be overridden via environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("PATHSNAP_DATA_DIR", BASE_DIR.parent / "Student folder" / "student_test"))
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def load_track(track_path: Path) -> np.ndarray:
    """Load GPS track: longitude, latitude, timestamp."""
    data = []
    with open(track_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                lon = float(parts[0])
                lat = float(parts[1])
                ts = float(parts[2]) if len(parts) > 2 else 0.0
                data.append([lon, lat, ts])
    return np.array(data, dtype=np.float64)


def load_nodes(nodes_path: Path) -> np.ndarray:
    """Load road network nodes: longitude, latitude. 0-based index = node_id."""
    data = []
    with open(nodes_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                data.append([float(parts[0]), float(parts[1])])
    return np.array(data, dtype=np.float64)


def load_arcs(arcs_path: Path) -> np.ndarray:
    """Load road segments: from_node_id, to_node_id. 0-based index = arc_id."""
    data = []
    with open(arcs_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                data.append([int(parts[0]), int(parts[1])])
    return np.array(data, dtype=np.int32)


def load_ground_truth_route(route_path: Path) -> Optional[List[int]]:
    """
    Load ground-truth reference route (arc IDs).
    STRICT SECURITY NOTE: Kept server-side only for accuracy computation.
    """
    if not route_path.exists():
        return None
    arcs = []
    with open(route_path, "r", encoding="utf-8") as f:
        for line in f:
            val = line.strip()
            if val:
                arcs.append(int(val))
    return arcs


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine formula in meters between two (lon, lat) points."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


class MapMatchingPipeline:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.classifier = None
        self.hmm_decoder = None
        self._load_production_models()

    def _load_production_models(self):
        """Attempts to load Role B model and Role C decoder from artifacts/."""
        model_path = ARTIFACTS_DIR / "final_model.joblib"
        hmm_path = ARTIFACTS_DIR / "hmm_config.json"

        if model_path.exists():
            try:
                import joblib
                self.classifier = joblib.load(model_path)
                logger.info(f"Loaded production classifier from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load classifier: {e}")

        if hmm_path.exists():
            try:
                logger.info(f"Found HMM configuration at {hmm_path}")
            except Exception as e:
                logger.warning(f"Could not load HMM: {e}")

    def list_available_trips(self) -> List[str]:
        """List the 10 student/evaluator test trips present in the data folder."""
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist!")
            return []
        
        trips = [
            d.name for d in self.data_dir.iterdir()
            if d.is_dir() and (d / f"{d.name}.track").exists()
        ]
        return sorted(trips)

    def _run_baseline_decoder(self, track: np.ndarray, nodes: np.ndarray, arcs: np.ndarray) -> List[int]:
        """
        Fast spatial baseline matcher (nearest arc midpoint) used when production
        Role B + Role C models are being trained.
        Guarantees the API can be developed, tested, and demoed anytime.
        """
        # Compute midpoints for a subset of nearby arcs or sample of arcs
        num_arcs = len(arcs)
        num_obs = len(track)

        # To keep baseline lightning-fast, calculate midpoints
        from_nodes = arcs[:, 0]
        to_nodes = arcs[:, 1]
        
        # Valid node checks
        valid_mask = (from_nodes < len(nodes)) & (to_nodes < len(nodes))
        valid_arc_indices = np.where(valid_mask)[0]
        
        midpoints_lon = (nodes[from_nodes[valid_mask], 0] + nodes[to_nodes[valid_mask], 0]) / 2.0
        midpoints_lat = (nodes[from_nodes[valid_mask], 1] + nodes[to_nodes[valid_mask], 1]) / 2.0
        
        # Match each GPS point to nearest arc midpoint
        predicted_arcs: List[int] = []
        for i in range(num_obs):
            gps_lon, gps_lat = track[i, 0], track[i, 1]
            # Euclidean approximation for initial rapid search
            d2 = (midpoints_lon - gps_lon)**2 + (midpoints_lat - gps_lat)**2
            nearest_idx = np.argmin(d2)
            predicted_arcs.append(int(valid_arc_indices[nearest_idx]))

        return predicted_arcs

    def evaluate_trip(self, trip_id: str) -> Dict[str, Any]:
        """
        Execute prediction pipeline for trip_id, compute metrics server-side,
        and package response for the UI.
        """
        trip_folder = self.data_dir / trip_id
        if not trip_folder.exists():
            raise FileNotFoundError(f"Trip directory {trip_folder} not found.")

        track_file = trip_folder / f"{trip_id}.track"
        nodes_file = trip_folder / f"{trip_id}.nodes"
        arcs_file = trip_folder / f"{trip_id}.arcs"
        route_file = trip_folder / f"{trip_id}.route"

        # Load inputs
        start_time = time.perf_counter()
        track = load_track(track_file)
        nodes = load_nodes(nodes_file)
        arcs = load_arcs(arcs_file)
        ground_truth = load_ground_truth_route(route_file)

        # Predict
        if self.classifier is not None and self.hmm_decoder is not None:
            # When Role B & C code is integrated:
            # predicted_route = self.hmm_decoder.decode(...)
            predicted_route = self._run_baseline_decoder(track, nodes, arcs)
        else:
            predicted_route = self._run_baseline_decoder(track, nodes, arcs)

        inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Compute Metrics Server-Side (Ground Truth never exposed)
        metrics: Dict[str, Any] = {
            "inference_time_ms": inference_time_ms,
            "observations_count": len(track),
            "predicted_route_length": len(predicted_route),
        }

        # Route continuity calculation:
        # Check percentage of transitions (arc_i -> arc_{i+1}) that are connected
        continuity_matches = 0
        total_transitions = max(1, len(predicted_route) - 1)
        for i in range(total_transitions):
            curr_arc = predicted_route[i]
            next_arc = predicted_route[i + 1]
            if curr_arc == next_arc:
                continuity_matches += 1
            else:
                curr_to = arcs[curr_arc][1] if curr_arc < len(arcs) else -1
                next_from = arcs[next_arc][0] if next_arc < len(arcs) else -2
                if curr_to == next_from:
                    continuity_matches += 1

        metrics["route_continuity"] = round(float(continuity_matches / total_transitions), 4)

        if ground_truth:
            metrics["has_ground_truth"] = True
            gt_set = set(ground_truth)
            
            # Pointwise matching: how many predicted arcs appear in reference route
            matches = sum(1 for a in predicted_route if a in gt_set)
            metrics["top1_accuracy"] = round(float(matches / max(1, len(predicted_route))), 4)
            
            # Sequence level comparison
            min_len = min(len(predicted_route), len(ground_truth))
            exact_pos_matches = sum(1 for i in range(min_len) if predicted_route[i] == ground_truth[i])
            metrics["exact_step_accuracy"] = round(float(exact_pos_matches / max(1, min_len)), 4)
        else:
            metrics["has_ground_truth"] = False
            metrics["top1_accuracy"] = None

        # Prepare downsampled GPS points for UI rendering (e.g. max 500 points for responsiveness)
        step = max(1, len(track) // 300)
        gps_display = [
            {"lat": float(track[i, 1]), "lon": float(track[i, 0]), "step": i}
            for i in range(0, len(track), step)
        ]

        # Prepare predicted route line geometry (lat/lon coordinates of arcs)
        route_coords = []
        for arc_id in predicted_route[::max(1, len(predicted_route) // 400)]:
            if arc_id < len(arcs):
                u, v = arcs[arc_id]
                if u < len(nodes) and v < len(nodes):
                    route_coords.append([float(nodes[u, 1]), float(nodes[u, 0])])
                    route_coords.append([float(nodes[v, 1]), float(nodes[v, 0])])

        return {
            "trip_id": trip_id,
            "metrics": metrics,
            "predicted_route": predicted_route,
            "gps_display": gps_display,
            "route_coords": route_coords
        }


# Global pipeline instance
pipeline_instance = MapMatchingPipeline()
