"""SpillTrace Day 1 SAR inference with reusable DeepLab model loading.

Run through the FastAPI backend so generated artifacts are written to
backend/day1_output_results and can be served from /artifacts.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import rasterio
import rasterio.features
import torch
import torch.nn.functional as F
from rasterio.transform import from_origin
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

try:
    from .seg_models import ResNet50DeepLabV3Plus
except ImportError:  # Allows: python ml/day1_inference.py
    from seg_models import ResNet50DeepLabV3Plus


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMAGE_PATH = "test1.tiff"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_WEIGHTS = PROJECT_ROOT / "ml" / "oil_spill_seg_resnet_50_deeplab_v3+_80.pt"

# Keep this aligned with backend/app/main.py's /artifacts static mount.
OUTPUT_DIR = Path(
    os.getenv(
        "SPILLTRACE_ARTIFACTS_DIR",
        PROJECT_ROOT / "backend" / "day1_output_results",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TILE_SIZE = 1024
OVERLAP = 256
OIL_CLASS_INDEX = 1
THRESHOLD = 0.5
DATASET_MEAN = 0.5185
DATASET_STD = 0.197

# These values persist for the life of one backend process. They ensure a
# second uncached detection does not reload DeepLab and its checkpoint.
_MODEL = None
_MODEL_DEVICE = None

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_pytorch_model(weights_path: str | Path, device: torch.device):
    """Create DeepLabV3+ and load the trained checkpoint."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"DeepLab checkpoint not found: {weights_path}")

    model = ResNet50DeepLabV3Plus(num_classes=5, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded DeepLabV3+ checkpoint from {weights_path} on {device}.")
    return model


def get_model():
    """Load the model once per process, then return the reusable instance."""
    global _MODEL, _MODEL_DEVICE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _MODEL is None or _MODEL_DEVICE != str(device):
        _MODEL = load_pytorch_model(MODEL_WEIGHTS, device)
        _MODEL_DEVICE = str(device)

    return _MODEL, device


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def preprocess_tile(tile_array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Standardize a HWC tile and return a 1-image CHW tensor."""
    tile = (tile_array - DATASET_MEAN) / DATASET_STD
    tile = np.transpose(tile, (2, 0, 1))
    return torch.from_numpy(tile).unsqueeze(0).to(device)


def clean_mask(binary_mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Remove connected components smaller than ``min_area`` pixels."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros_like(binary_mask)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 1
    return cleaned


def _geo_reference_if_missing(transform, crs, scene_id: str, file_path: str):
    """Supply documented demo coordinates only when an input has no CRS."""
    if not (transform.is_identity or crs is None):
        return transform, crs

    scene_marker = f"{scene_id} {file_path}".upper()
    if "TEST3" in scene_marker:
        print("No georeferencing found; using Arabian Sea demo coordinates.")
        return from_origin(70.5, 19.5, 0.0001, 0.0001), "EPSG:4326"

    print("No georeferencing found; using Gulf of Mexico demo coordinates.")
    return from_origin(-89.7, 28.85, 0.00003, 0.00003), "EPSG:4326"


# ---------------------------------------------------------------------------
# Main API entry point
# ---------------------------------------------------------------------------
def process_sar_scene(file_path: str = IMAGE_PATH, scene_id: str = "test1_scene") -> dict:
    """Process one SAR image and return the backend detector contract."""
    if not file_path:
        file_path = IMAGE_PATH

    model, device = get_model()
    print(f"Using device: {device} for image: {file_path}")

    with rasterio.open(file_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
        height = src.height
        width = src.width
        raw_bands = src.read()

    transform, crs = _geo_reference_if_missing(transform, crs, scene_id, file_path)
    meta.update({"transform": transform, "crs": crs})

    if raw_bands.shape[0] >= 3:
        full_image = np.stack([raw_bands[0], raw_bands[1], raw_bands[2]], axis=-1)
    else:
        full_image = np.stack([raw_bands[0], raw_bands[0], raw_bands[0]], axis=-1)

    full_image = full_image.astype(np.float32)
    p_min, p_max = np.percentile(full_image, 1), np.percentile(full_image, 99)
    if p_max > p_min:
        full_image = np.clip(full_image, p_min, p_max)
        full_image = (full_image - p_min) / (p_max - p_min)
 
    stride = TILE_SIZE - OVERLAP
    full_prob = np.zeros((height, width), dtype=np.float32)
    full_mask_accum = np.zeros((height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)
 
    print("Running sliding-window inference with PyTorch...")
    with torch.inference_mode():
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                w_width = min(TILE_SIZE, width - x)
                w_height = min(TILE_SIZE, height - y)
                tile = full_image[y:y + w_height, x:x + w_width, :]

                if w_height < TILE_SIZE or w_width < TILE_SIZE:
                    padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.float32)
                    padded[:w_height, :w_width, :] = tile
                    tile = padded

                prediction = model(preprocess_tile(tile, device))
                probabilities = F.softmax(prediction, dim=1)
                oil_probs = probabilities[0, OIL_CLASS_INDEX].cpu().numpy()
                labels = torch.argmax(probabilities, dim=1)[0].cpu().numpy()
                oil_mask = (labels == OIL_CLASS_INDEX).astype(np.float32)

                full_prob[y:y + w_height, x:x + w_width] += oil_probs[:w_height, :w_width]
                full_mask_accum[y:y + w_height, x:x + w_width] += oil_mask[:w_height, :w_width]
                weight_map[y:y + w_height, x:x + w_width] += 1.0
 
    full_prob = np.divide(full_prob, weight_map, out=np.zeros_like(full_prob), where=weight_map != 0)
    full_mask_accum = np.divide(
        full_mask_accum, weight_map, out=np.zeros_like(full_mask_accum), where=weight_map != 0
    )
    binary_mask = clean_mask((full_mask_accum > THRESHOLD).astype(np.uint8), min_area=50)

    out_mask_tif = str(OUTPUT_DIR / f"{scene_id}_pytorch_mask.tif")
    meta.update({"driver": "GTiff", "count": 1, "dtype": "uint8"})
    with rasterio.open(out_mask_tif, "w", **meta) as dst:
        dst.write(binary_mask * 255, 1)

    out_prob_tif = str(OUTPUT_DIR / f"{scene_id}_pytorch_prob.tif")
    meta.update({"driver": "GTiff", "count": 1, "dtype": "float32"})
    with rasterio.open(out_prob_tif, "w", **meta) as dst:
        dst.write(full_prob, 1)

    polygons = [
        shape(geometry)
        for geometry, value in rasterio.features.shapes(binary_mask, transform=transform)
        if value == 1
    ]

    out_geojson = None
    centroid = None
    if polygons:
        out_geojson = str(OUTPUT_DIR / f"{scene_id}_pytorch_slick.geojson")
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {}, "geometry": mapping(polygon)}
                for polygon in polygons
            ],
        }
        with open(out_geojson, "w", encoding="utf-8") as file:
            json.dump(geojson, file)

        union_geometry = unary_union(polygons)
        centroid = [union_geometry.centroid.x, union_geometry.centroid.y]

    return {
        "status": "COMPLETED",
        "message": "Detection completed successfully.",
        "artifacts": {
            "oil_mask": out_mask_tif,
            "probability_map": out_prob_tif,
            "geojson": out_geojson,
            "metadata_path": None,
        },
        "metadata": {
            "detector_name": "SpillTrace DeepLabV3+ Engine",
            "model_name": "ResNet50DeepLabV3Plus",
            "checkpoint": str(MODEL_WEIGHTS),
            "oil_class_index": OIL_CLASS_INDEX,
            "probability_threshold": THRESHOLD,
            "centroid": centroid,
        },
    }
 
 
if __name__ == "__main__":
    process_sar_scene()
