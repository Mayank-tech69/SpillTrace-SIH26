import os
import numpy as np
import rasterio
from rasterio.windows import Window
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape
import torch
import torch.nn.functional as F

# Import the model architecture from the downloaded repo files
from seg_models import ResNet50DeepLabV3Plus  # Change to ResNet18... if you downloaded the smaller one

# ==========================================
# 1. Configuration
# ==========================================
IMAGE_PATH = "test3.tiff"
MODEL_WEIGHTS = "oil_spill_seg_resnet_50_deeplab_v3%2B_80.pt"  # Update this if your .pt file has a different name
OUTPUT_DIR = "./output_results"

TILE_SIZE = 1024
OVERLAP = 256
OIL_CLASS_INDEX = 1  # 1 = Oil Spill (from EDA dictionary)
THRESHOLD = 0.5      # For overlap averaging

# Normalization stats from image_stats.json
DATASET_MEAN = 0.5185
DATASET_STD = 0.197

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. Load PyTorch Model
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize architecture (5 classes, pretrained=False because we are loading weights)
model = ResNet50DeepLabV3Plus(num_classes=5, pretrained=False)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
model.to(device)
model.eval()
print(f"Successfully loaded DeepLabV3+ weights from {MODEL_WEIGHTS}")

# ==========================================
# 3. Helper: Preprocessing
# ==========================================
def preprocess_tile(tile_array):
    """ Applies Mean/Std standardization and formats for PyTorch. """
    # The tile is ALREADY globally scaled to [0, 1] before this function
    tile = (tile_array - DATASET_MEAN) / DATASET_STD
    tile = np.transpose(tile, (2, 0, 1))
    tensor = torch.from_numpy(tile).unsqueeze(0)
    return tensor.to(device)

# ==========================================
# 4. Inference & Tiling
# ==========================================
from rasterio.transform import from_origin

# ==========================================
# 4. Inference & Tiling
# ==========================================
print(f"Processing image: {IMAGE_PATH}")

with rasterio.open(IMAGE_PATH) as src:
    meta = src.meta.copy()
    transform = src.transform
    crs = src.crs
    height = src.height
    width = src.width
    raw_bands = src.read()

# ---> INJECT DUMMY GPS COORDINATES IF MISSING <---
if transform.is_identity or crs is None:
    print("Warning: No spatial metadata found in input. Injecting dummy GPS coordinates...")
    # Places the image roughly in the Arabian Sea (Longitude 70.5, Latitude 19.5)
    # 0.0001 degrees is approx 10 meters per pixel resolution
    transform = from_origin(70.5, 19.5, 0.0001, 0.0001)
    crs = "EPSG:4326"
    
    # Update the metadata so the output .tif files save with this GPS data!
    meta.update({"transform": transform, "crs": crs})

# Stack into 3 channels
if raw_bands.shape[0] >= 3:
    full_image = np.stack([raw_bands[0], raw_bands[1], raw_bands[2]], axis=-1)
else:
    full_image = np.stack([raw_bands[0], raw_bands[0], raw_bands[0]], axis=-1)

# ---> THE GLOBAL CONTRAST FIX <---
# Clip extreme SAR outliers globally and scale to [0.0, 1.0] once.
# This prevents per-tile stretching and mimics the training dataset's contrast.
full_image = full_image.astype(np.float32)
p_min, p_max = np.percentile(full_image, 1), np.percentile(full_image, 99)

if p_max > p_min:
    full_image = np.clip(full_image, p_min, p_max)
    full_image = (full_image - p_min) / (p_max - p_min)

stride = TILE_SIZE - OVERLAP
# We need TWO arrays now: one for float probabilities, one for the binary mask
full_prob = np.zeros((height, width), dtype=np.float32)
full_mask_accum = np.zeros((height, width), dtype=np.float32)
weight_map = np.zeros((height, width), dtype=np.float32)

print("Running sliding-window inference with PyTorch...")

with torch.no_grad():
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            w_width = min(TILE_SIZE, width - x)
            w_height = min(TILE_SIZE, height - y)

            tile = full_image[y:y + w_height, x:x + w_width, :]

            if w_height < TILE_SIZE or w_width < TILE_SIZE:
                padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.float32)
                padded[:w_height, :w_width, :] = tile
                tile = padded

            tensor_batch = preprocess_tile(tile)

            # Predict
            pred_logits = model(tensor_batch)             
            pred_probs = F.softmax(pred_logits, dim=1)    
            
            # 1. REAL PROBABILITIES: Get the raw Softmax floats for Class 1 (Oil Spill)
            raw_oil_probs = pred_probs[0, OIL_CLASS_INDEX, :, :].cpu().numpy()
            
            # 2. BINARY MASK: Use argmax to see if Class 1 actually "won"
            pred_label = torch.argmax(pred_probs, dim=1) 
            class_mask = pred_label[0].cpu().numpy()
            binary_tile = (class_mask == OIL_CLASS_INDEX).astype(np.float32)

            # Accumulate BOTH separately
            full_prob[y:y + w_height, x:x + w_width] += raw_oil_probs[:w_height, :w_width]
            full_mask_accum[y:y + w_height, x:x + w_width] += binary_tile[:w_height, :w_width]
            weight_map[y:y + w_height, x:x + w_width] += 1.0

# Average overlap probabilities
full_prob = np.divide(full_prob, weight_map, out=np.zeros_like(full_prob), where=weight_map != 0)
full_mask_accum = np.divide(full_mask_accum, weight_map, out=np.zeros_like(full_mask_accum), where=weight_map != 0)

# Create the final binary mask (thresholding the overlaps)
binary_mask = (full_mask_accum > THRESHOLD).astype(np.uint8)

# ==========================================
# 5. Save Outputs (GeoTIFF Mask + GeoJSON)
# ==========================================
base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

# A. Save the Binary Mask (Multiplied by 255 so you can see it!)
out_mask_tif = os.path.join(OUTPUT_DIR, f"{base_name}_pytorch_mask.tif")
meta.update({"driver": "GTiff", "count": 1, "dtype": "uint8"})
with rasterio.open(out_mask_tif, "w", **meta) as dst:
    dst.write(binary_mask * 255, 1)
print(f"Saved binary mask raster to: {out_mask_tif}")

# B. Save the Probability Map (Float32 array of actual confidences)
out_prob_tif = os.path.join(OUTPUT_DIR, f"{base_name}_pytorch_prob.tif")
meta.update({"driver": "GTiff", "count": 1, "dtype": "float32"})
with rasterio.open(out_prob_tif, "w", **meta) as dst:
    dst.write(full_prob, 1)
print(f"Saved probability map raster to: {out_prob_tif}")

# (You can remove the old GeoJSON export block from here, 
# since postprocess.py handles it much better now!)

# Extract Polygons
shapes = rasterio.features.shapes(binary_mask, transform=transform)
polygons = [shape(geom) for geom, val in shapes if val == 1]

if polygons:
    out_geojson = os.path.join(OUTPUT_DIR, f"{base_name}_pytorch_slick.geojson")
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs if crs else "EPSG:4326")
    gdf.to_file(out_geojson, driver="GeoJSON")
    print(f"Extracted {len(polygons)} slick polygons and saved to: {out_geojson}")
else:
    print("No slick detections found.")