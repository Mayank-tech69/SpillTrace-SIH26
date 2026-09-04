import os
import cv2
import numpy as np
import rasterio
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape
from scipy.ndimage import binary_fill_holes
import math

# ==========================================
# 1. Configuration
# ==========================================
MASK_PATH = "output_results/test3_pytorch_mask.tif"
PROB_PATH = "output_results/test3_pytorch_prob.tif"
OUTPUT_GEOJSON = "output_results/test3_cleaned_slicks.geojson"

MIN_BLOB_AREA_PX = 100  # Filters out tiny noise specks

# ==========================================
# 2. Morphology Cleanup (Hole Filling + Blob Removal)
# ==========================================
def clean_mask(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    print("Applying morphological cleanup...")
    binary_mask = (binary_mask > 0).astype(np.uint8)

    # A. Fill internal voids/holes completely
    filled_mask = binary_fill_holes(binary_mask).astype(np.uint8)

    # B. Morphological closing to seal boundary fractures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)

    # C. Remove small noise blobs by area threshold
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
    cleaned_mask = np.zeros_like(closed_mask)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == i] = 1
            
    return cleaned_mask

# ==========================================
# 3. Geometry Math (Orientation)
# ==========================================
def calculate_orientation(poly) -> float:
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    if len(coords) < 4: 
        return 0.0
        
    dx1, dy1 = coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]
    dx2, dy2 = coords[2][0] - coords[1][0], coords[2][1] - coords[1][1]
    
    len1, len2 = math.hypot(dx1, dy1), math.hypot(dx2, dy2)
    angle = math.degrees(math.atan2(dy1, dx1)) if len1 >= len2 else math.degrees(math.atan2(dy2, dx2))
    return round(angle % 180, 2)

# ==========================================
# 4. Execution Pipeline
# ==========================================
print(f"Loading mask and probability maps...")
with rasterio.open(MASK_PATH) as src_mask:
    raw_mask = src_mask.read(1)
    transform = src_mask.transform
    crs = src_mask.crs
    img_shape = (src_mask.height, src_mask.width)
    acquisition_time = src_mask.tags().get("ACQUISITION_TIME", "2026-09-02T00:00:00Z")

with rasterio.open(PROB_PATH) as src_prob:
    prob_map = src_prob.read(1)

cleaned_mask = clean_mask(raw_mask, MIN_BLOB_AREA_PX)

print("Extracting geometries, physical attributes, and dynamic confidence scores...")
shapes = rasterio.features.shapes(cleaned_mask, transform=transform)

features = []
slick_id = 1

for geom, val in shapes:
    if val == 1:
        poly = shape(geom)
        
        # Overlay polygon onto probability map to calculate exact mean score
        poly_mask = rasterio.features.geometry_mask(
            [poly], transform=transform, invert=True, out_shape=img_shape
        )
        mean_confidence = float(np.mean(prob_map[poly_mask])) if np.any(poly_mask) else 0.0
        
        # Physical metric conversions (Assuming EPSG:4326 degrees, approx 111.32 km/degree)
        area_deg2 = poly.area
        perimeter_deg = poly.length
        
        area_km2 = area_deg2 * (111.32 ** 2)
        perimeter_m = perimeter_deg * 111320.0
        
        # Compile attributes required for drift analysis and tracking
        features.append({
            "geometry": poly,
            "slick_id": slick_id,
            "area_km2": round(area_km2, 4),
            "perimeter_m": round(perimeter_m, 2),
            "centroid_lon": round(poly.centroid.x, 5),
            "centroid_lat": round(poly.centroid.y, 5),
            "orientation_deg": calculate_orientation(poly),
            "confidence": round(mean_confidence, 4),
            "acquisition_time_utc": acquisition_time,
            "model_status": "experimental",
            "detector_name": "abhishekrs4_HTSM"
        })
        slick_id += 1

if features:
    # 1. Define fallback CRS if rasterio didn't catch one (defaults to WGS84)
    if crs is None:
        crs = "EPSG:4326"
        
    # 2. Initialize GeoDataFrame with the confirmed CRS
    gdf = gpd.GeoDataFrame(features, crs=crs)
    
    # 3. Safely convert to EPSG:4326 if it has a valid CRS
    if gdf.crs is not None and gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    elif gdf.crs is None:
        # If still missing, force assign EPSG:4326 for downstream compatibility
        gdf.set_crs("EPSG:4326", allow_override=True)
        
    gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print(f"Success! {len(features)} clean slicks with complete drift attributes exported to {OUTPUT_GEOJSON}")
else:
    print("No slicks remained after cleanup.")