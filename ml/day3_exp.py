import os
import cv2
import json
import numpy as np
import rasterio

# --- CONFIGURATION ---
SAR_PATH = "test3.tiff"
PROB_PATH = "output_results/test3_pytorch_prob.tif"
ARGMAX_OIL_PATH = "audit_outputs/oil_argmax_mask.png"
LOOK_ALIKE_PATH = "audit_outputs/look_alike_mask.png"
DAY3_DIR = "day3_outputs"

os.makedirs(DAY3_DIR, exist_ok=True)
MIN_BLOB_AREA = 100

# 1. Load Existing Data
print("Loading Day 2 outputs for Day 3 experiments...")
with rasterio.open(SAR_PATH) as src:
    sar_img = src.read(1)
    sar_base = cv2.normalize(sar_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sar_bgr = cv2.cvtColor(sar_base, cv2.COLOR_GRAY2BGR)

with rasterio.open(PROB_PATH) as src:
    oil_probs = src.read(1)

# Read masks as binary (0 or 1)
argmax_oil_mask = (cv2.imread(ARGMAX_OIL_PATH, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
look_alike_mask = (cv2.imread(LOOK_ALIKE_PATH, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

# ==========================================
# 2. Threshold Experiment (0.2, 0.3, 0.4, 0.5)
# ==========================================
print("Running confidence threshold sweep...")
thresholds = [0.20, 0.30, 0.40, 0.50]
experiment_results = {}

for t in thresholds:
    t_mask = (oil_probs >= t).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(t_mask, connectivity=8)
    
    # Ignore background (label 0)
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else []
    retained_pixels = int(np.sum(t_mask))
    
    experiment_results[f"threshold_{t}"] = {
        "oil_pixels_retained": retained_pixels,
        "connected_component_count": int(num_labels - 1),
        "total_component_area_px": int(np.sum(areas)) if len(areas) > 0 else 0,
        "largest_component_area_px": int(np.max(areas)) if len(areas) > 0 else 0,
        "mean_probability_inside_mask": float(np.mean(oil_probs[t_mask == 1])) if retained_pixels > 0 else 0.0
    }
    
    # Save the raw pre-cleanup mask for this threshold
    cv2.imwrite(os.path.join(DAY3_DIR, f"pre_cleanup_mask_t{t}.png"), t_mask * 255)

with open(os.path.join(DAY3_DIR, "threshold_experiment_stats.json"), "w") as f:
    json.dump(experiment_results, f, indent=4)

# ==========================================
# 3. Apply Morphology to Selected Threshold
# ==========================================
# Documented Rationale: We select 0.30 to preserve faint slick tails, 
# relying on morphological opening to destroy the resulting low-confidence noise.
SELECTED_THRESHOLD = 0.30
print(f"Applying morphology to selected threshold {SELECTED_THRESHOLD}...")

raw_selected_mask = (oil_probs >= SELECTED_THRESHOLD).astype(np.uint8)

# A. Opening (Erosion followed by Dilation) to remove isolated noise specks
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened_mask = cv2.morphologyEx(raw_selected_mask, cv2.MORPH_OPEN, kernel_open)

# B. Closing (Dilation followed by Erosion) to fill small internal gaps without merging distant slicks
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel_close)

# C. Size Filtering (remove components below MIN_BLOB_AREA)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
final_cleaned_mask = np.zeros_like(closed_mask)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA:
        final_cleaned_mask[labels == i] = 1

cv2.imwrite(os.path.join(DAY3_DIR, "post_cleanup_mask_final.png"), final_cleaned_mask * 255)

# ==========================================
# 4. Create Diagnostic Overlay
# ==========================================
print("Generating Day 3 diagnostic overlay...")
diagnostic_overlay = sar_bgr.copy()

# Add Orange transparent look-alike pixels (BGR: 0, 165, 255)
diagnostic_overlay[look_alike_mask == 1] = [0, 165, 255]

# Add Yellow transparent argmax oil pixels (BGR: 0, 255, 255)
# This uses the original argmax map for comparison against our thresholded polygon
diagnostic_overlay[argmax_oil_mask == 1] = [0, 255, 255]

# Blend the transparent layers
blended = cv2.addWeighted(sar_bgr, 0.5, diagnostic_overlay, 0.5, 0)

# Add strict Red outline of final cleaned polygons (BGR: 0, 0, 255)
contours, _ = cv2.findContours(final_cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(blended, contours, -1, (0, 0, 255), 2)

cv2.imwrite(os.path.join(DAY3_DIR, "diagnostic_overlay_master.png"), blended)

print(f"Day 3 deliverables complete! Check the {DAY3_DIR} folder.")

# ==========================================
# 5. Adaptive-Threshold Baseline Comparison
# ==========================================
print("Running adaptive-threshold baseline comparison...")

# Generate the baseline mask
adaptive_baseline = cv2.adaptiveThreshold(
    sar_base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5
)
cv2.imwrite(os.path.join(DAY3_DIR, "adaptive_baseline_mask.png"), adaptive_baseline)

# Create a comparison overlay (Baseline marked in Magenta)
adaptive_overlay = sar_bgr.copy()
adaptive_overlay[adaptive_baseline == 255] = [255, 0, 255] 
baseline_blended = cv2.addWeighted(sar_bgr, 0.6, adaptive_overlay, 0.4, 0)

cv2.imwrite(os.path.join(DAY3_DIR, "adaptive_baseline_overlay.png"), baseline_blended)
print("Day 3 baseline comparison saved!")