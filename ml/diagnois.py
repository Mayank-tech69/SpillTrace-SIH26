import os
import cv2
import json
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from seg_models import ResNet50DeepLabV3Plus

# Configuration
IMAGE_PATH = "test3.tiff"
MODEL_WEIGHTS = "oil_spill_seg_resnet_50_deeplab_v3%2B_80.pt"
OUTPUT_DIR = "./audit_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_MEAN = 0.5185
DATASET_STD = 0.197
TILE_SIZE = 1024
OVERLAP = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load 5-Class Model
model = ResNet50DeepLabV3Plus(num_classes=5, pretrained=False)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
model.to(device)
model.eval()

with rasterio.open(IMAGE_PATH) as src:
    height = src.height
    width = src.width
    raw_bands = src.read()

if raw_bands.shape[0] >= 3:
    full_image = np.stack([raw_bands[0], raw_bands[1], raw_bands[2]], axis=-1)
else:
    full_image = np.stack([raw_bands[0], raw_bands[0], raw_bands[0]], axis=-1)

# Global Percentile Scaling
full_image = full_image.astype(np.float32)
p_min, p_max = np.percentile(full_image, 1), np.percentile(full_image, 99)
if p_max > p_min:
    full_image = np.clip(full_image, p_min, p_max)
    full_image = (full_image - p_min) / (p_max - p_min)

stride = TILE_SIZE - OVERLAP
logits_accumulator = np.zeros((height, width, 5), dtype=np.float32)
weight_map = np.zeros((height, width), dtype=np.float32)

print("Executing sliding-window inference...")
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

            t_tile = (tile - DATASET_MEAN) / DATASET_STD
            t_tile = np.transpose(t_tile, (2, 0, 1))
            tensor_batch = torch.from_numpy(t_tile).unsqueeze(0).to(device, dtype=torch.float)

            pred_logits = model(tensor_batch)
            pred_logits_np = pred_logits[0].permute(1, 2, 0).cpu().numpy()

            logits_accumulator[y:y + w_height, x:x + w_width, :] += pred_logits_np[:w_height, :w_width, :]
            weight_map[y:y + w_height, x:x + w_width] += 1.0

weight_map_expanded = np.expand_dims(weight_map, axis=-1)
avg_logits = np.divide(logits_accumulator, weight_map_expanded, out=np.zeros_like(logits_accumulator), where=weight_map_expanded != 0)

# Softmax Calculation (Point 3)
exp_logits = np.exp(avg_logits - np.max(avg_logits, axis=-1, keepdims=True))
probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

# Pixelwise Argmax Class Map (Point 1 & 2)
final_class_map = np.argmax(probabilities, axis=-1).astype(np.uint8)

# Class Pixel Counts (Point 3)
class_names = {0: "sea_surface", 1: "oil_spill", 2: "look_alike", 3: "ship", 4: "land"}
total_pixels = height * width
print("\n--- Class Pixel Counts & Ratios ---")
for idx, name in class_names.items():
    count = int(np.sum(final_class_map == idx))
    print(f"Class {idx} ({name}): {count} pixels ({(count/total_pixels)*100:.2f}%)")

# Softmax Probability Statistics for Oil Class (Point 3)
oil_probs = probabilities[:, :, 1]
oil_stats = {
    "mean": float(np.mean(oil_probs)),
    "min": float(np.min(oil_probs)),
    "max": float(np.max(oil_probs)),
    "p50_median": float(np.percentile(oil_probs, 50)),
    "p95": float(np.percentile(oil_probs, 95)),
    "p99": float(np.percentile(oil_probs, 99))
}
print(f"\n--- Oil Class Probability Stats ---\n{json.dumps(oil_stats, indent=2)}")

# 7. Generate Separate Class Masks & Combined Colored Overlay (Point 7)
# Palette (BGR format): sea=Dark Blue, oil=Yellow, look-alike=Orange, ship=Cyan, land=Green
COLOR_PALETTE = {
    0: [139, 0, 0],
    1: [0, 255, 255],
    2: [0, 165, 255],
    3: [255, 255, 0],
    4: [0, 128, 0]
}

combined_overlay = np.zeros((height, width, 3), dtype=np.uint8)
for class_idx, name in class_names.items():
    c_mask = (final_class_map == class_idx).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"mask_{name}.png"), c_mask)
    combined_overlay[final_class_map == class_idx] = COLOR_PALETTE[class_idx]

cv2.imwrite(os.path.join(OUTPUT_DIR, "combined_colored_overlay.png"), combined_overlay)

# Blended SAR Overlay for Alignment Verification
sar_base = cv2.normalize(full_image[:, :, 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
sar_bgr = cv2.cvtColor(sar_base, cv2.COLOR_GRAY2BGR)
blended = cv2.addWeighted(sar_bgr, 0.6, combined_overlay, 0.4, 0)
cv2.imwrite(os.path.join(OUTPUT_DIR, "sar_model_blended_overlay.png"), blended)

# 8. Adaptive-Threshold Baseline Comparison (Point 8)
# 8. Adaptive-Threshold Baseline Comparison (Point 8)
adaptive_baseline = cv2.adaptiveThreshold(
    sar_base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5
)
cv2.imwrite(os.path.join(OUTPUT_DIR, "adaptive_threshold_baseline.png"), adaptive_baseline)

# =========================================================
# DAY 2 FINAL DELIVERABLES: JSONS & EXACT FILE NAMING
# =========================================================
AUDIT_DIR = "./audit_outputs"
os.makedirs(AUDIT_DIR, exist_ok=True)

# 1. Dual-Probability Statistics (Global vs. Masked)
oil_probs_global = probabilities[:, :, 1]
oil_probs_masked = oil_probs_global[final_class_map == 1]

global_stats = {
    "mean": float(np.mean(oil_probs_global)),
    "max": float(np.max(oil_probs_global)),
    "min": float(np.min(oil_probs_global))
}

masked_stats = {
    "mean": float(np.mean(oil_probs_masked)) if len(oil_probs_masked) > 0 else 0.0,
    "max": float(np.max(oil_probs_masked)) if len(oil_probs_masked) > 0 else 0.0,
    "min": float(np.min(oil_probs_masked)) if len(oil_probs_masked) > 0 else 0.0
}

# 2. JSON Exports
# 2. JSON Exports
total_pixels = height * width
counts_dict = {}

# Loop through and calculate both count and percentage for the JSON
for idx, name in {0: "sea_surface", 1: "oil_spill", 2: "look_alike", 3: "ship", 4: "land"}.items():
    count = int(np.sum(final_class_map == idx))
    counts_dict[name] = {
        "count": count,
        "percentage_pct": round((count / total_pixels) * 100, 2)
    }

with open(os.path.join(AUDIT_DIR, "class_pixel_counts.json"), "w") as f:
    json.dump(counts_dict, f, indent=4)

audit_metadata = {
    "model_name": "ResNet50DeepLabV3Plus",
    "checkpoint": MODEL_WEIGHTS,
    "input_channels": 3,
    "classification_method": "pixelwise_argmax",
    "oil_class_index": 1,
    "normalization": {"mean": DATASET_MEAN, "std": DATASET_STD},
    "oil_probability_stats_global": global_stats,
    "oil_probability_stats_masked": masked_stats
}
with open(os.path.join(AUDIT_DIR, "audit_metadata.json"), "w") as f:
    json.dump(audit_metadata, f, indent=4)

# 3. Exact File Naming Image Exports
# 3. Exact File Naming Image Exports
cv2.imwrite(os.path.join(AUDIT_DIR, "raw_sar.png"), sar_base)
cv2.imwrite(os.path.join(AUDIT_DIR, "multiclass_argmax.png"), combined_overlay)

# ---> GENERATE HEATMAP BEFORE SAVING <---
heatmap_uint8 = (oil_probs_global * 255).astype(np.uint8)
heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
cv2.imwrite(os.path.join(AUDIT_DIR, "oil_probability_heatmap.png"), heatmap_colored)

# Individual binary masks
cv2.imwrite(os.path.join(AUDIT_DIR, "sea_surface_mask.png"), (final_class_map == 0).astype(np.uint8) * 255)
cv2.imwrite(os.path.join(AUDIT_DIR, "oil_argmax_mask.png"), (final_class_map == 1).astype(np.uint8) * 255)
cv2.imwrite(os.path.join(AUDIT_DIR, "look_alike_mask.png"), (final_class_map == 2).astype(np.uint8) * 255)
cv2.imwrite(os.path.join(AUDIT_DIR, "ship_mask.png"), (final_class_map == 3).astype(np.uint8) * 255)
cv2.imwrite(os.path.join(AUDIT_DIR, "land_mask.png"), (final_class_map == 4).astype(np.uint8) * 255)

print("Day 2 Audit deliverables fully saved to ml/audit_outputs/!")