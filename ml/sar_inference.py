"""
SAR inference for SpillTrace — plain image version (no georeferencing).

Pipeline: image chip -> U-Net -> probability mask -> binary mask ->
cleaned mask -> PNG export.

Matches a model trained on flat PNG/JPG chips (e.g. Kaggle
nabilsherif/oil-spill). For real Sentinel-1 GeoTIFF scenes with a
coordinate reference system, use sar_inference.py (the GeoTIFF
version) instead, which exports GeoJSON polygons in real-world
coordinates rather than plain PNG masks.
"""

import os
import numpy as np
import cv2
import tensorflow as tf

from sar_preprocessing import preprocess_image, preprocess_folder


def load_unet(weights_path):
    return tf.keras.models.load_model(weights_path, compile=False)


def predict_mask(model, chip, spill_class_index=1):
    """
    chip: (256, 256, 3) normalized float32 array.

    Model outputs (256, 256, 5) softmax — 5 classes per the training
    script's COLOR_MAP:
      0 = background/sea, 1 = oil spill, 2 = look-alike,
      3 = ship, 4 = land

    Returns:
      prob_mask   - (256, 256) softmax probability for the spill class
      binary_mask - (256, 256) 1 where spill is the argmax class, else 0
      class_mask  - (256, 256) full per-pixel class index (0-4), useful
                    if you later want to distinguish look-alikes/ships
                    rather than collapsing everything to binary
    """
    batch = np.expand_dims(chip, axis=0)  # (1, 256, 256, 3)
    pred = model.predict(batch, verbose=0)[0]  # (256, 256, 5)

    class_mask = np.argmax(pred, axis=-1)               # (256, 256)
    prob_mask = pred[..., spill_class_index]             # (256, 256)
    binary_mask = (class_mask == spill_class_index).astype(np.uint8)

    return prob_mask, binary_mask, class_mask


def clean_mask(binary_mask, min_area=50):
    """Drop tiny speckle-noise blobs under min_area pixels."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    cleaned = np.zeros_like(binary_mask)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 1
    return cleaned


def mask_to_png(binary_mask, out_path):
    cv2.imwrite(out_path, binary_mask * 255)


def overlay_mask_on_image(image, binary_mask, out_path, color_bgr=(0, 0, 255), alpha=0.4):
    """
    Save a visual QA overlay — original chip with predicted mask
    highlighted in red. Useful for eyeballing whether the model is
    finding real slick shapes or just noise.

    `image` is expected in BGR order (as produced by
    sar_preprocessing_plain.read_image / preprocess_image), matching
    what cv2.imwrite expects directly — no color conversion needed.
    Default highlight color is red in BGR: (0, 0, 255).
    """
    img_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    if img_uint8.ndim == 2:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

    overlay = img_uint8.copy()
    overlay[binary_mask == 1] = color_bgr
    blended = cv2.addWeighted(overlay, alpha, img_uint8, 1 - alpha, 0)
    cv2.imwrite(out_path, blended)


def run_inference_on_image(image_path, weights_path, out_dir, tile_size=256,
                            spill_class_index=1, min_area=50, save_overlay=True):
    """
    Run inference on a single image chip. Saves binary mask PNG and
    optionally a visual overlay for quick QA.
    """
    os.makedirs(out_dir, exist_ok=True)
    model = load_unet(weights_path)
    tiles = preprocess_image(image_path, tile_size=tile_size)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    results = []

    for chip, offset in tiles:
        prob, binary, class_mask = predict_mask(model, chip, spill_class_index=spill_class_index)
        binary = clean_mask(binary, min_area=min_area)

        mask_path = os.path.join(out_dir, f"{base_name}_mask_{offset[0]}_{offset[1]}.png")
        mask_to_png(binary, mask_path)
        results.append({"offset": offset, "mask_path": mask_path, "spill_pixels": int(binary.sum())})

        if save_overlay:
            overlay_path = os.path.join(out_dir, f"{base_name}_overlay_{offset[0]}_{offset[1]}.png")
            overlay_mask_on_image(chip, binary, overlay_path)

    return results


def run_inference_on_folder(folder_path, weights_path, out_dir, tile_size=256,
                             spill_class_index=1, min_area=50, save_overlay=True):
    """Batch inference over every image in a folder."""
    os.makedirs(out_dir, exist_ok=True)
    model = load_unet(weights_path)
    all_results = {}

    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(folder_path, fname)
        results = run_inference_on_image(
            path, weights_path, out_dir,
            tile_size=tile_size, spill_class_index=spill_class_index,
            min_area=min_area, save_overlay=save_overlay,
        )
        all_results[fname] = results
        total_spill_px = sum(r["spill_pixels"] for r in results)
        print(f"{fname}: {total_spill_px} spill pixels detected")

    return all_results


if __name__ == "__main__":
    run_inference_on_image(
        image_path="img_1.png",
        weights_path="unet_model.h5",
        out_dir="output/",
        tile_size=256,
        spill_class_index=1,  # 1 = oil spill, per training COLOR_MAP
        min_area=50,
    )