import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tensorflow.keras.models import load_model
from scipy.ndimage import gaussian_filter

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = r"C:\Users\MAYANK\Desktop\sih-2026\SpillTrace-SIH26\ml\unet_model.h5"
TIFF_PATH = r"C:\Users\MAYANK\Desktop\sih-2026\SpillTrace-SIH26\ml\test1.tiff"   # <-- Replace with your .tiff file path
TILE_SIZE = 256
OIL_CLASS_INDEX = 1                         # Class 1 corresponds to Oil Spill

# Match the 5-class color palette from your training script
COLOR_MAP = [
    [0, 0, 0],       # Class 0: Background / Water (Black)
    [0, 255, 255],   # Class 1: Oil Spill (Cyan)
    [255, 0, 0],     # Class 2: Look-alike / Red
    [153, 76, 0],    # Class 3: Ships / Brown
    [0, 153, 0],     # Class 4: Land / Green
]
scaled_colors = [[c[0] / 255.0, c[1] / 255.0, c[2] / 255.0] for c in COLOR_MAP]
cmap = mcolors.ListedColormap(scaled_colors)


# ==========================================
# 2. LOAD & NORMALIZE TIFF
# ==========================================
def load_tiff(file_path):
    """Loads a GeoTIFF, checks its dynamic range, and converts to 3 channels."""
    with rasterio.open(file_path) as src:
        image = src.read()  # Shape: (Channels, Height, Width)
        transform = src.transform
        crs = src.crs

    # Transpose to standard (Height, Width, Channels)
    image = np.moveaxis(image, 0, -1).astype(np.float32)
    h, w, c = image.shape
    print(f"Loaded TIFF shape: {h}x{w} with {c} band(s). Min value: {np.min(image):.2f}, Max value: {np.max(image):.2f}")

    # Handling SAR dB values (e.g. -30 dB to 0 dB) vs 8-bit standard imagery (0 to 255)
    if np.min(image) < 0:
        # Decibel SAR data normalization
        image = np.clip(image, -30.0, 0.0)
        image = (image + 30.0) / 30.0
    elif np.max(image) > 1.0:
        # Standard 8-bit / scaled data normalization
        image = image / 255.0

    # Ensure 3 channels for U-Net input
    if c == 1:
        image = np.repeat(image, 3, axis=-1)
    elif c == 2:
        image = np.concatenate([image, image[:, :, :1]], axis=-1)
    elif c > 3:
        image = image[:, :, :3]

    return image, transform, crs


# ==========================================
# 3. TILING & INFERENCE PIPELINE
# ==========================================
def create_2d_gaussian_window(size=256, sigma=64):
    """Generates a smooth 2D Gaussian weight mask to favor tile centers."""
    grid = np.zeros((size, size), dtype=np.float32)
    grid[size // 2, size // 2] = 1.0
    gaussian = gaussian_filter(grid, sigma=sigma)
    gaussian /= gaussian.max()
    return np.expand_dims(gaussian, axis=-1)

def predict_clean_tiff(image_array, model, tile_size=256, overlap=128, min_confidence=0.60):
    h, w, _ = image_array.shape
    num_classes = 5
    
    prob_map = np.zeros((h, w, num_classes), dtype=np.float32)
    weight_map = np.zeros((h, w, 1), dtype=np.float32)
    
    window_weight = create_2d_gaussian_window(size=tile_size)
    stride = tile_size - overlap

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            tile = image_array[y:y + tile_size, x:x + tile_size, :]
            tile_h, tile_w, _ = tile.shape
            
            # Zero-pad edges
            if tile_h < tile_size or tile_w < tile_size:
                tile_padded = np.pad(tile, ((0, tile_size - tile_h), (0, tile_size - tile_w), (0, 0)), mode='constant')
            else:
                tile_padded = tile
            
            # Predict softmax probabilities: shape (256, 256, 5)
            tile_input = np.expand_dims(tile_padded, axis=0)
            pred_probs = model.predict(tile_input, verbose=0)[0]
            
            # Apply Gaussian center weighting
            weighted_preds = pred_probs[:tile_h, :tile_w] * window_weight[:tile_h, :tile_w]
            
            prob_map[y:y + tile_h, x:x + tile_w] += weighted_preds
            weight_map[y:y + tile_h, x:x + tile_w] += window_weight[:tile_h, :tile_w]

    # Calculate weighted average probabilities
    avg_probs = prob_map / np.maximum(weight_map, 1e-6)
    
    # 1. Start with everything as Background (Class 0)
    final_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 2. Extract highest predicted class and confidence
    max_class = np.argmax(avg_probs, axis=-1)
    max_prob = np.max(avg_probs, axis=-1)
    
    # 3. Only accept non-background classes if confidence is solid
    confident_mask = (max_prob >= min_confidence) & (max_class != 0)
    final_mask[confident_mask] = max_class[confident_mask]
    
    return final_mask


# ==========================================
# 4. EXECUTE & VISUALIZE
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(TIFF_PATH):
        print(f"File not found: {TIFF_PATH}. Please provide a valid .tif file path.")
    else:
        print("Loading U-Net model...")
        model = load_model(MODEL_PATH)

        # 1. Load image
        img, transform, crs = load_tiff(TIFF_PATH)

        # 2. Predict full scene
        predicted_full_mask = predict_clean_tiff(img, model, TILE_SIZE)

        # 3. Check oil detection stats
        oil_pixel_count = np.sum(predicted_full_mask == OIL_CLASS_INDEX)
        total_pixels = img.shape[0] * img.shape[1]
        coverage_percent = (oil_pixel_count / total_pixels) * 100

        print("\n" + "=" * 40)
        print(f"Total Oil Spill Pixels Detected: {oil_pixel_count}")
        print(f"Scene Coverage: {coverage_percent:.4f}%")
        print("=" * 40)

        # 4. Visualization
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.title("Original SAR Scene")
        plt.imshow(img[:, :, 0], cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("U-Net Multi-Class Segmentation")
        im = plt.imshow(predicted_full_mask, cmap=cmap, vmin=0, vmax=len(COLOR_MAP) - 1, interpolation='none')
        cbar = plt.colorbar(im, ticks=range(len(COLOR_MAP)), fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(['Background', 'Oil Spill', 'Look-alike', 'Ships', 'Land'])
        plt.axis('off')

        plt.tight_layout()
        plt.show()