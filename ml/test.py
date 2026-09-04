import numpy as np
from sar_preprocessing import preprocess_image
from sar_inference import load_unet, predict_mask

model = load_unet("unet_model.h5")
chip, offset = preprocess_image("img_0003.jpg")[0]
prob, binary, class_mask = predict_mask(model, chip)

print(np.unique(class_mask, return_counts=True))