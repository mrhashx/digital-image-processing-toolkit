from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def apply_log_transformation(image_path):
    img = Image.open(image_path).convert('L')
    original_pixels = np.array(img)
    height, width = original_pixels.shape
    log_transformed_pixels = np.zeros((height, width), dtype=np.uint8)
    max_original_value = np.max(original_pixels)
    if max_original_value == 0:
        c = 0 
    else:
        c = 255 / np.log(1 + max_original_value)     
    for y in range(height):
        for x in range(width):
            r = original_pixels[y, x]
            s = c * np.log(1 + r)
            log_transformed_pixels[y, x] = np.uint8(np.clip(s, 0, 255))      
    return original_pixels, log_transformed_pixels

image_file_path = 'image_6f2226.png' 
original, log_transformed = apply_log_transformation(image_file_path)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original, cmap='gray')
axes[0].set_title('Original Fourier Spectrum')
axes[0].axis('off')
axes[1].imshow(log_transformed, cmap='gray')
axes[1].set_title('Log Transformed Image')
axes[1].axis('off')
plt.show()