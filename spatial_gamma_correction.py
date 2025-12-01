from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def apply_gamma_correction(image_path, gamma_values):
    img = Image.open(image_path).convert('L')
    original_pixels = np.array(img)
    normalized_pixels = original_pixels / 255.0
    transformed_images = []
    for gamma in gamma_values:
        corrected_pixels_normalized = np.power(normalized_pixels, gamma)
        corrected_pixels = (corrected_pixels_normalized * 255).astype(np.uint8)
        transformed_images.append(corrected_pixels)
    return original_pixels, transformed_images

image_file_path = 'image_984c7e.png' 
gamma_list = [2.5, 0.4, 0.1]
original, transformed_list = apply_gamma_correction(image_file_path, gamma_list)
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes[0, 0].imshow(original, cmap='gray')
axes[0, 0].set_title('Original Image (γ=1.0)')
axes[0, 1].imshow(transformed_list[0], cmap='gray')
axes[0, 1].set_title(f'Gamma Corrected (γ={gamma_list[0]})')
axes[1, 0].imshow(transformed_list[1], cmap='gray')
axes[1, 0].set_title(f'Gamma Corrected (γ={gamma_list[1]})')
axes[1, 1].imshow(transformed_list[2], cmap='gray')
axes[1, 1].set_title(f'Gamma Corrected (γ={gamma_list[2]})')
for ax in axes.flat:
    ax.axis('off')
plt.tight_layout()
plt.show()