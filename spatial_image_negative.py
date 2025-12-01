from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def create_negative_image(image_path):
    img = Image.open(image_path).convert('L')
    original_pixels = np.array(img)
    height, width = original_pixels.shape
    negative_pixels = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            original_value = original_pixels[y, x]
            negative_value = 255 - original_value
            negative_pixels[y, x] = negative_value
            
    return original_pixels, negative_pixels
image_file_path = 'image_98c879.png'
original, negative = create_negative_image(image_file_path)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(negative, cmap='gray')
axes[1].set_title('Negative Image')
axes[1].axis('off')
plt.show()