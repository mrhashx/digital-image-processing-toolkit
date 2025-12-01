import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def manual_convolve(image, kernel):
    """
    Performs manual 2D convolution using reflect padding.
    """
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        
    output_image = np.zeros_like(image, dtype=np.float64)

    # Iterate over each pixel of the original image
    for y in range(image_h):
        for x in range(image_w):
            # Extract the region of interest
            roi = padded_image[y : y + kernel_h, x : x + kernel_w]
            # Perform element-wise multiplication and sum
            output_pixel = np.sum(roi * kernel)
            output_image[y, x] = output_pixel
            
    return output_image.astype(np.uint8)

def create_gaussian_kernel(size, sigma=1.0):
    """
    Creates a 2D Gaussian kernel.
    """
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

# You can replace this with the path to your own image
image_file_path = 'char.png' 

try:
    img = Image.open(image_file_path).convert('L')
    original_pixels = np.array(img)

    # --- 1. Apply a small Gaussian filter ---
    small_kernel_size = 9
    small_sigma = 2.0
    print(f"Applying small Gaussian filter ({small_kernel_size}x{small_kernel_size})...")
    small_kernel = create_gaussian_kernel(small_kernel_size, sigma=small_sigma)
    lightly_blurred = manual_convolve(original_pixels, small_kernel)
    print("Small filter applied.")

    # --- 2. Apply a large Gaussian filter ---
    large_kernel_size = 41 # This is large and will be slow
    large_sigma = 12.0
    print(f"Applying large Gaussian filter ({large_kernel_size}x{large_kernel_size})... (This will be slow)")
    large_kernel = create_gaussian_kernel(large_kernel_size, sigma=large_sigma)
    heavily_blurred = manual_convolve(original_pixels, large_kernel)
    print("Large filter applied.")

    # --- Display the results for comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('Smoothing Performance vs. Kernel Size', fontsize=16)

    axes[0].imshow(original_pixels, cmap='gray')
    axes[0].set_title('Original Image')

    axes[1].imshow(lightly_blurred, cmap='gray')
    axes[1].set_title(f'Result with Small Kernel ({small_kernel_size}x{small_kernel_size})')

    axes[2].imshow(heavily_blurred, cmap='gray')
    axes[2].set_title(f'Result with Large Kernel ({large_kernel_size}x{large_kernel_size})')

    for ax in axes.flat:
        ax.axis('off')

    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{image_file_path}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")