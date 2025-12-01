# -*- coding: utf-8 -*-
"""
A comprehensive toolkit for fundamental image processing operations.

This script combines various image manipulation techniques into a single,
well-documented file for educational and practical use. It includes:
- Point Operations: Negative, Log, Gamma Correction
- Bit-Plane Slicing and Reconstruction
- Spatial Filtering: Convolution, Gaussian Blur, Median Filter
- Noise Generation: Salt-and-Pepper
- Padding Utilities for Convolution

Dependencies:
- Pillow (PIL)
- NumPy
- Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# =============================================================================
# 1. Point Operations (Transformations on Individual Pixels)
# =============================================================================

def create_negative_image(image: np.ndarray) -> np.ndarray:
    """
    Creates the negative of a grayscale image.
    s = L - 1 - r, where L=256.
    
    Args:
        image (np.ndarray): Input grayscale image as a NumPy array.
        
    Returns:
        np.ndarray: The negative image.
    """
    return 255 - image

def apply_log_transformation(image: np.ndarray) -> np.ndarray:
    """
    Applies a logarithmic transformation to enhance dark areas of an image.
    s = c * log(1 + r)
    
    Args:
        image (np.ndarray): Input grayscale image.
        
    Returns:
        np.ndarray: The log-transformed image.
    """
    max_val = np.max(image)
    if max_val == 0:
        c = 0
    else:
        c = 255 / np.log(1 + max_val)
    
    # Vectorized operation is much faster than looping
    log_transformed = c * np.log(1 + image.astype(np.float64))
    return np.clip(log_transformed, 0, 255).astype(np.uint8)

def apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Applies gamma correction to an image.
    s = c * r^gamma
    
    Args:
        image (np.ndarray): Input grayscale image.
        gamma (float): The gamma value. gamma > 1 darkens, gamma < 1 lightens.
        
    Returns:
        np.ndarray: The gamma-corrected image.
    """
    normalized_image = image / 255.0
    corrected_normalized = np.power(normalized_image, gamma)
    return (corrected_normalized * 255).astype(np.uint8)

# =============================================================================
# 2. Bit-Plane Slicing
# =============================================================================

def extract_bit_planes(image: np.ndarray) -> list:
    """
    Extracts all 8 bit-planes from a grayscale image.
    
    Args:
        image (np.ndarray): Input grayscale image.
        
    Returns:
        list: A list of 8 NumPy arrays, where each array is a bit-plane (0-7).
    """
    bit_planes = []
    for i in range(8):
        # Extract the i-th bit and scale it to 255 for visibility
        plane = np.bitwise_and(np.right_shift(image, i), 1)
        bit_planes.append(plane * 255)
    return bit_planes

def reconstruct_from_bit_planes(bit_planes: list) -> np.ndarray:
    """
    Reconstructs an image from a list of bit-planes.
    
    Args:
        bit_planes (list): A list of 8 bit-plane images (0 or 255 values).
        
    Returns:
        np.ndarray: The reconstructed grayscale image.
    """
    reconstructed_image = np.zeros(bit_planes[0].shape, dtype=np.uint8)
    for i, plane in enumerate(bit_planes):
        # Convert plane back to 0s and 1s and apply the correct bit weight
        reconstructed_image += (plane // 255) * (2**i)
    return reconstructed_image

# =============================================================================
# 3. Spatial Filtering and Convolution
# =============================================================================

def create_gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
    """
    Creates a 2D Gaussian kernel.
    
    Args:
        size (int): The size of the kernel (e.g., 5 for a 5x5 kernel). Must be odd.
        sigma (float): The standard deviation of the Gaussian distribution.
        
    Returns:
        np.ndarray: The normalized Gaussian kernel.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

def manual_convolve(image: np.ndarray, kernel: np.ndarray, padding_mode: str = 'reflect') -> np.ndarray:
    """
    Performs 2D convolution with specified padding.
    
    Args:
        image (np.ndarray): Input grayscale image.
        kernel (np.ndarray): The convolution kernel.
        padding_mode (str): Padding mode. Options: 'zero', 'reflect', 'symmetric'.
        
    Returns:
        np.ndarray: The convolved image.
    """
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    if padding_mode == 'zero':
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    else:
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=padding_mode)
        
    output_image = np.zeros_like(image, dtype=np.float64)

    # Note: This is a slow, educational implementation.
    # For performance, use libraries like scipy.signal.convolve2d
    for y in range(image_h):
        for x in range(image_w):
            # Region of Interest
            roi = padded_image[y : y + kernel_h, x : x + kernel_w]
            output_pixel = np.sum(roi * kernel)
            output_image[y, x] = output_pixel
            
    return np.clip(output_image, 0, 255).astype(np.uint8)

def manual_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Applies a median filter to an image, effective for salt-and-pepper noise.
    
    Args:
        image (np.ndarray): Input grayscale image.
        kernel_size (int): The size of the neighborhood window (e.g., 3 for 3x3).
        
    Returns:
        np.ndarray: The median-filtered image.
    """
    image_h, image_w = image.shape
    pad_size = kernel_size // 2

    padded_image = np.pad(image, pad_size, mode='reflect')
    output_image = np.zeros_like(image)

    for y in range(image_h):
        for x in range(image_w):
            neighborhood = padded_image[y : y + kernel_size, x : x + kernel_size]
            median_value = np.median(neighborhood)
            output_image[y, x] = median_value
            
    return output_image.astype(np.uint8)

# =============================================================================
# 4. Noise Generation
# =============================================================================

def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.05) -> np.ndarray:
    """
    Adds salt-and-pepper noise to an image.
    
    Args:
        image (np.ndarray): Input grayscale image.
        amount (float): The proportion of pixels to be affected by noise.
        
    Returns:
        np.ndarray: The noisy image.
    """
    output_image = np.copy(image)
    num_salt = int(np.ceil(amount * image.size * 0.5))
    num_pepper = int(np.ceil(amount * image.size * 0.5))
    
    # Salt noise (white pixels)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    output_image[tuple(salt_coords)] = 255

    # Pepper noise (black pixels)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    output_image[tuple(pepper_coords)] = 0
    
    return output_image

# =============================================================================
# Main Execution Block (Example Usage)
# =============================================================================

if __name__ == "__main__":
    # You need to have an image file named 'sample_image.png' in the same directory
    # or provide the full path to your image.
    try:
        image_path = 'sample_image.png'  # <--- CHANGE THIS TO YOUR IMAGE FILE
        img_pil = Image.open(image_path).convert('L')
        original_image = np.array(img_pil)
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        print("Please create a dummy grayscale image named 'sample_image.png' to run the demo.")
        # Create a dummy image if not found
        original_image = np.zeros((200, 200), dtype=np.uint8)
        original_image[50:150, 50:150] = 128
        Image.fromarray(original_image).save('sample_image.png')
        print("A 'sample_image.png' has been created for demonstration.")

    # --- Demo 1: Point Operations ---
    negative_img = create_negative_image(original_image)
    log_img = apply_log_transformation(original_image)
    gamma_img = apply_gamma_correction(original_image, gamma=0.4)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Point Operations', fontsize=16)
    axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original')
    axes[1].imshow(negative_img, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Negative')
    axes[2].imshow(log_img, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Log Transform')
    axes[3].imshow(gamma_img, cmap='gray', vmin=0, vmax=255)
    axes[3].set_title('Gamma (γ=0.4)')
    for ax in axes: ax.axis('off')
    plt.show()

    # --- Demo 2: Bit-Plane Slicing ---
    planes = extract_bit_planes(original_image)
    reconstructed_top4 = reconstruct_from_bit_planes(planes[:4] + [np.zeros_like(p) for p in planes[4:]])
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('Bit-Planes (1 to 8)', fontsize=16)
    for i, ax in enumerate(axes.flat):
        ax.imshow(planes[i], cmap='gray')
        ax.set_title(f'Bit Plane {i+1}')
        ax.axis('off')
    plt.show()

    # --- Demo 3: Noise and Filtering ---
    noisy_img = add_salt_pepper_noise(original_image, amount=0.1)
    gauss_kernel = create_gaussian_kernel(size=7, sigma=2.0)
    gaussian_filtered = manual_convolve(noisy_img, gauss_kernel, padding_mode='reflect')
    median_filtered = manual_median_filter(noisy_img, kernel_size=5)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Noise Reduction Comparison', fontsize=16)
    axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original')
    axes[1].imshow(noisy_img, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Salt & Pepper Noise')
    axes[2].imshow(gaussian_filtered, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Gaussian Filter')
    axes[3].imshow(median_filtered, cmap='gray', vmin=0, vmax=255)
    axes[3].set_title('Median Filter')
    for ax in axes: ax.axis('off')
    plt.show()