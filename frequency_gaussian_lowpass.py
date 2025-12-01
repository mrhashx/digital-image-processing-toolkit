import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_lowpass_filter_practice(image_path, cutoff_frequency):
    """
    Applies a Gaussian Lowpass Filter for practice and learning purposes.

    Args:
        image_path (str): Path to the input image file.
        cutoff_frequency (int): The cutoff frequency (D0), which acts as the
                                standard deviation for the Gaussian function.
    """
    # Step 1: Read the image in grayscale
    try:
        img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_original is None:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"Error: File not found at '{image_path}'.")
        return
    except Exception as e:
        print(f"Error reading the image: {e}")
        return

    f_transform = np.fft.fft2(img_original.astype(np.float32))
    f_transform_shifted = np.fft.fftshift(f_transform)

    rows, cols = img_original.shape
    center_row, center_col = rows // 2, cols // 2

    u, v = np.meshgrid(np.arange(cols), np.arange(rows))

    distance = np.sqrt((u - center_col)**2 + (v - center_row)**2)
    
    # Create the Gaussian mask using the formula
    # H(u,v) = exp(-D(u,v)^2 / (2 * D0^2))
    mask = np.exp(-(distance**2) / (2 * (cutoff_frequency**2)))

    f_transform_filtered = f_transform_shifted * mask

    f_ishift = np.fft.ifftshift(f_transform_filtered)
    img_filtered_complex = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered_complex).astype(np.uint8)

    plt.style.use('seaborn-v0_8-dark')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img_original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'Gaussian Filter Mask (D0 = {cutoff_frequency})')
    axes[1].axis('off')

    axes[2].imshow(img_filtered, cmap='gray')
    axes[2].set_title('Filtered Image')
    axes[2].axis('off')
    
    fig.suptitle('Gaussian Lowpass Filter Practice', fontsize=16)
    plt.show()

IMAGE_PATH = 'lena.jpeg' 

# Change the cutoff frequency (D0) to see its effect
# Smaller number = More blur
# Larger number = Less blur
CUTOFF_FREQUENCY = 10

# Run the function
gaussian_lowpass_filter_practice(IMAGE_PATH, CUTOFF_FREQUENCY)
