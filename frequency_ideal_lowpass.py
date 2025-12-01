import cv2
import numpy as np
import matplotlib.pyplot as plt

def ideal_lowpass_filter_practice(image_path, cutoff_radius):
    """
    Applies an Ideal Lowpass Filter for practice and learning purposes.

    Args:
        image_path (str): Path to the input image file.
        cutoff_radius (int): The cutoff radius (D0) for the filter.
    """

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
    
    mask = np.zeros((rows, cols), np.uint8)

    cv2.circle(mask, (center_col, center_row), cutoff_radius, 1, thickness=-1)


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
    axes[1].set_title(f'Ideal Filter Mask (D0 = {cutoff_radius})')
    axes[1].axis('off')

    axes[2].imshow(img_filtered, cmap='gray')
    axes[2].set_title('Filtered Image')
    axes[2].axis('off')
    
    fig.suptitle('Ideal Lowpass Filter Practice', fontsize=16)
    plt.show()

IMAGE_PATH = 'lena.jpeg' 

# Change the cutoff radius to see its effect
# Smaller number = More blur and stronger Ringing
# Larger number = Less blur and weaker Ringing
CUTOFF_RADIUS = 30

ideal_lowpass_filter_practice(IMAGE_PATH, CUTOFF_RADIUS)

