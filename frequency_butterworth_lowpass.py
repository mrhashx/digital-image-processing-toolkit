import cv2
import numpy as np
import matplotlib.pyplot as plt

def butterworth_lowpass_filter_practice(image_path, cutoff_frequency, filter_order):
    """
    Applies a Butterworth Lowpass Filter for practice and learning purposes.

    Args:
        image_path (str): Path to the input image file.
        cutoff_frequency (int): The cutoff frequency (D0).
        filter_order (int): The order (n) of the filter.
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

    # Step 2: Move to the frequency domain using Fourier Transform
    f_transform = np.fft.fft2(img_original.astype(np.float32))
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Step 3: Create the Butterworth Lowpass Filter mask
    rows, cols = img_original.shape
    center_row, center_col = rows // 2, cols // 2

    # Create a grid of coordinates and calculate distance from the center
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    distance = np.sqrt((u - center_col)**2 + (v - center_row)**2)
    
    # Create the Butterworth mask using the formula
    # H(u,v) = 1 / (1 + (D(u,v) / D0)^(2*n))
    # Add a small epsilon to avoid division by zero if D0 is 0
    epsilon = 1e-8 
    mask = 1 / (1 + (distance / (cutoff_frequency + epsilon))**(2 * filter_order))

    # Step 4: Apply the filter in the frequency domain
    f_transform_filtered = f_transform_shifted * mask

    # Step 5: Return to the spatial domain with Inverse Fourier Transform
    f_ishift = np.fft.ifftshift(f_transform_filtered)
    img_filtered_complex = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered_complex).astype(np.uint8)

    # Step 6: Display the results
    plt.style.use('seaborn-v0_8-dark')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img_original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'Butterworth Mask (D0={cutoff_frequency}, n={filter_order})')
    axes[1].axis('off')

    axes[2].imshow(img_filtered, cmap='gray')
    axes[2].set_title('Filtered Image')
    axes[2].axis('off')
    
    fig.suptitle('Butterworth Lowpass Filter Practice', fontsize=16)
    plt.show()

# --- Change the values here for practice ---

# Enter the path to your desired image
IMAGE_PATH = 'camra.png' 

# Set the cutoff frequency (D0)
CUTOFF_FREQUENCY = 40

# Set the filter order (n). Try values like 1, 2, 5, 20.
FILTER_ORDER = 1

# Run the function
butterworth_lowpass_filter_practice(IMAGE_PATH, CUTOFF_FREQUENCY, FILTER_ORDER)
