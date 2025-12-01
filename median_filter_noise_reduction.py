import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# --- توابع کمکی برای نویز و فیلترها (از کدهای قبلی) ---

def add_salt_pepper_noise(image, amount=0.05):
    """
    نویز نمک و فلفل را به تصویر اضافه می‌کند.
    """
    output_image = np.copy(image)
    num_salt = np.ceil(amount * image.size * 0.5)
    num_pepper = np.ceil(amount * image.size * 0.5)
    
    # اضافه کردن نویز نمک (پیکسل‌های سفید)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    output_image[tuple(coords)] = 255

    # اضافه کردن نویز فلفل (پیکسل‌های سیاه)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    output_image[tuple(coords)] = 0
    
    return output_image

def manual_convolve(image, kernel):
    """
    عملیات کانولوشن دو بعدی را به صورت دستی پیاده‌سازی می‌کند (با reflect padding).
    """
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        
    output_image = np.zeros_like(image, dtype=np.float64)

    for y in range(image_h):
        for x in range(image_w):
            roi = padded_image[y : y + kernel_h, x : x + kernel_w]
            output_pixel = np.sum(roi * kernel)
            output_image[y, x] = output_pixel
            
    return output_image.astype(np.uint8)

def create_gaussian_kernel(size, sigma=1.0):

    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

def manual_median_filter(image, kernel_size=3):

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

# --- بخش اصلی برنامه ---

# مسیر فایل تصویر
image_file_path = 'salt.png' 

try:
    img = Image.open(image_file_path).convert('L')
    original_pixels = np.array(img)

    # 1. اضافه کردن نویز نمک و فلفل به تصویر اصلی
    noisy_image = add_salt_pepper_noise(original_pixels, amount=0.1)

    # 2. اعمال فیلتر گوسی با σ=3 (مانند تصویر b در اسلاید)
    gaussian_kernel_size = 19 
    gaussian_sigma = 3.0      
    
    print(f"Applying Gaussian filter with sigma={gaussian_sigma}... (This will be slow)")
    gaussian_kernel = create_gaussian_kernel(gaussian_kernel_size, sigma=gaussian_sigma)
    gaussian_filtered_image = manual_convolve(noisy_image, gaussian_kernel)
    print("Gaussian filter applied.")

    # 3. اعمال فیلتر میانه (مانند تصویر c در اسلاید)
    median_kernel_size = 7 
    
    print(f"Applying Median filter with kernel size={median_kernel_size}... (This will be slow)")
    median_filtered_image = manual_median_filter(noisy_image, kernel_size=median_kernel_size)
    print("Median filter applied.")

    # --- نمایش نتایج برای مقایسه ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('Noise Reduction: Gaussian vs. Median Filtering', fontsize=16)

    axes[0].imshow(noisy_image, cmap='gray')
    axes[0].set_title('Noisy Image (Salt-and-Pepper)') # مشابه (a) در اسلاید

    axes[1].imshow(gaussian_filtered_image, cmap='gray')
    axes[1].set_title(f'Gaussian Filter (σ={gaussian_sigma})') # مشابه (b) در اسلاید

    axes[2].imshow(median_filtered_image, cmap='gray')
    axes[2].set_title(f'Median Filter ({median_kernel_size}x{median_kernel_size})') # مشابه (c) در اسلاید

    for ax in axes.flat:
        ax.axis('off')

    plt.show()

except FileNotFoundError:
    print(f"خطا: فایل '{image_file_path}' پیدا نشد.")
except Exception as e:
    print(f"یک خطای غیرمنتظره رخ داد: {e}")