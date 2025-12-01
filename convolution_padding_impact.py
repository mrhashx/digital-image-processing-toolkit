import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def manual_convolve(image, kernel, padding_mode):
    """
    کانولوشن دستی را با یک روش حاشیه‌گذاری مشخص پیاده‌سازی می‌کند.
    """
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
   
    if padding_mode == 'zero':
     
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    else:
       
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=padding_mode)
        
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


image_file_path = 'char.png' 

try:
    img = Image.open(image_file_path).convert('L')
    original_pixels = np.array(img)

    kernel_size = 51
    sigma_val = 15.0
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma=sigma_val)

  
    ##print("Applying Zero Padding... (This will be slow)")
    zero_padded_result = manual_convolve(original_pixels, gaussian_kernel, padding_mode='zero')

    #print("Applying Mirror Padding (reflect)... (This will be slow)")
    mirror_padded_result = manual_convolve(original_pixels, gaussian_kernel, padding_mode='reflect')

    ##print("Applying Replicate Padding (symmetric)... (This will be slow)")
    replicate_padded_result = manual_convolve(original_pixels, gaussian_kernel, padding_mode='symmetric')


    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(f'Impact of Padding Methods with a Large Gaussian Filter (σ={sigma_val})', fontsize=16)

    axes[0].imshow(zero_padded_result, cmap='gray')
    axes[0].set_title('Zero Padding')

    axes[1].imshow(mirror_padded_result, cmap='gray')
    axes[1].set_title('Mirror Padding (reflect)')

    axes[2].imshow(replicate_padded_result, cmap='gray')
    axes[2].set_title('Replicate Padding (symmetric)')

    for ax in axes.flat:
        ax.axis('off')

    plt.show()

except FileNotFoundError:
    print(f"خطا: فایل '{image_file_path}' پیدا نشد.")
except Exception as e:
    print(f"یک خطای غیرمنتظره رخ داد: {e}")