from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def process_bit_planes(image_path):
 
    img = Image.open(image_path).convert('L')
    original_pixels = np.array(img)
    

    bit_planes = []
    for i in range(8):
        plane = np.bitwise_and(np.right_shift(original_pixels, i), 1)
        bit_planes.append(plane * 255)
    

    reconstructed_images = {}
    

    recon_4_planes = np.zeros(original_pixels.shape, dtype=np.uint8)
    for i in range(4, 8):
        recon_4_planes += (bit_planes[i] // 255) * (2**i)
    reconstructed_images['Top 4 Planes (8,7,6,5)'] = recon_4_planes
    
  
    recon_2_planes = np.zeros(original_pixels.shape, dtype=np.uint8)
    for i in range(6, 8):
        recon_2_planes += (bit_planes[i] // 255) * (2**i)
    reconstructed_images['Top 2 Planes (8,7)'] = recon_2_planes
    
    return original_pixels, bit_planes, reconstructed_images


image_file_path = 'ddd.png' 

try:
    original_img, planes, reconstructions = process_bit_planes(image_file_path)

  
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('Extracted Bit Planes (1 to 8)', fontsize=16)
    for i, ax in enumerate(axes.flat):
        ax.imshow(planes[i], cmap='gray')
        ax.set_title(f'Bit Plane {i+1}')
        ax.axis('off')
    plt.show()


    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Image Reconstruction from Bit Planes', fontsize=16)

    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original Image (8 planes)')

    axes[1].imshow(reconstructions['Top 4 Planes (8,7,6,5)'], cmap='gray')
    axes[1].set_title('Reconstructed (Top 4 planes)')

    axes[2].imshow(reconstructions['Top 2 Planes (8,7)'], cmap='gray')
    axes[2].set_title('Reconstructed (Top 2 planes)')

    for ax in axes.flat:
        ax.axis('off')
    plt.show()

except FileNotFoundError:
    print(f"خطا: فایل در مسیر '{image_file_path}' پیدا نشد. لطفاً آدرس را بررسی کنید.")
except Exception as e:
    print(f"یک خطای غیرمنتظره رخ داد: {e}")