import numpy as np
import matplotlib.pyplot as plt

def manual_convolve(image, kernel):

    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
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

# --- 1. ساخت تصویر مصنوعی ---

image_size = 256
original_image = np.zeros((image_size, image_size), dtype=np.uint8)

# ایجاد یک مستطیل سفید در مرکز تصویر
rect_h, rect_w = 150, 40
start_h = (image_size - rect_h) // 2
start_w = (image_size - rect_w) // 2
original_image[start_h : start_h + rect_h, start_w : start_w + rect_w] = 255

kernel_size = 25

# اعمال فیلتر باکس
print("Applying Box Filter...")
box_kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
box_smoothed = manual_convolve(original_image, box_kernel)
print("Box filter applied.")

# اعمال فیلتر گوسی
print("Applying Gaussian Filter...")
# سیگمای بزرگتر باعث محوشدگی نرم‌تر و بیشتری می‌شود
gaussian_kernel = create_gaussian_kernel(kernel_size, sigma=8.0)
gaussian_smoothed = manual_convolve(original_image, gaussian_kernel)
print("Gaussian filter applied.")

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Comparison of Box and Gaussian Smoothing on Synthetic Image', fontsize=16)

images = [original_image, box_smoothed, gaussian_smoothed]
titles = ['Original Image', 'Box Filter Result', 'Gaussian Filter Result']

for i, ax in enumerate(axes[0]):
    ax.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    ax.set_title(titles[i])
    ax.axhline(y=image_size // 2, color='white', linestyle='--', linewidth=1, alpha=0.7)
    ax.axis('off')

middle_row = image_size // 2
for i, ax in enumerate(axes[1]):
    intensity_profile = images[i][middle_row, :]
    ax.plot(intensity_profile, color='black')
    ax.set_title(f'Intensity Profile of {titles[i]}')
    ax.set_xlim([0, image_size])
    ax.set_ylim([-5, 260])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()