# Digital Image Processing (DIP) Algorithms & Frequency Filters Toolkit

A production-grade, mathematically rigorous **Digital Image Processing (DIP) Toolkit** developed from scratch in Python. This repository implements fundamental point operations, spatial domain neighborhood filtering, noise attenuation mechanics, and advanced frequency domain filtering via the 2D Discrete Fourier Transform (DFT).

Rather than relying on high-level computer vision wrappers, the core operations (including 2D convolution and matrix padding) are engineered algorithmically to analyze pixel-level transformations and frequency attenuation behaviors.

---

## 🚀 Toolkit Architecture & Algorithmic Modules

### 1. Spatial Domain Point Transformations
Pixel-by-pixel mapping functions engineered to manipulate image dynamic range, contrast, and grayscale distributions:
- **Image Negative:** Implements the $s = (L - 1) - r$ transformation to invert grayscale intensity levels.
- **Logarithmic Transformations:** Applied via $s = c \cdot \log(1 + r)$ to expand compressed dark pixel regions within high-dynamic-range data, such as shifting raw Fourier spectrum ranges into human-viewable profiles.
- **Gamma Correction (Power-Law):** Implements $s = c \cdot r^\gamma$ to dynamically balance contrast and display luminance metrics across varying threshold limits.
- **Bit-Plane Slicing:** Deploys bitwise right-shift and logical AND operations to decompose an 8-bit image into 8 distinct binary planes, evaluating the structural weight contribution of each individual bit.

### 2. Spatial Filtering & Neighborhood Smoothing
Neighborhood operations driven by manual 2D convolution spatial engines:
- **Convolution Padding Impact:** Analyzes boundary artifacts and interpolation errors by comparing **Zero Padding**, **Mirror Padding (Reflect)**, and **Replicate Padding (Symmetric)** techniques.
- **Smoothing Characteristics (Box vs. Gaussian):** Contrasts the sharp, blocky attenuation of pixel-averaging Box filters against the isotropic, mathematically smooth blur of Gaussian kernels.
- **Median Filter Noise Reduction:** A non-linear spatial filter engineered to eliminate high-density **Salt-and-Pepper noise** without blurring crucial structural edges, outperforming linear Gaussian smoothing.

### 3. Frequency Domain Filtering (Fourier Analysis)
Transforms spatial pixels into frequency spectrum vectors via the 2D Fast Fourier Transform (FFT) to perform explicit frequency shaping:
- **Ideal Lowpass Filter (ILPF):** Cuts off all frequencies outside a strict spatial radius ($D_0$), demonstrating the physical constraints of the **Ringing Phenonmenon** caused by the sinc-function Fourier pair.
- **Butterworth Lowpass Filter (BLPF):** Implements a smooth, higher-order alternative that allows continuous attenuation control, suppressing ringing artifacts by tuning the filter order ($n$).
- **Gaussian Lowpass Filter (GLPF):** Filters high-frequency noise with a mathematically smooth Gaussian mask, ensuring a clean spatial blur completely free of ringing artifacts.

---

## 📊 Algorithmic Structure & Package Entrypoints

The repository is structured as a modular library, allowing execution of distinct transformation concepts or running the unified orchestrator tool:

- **`dip_comprehensive_toolkit.py`:** The main unified entry point providing an object-oriented toolkit to execute and plot all spatial domain, bit-plane, and filtering operations concurrently.
- **`gaussian_kernel_size_comparison.py`:** Benchmarks processing footprints and spatial blur behaviors across dynamic Gaussian scale spaces.
- **`spatial_gamma_correction.py` / `spatial_log_transformation.py` / `spatial_image_negative.py`:** Individual dedicated mathematical operations targeting point-wise grayscale modification.
- **`frequency_ideal_lowpass.py` / `frequency_butterworth_lowpass.py` / `frequency_gaussian_lowpass.py`:** Signal processing modules handling 2D Fast Fourier Transform shifting, filtering, and reconstruction matrices.

---

## 💻 Installation & Local Execution

### 1. Clone the Workspace
```bash
git clone [https://github.com/mrhashx/digital-image-processing-toolkit.git](https://github.com/mrhashx/digital-image-processing-toolkit.git)
cd digital-image-processing-toolkit
```

### 2. Install Infrastructure Prerequisites
```Bash
pip install numpy matplotlib pillow opencv-python
```
### 3. Run the Comprehensive Toolkit
```Bash
python dip_comprehensive_toolkit.py
```
## 🛠️ Core Technology Stack
Language Framework: Python 3.x

Matrix Architecture: NumPy (2D array manipulation, Fourier transformation engines via np.fft)

Imaging Core: Pillow (PIL), OpenCV-Python (cv2 used primarily for frequency operations)

Data Plotting Engine: Matplotlib

Note: This digital signal and image processing ecosystem was successfully designed, verified, and archived in December 2025.
