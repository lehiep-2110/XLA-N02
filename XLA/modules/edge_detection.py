import numpy as np
from scipy import signal

def to_gray(img: np.ndarray) -> np.ndarray:
    """Chuyển ảnh BGR sang grayscale"""
    if img is None:
        raise ValueError("Input image is None")
    if len(img.shape) == 2:
        return img
    # Công thức: Gray = 0.299*R + 0.587*G + 0.114*B
    gray = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
    return gray.astype(np.uint8)

def convolve2d_fast(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolution 2D"""
    return signal.correlate2d(img, kernel, mode='same', boundary='fill', fillvalue=0)

def gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Làm mờ Gaussian - separable convolution"""
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    
    # 2. Tạo kernel 1D
    ax = np.linspace(-(k // 2), k // 2, k)  
    kernel_1d = np.exp(-(ax**2) / (2. * sigma**2))
    kernel_1d = kernel_1d / np.sum(kernel_1d)
    
    # 3. Separable convolution
    blurred = signal.convolve(img, kernel_1d[np.newaxis, :], mode='same')
    blurred = signal.convolve(blurred, kernel_1d[:, np.newaxis], mode='same')
    
    # 4. Clip và convert
    return np.clip(blurred, 0, 255).astype(np.uint8)

def sobel_edges(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Phát hiện biên Sobel"""
    gray = to_gray(img)
    gray = gray.astype(np.float32)
    
    # Sobel kernels
    if ksize == 3:
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=np.float32)
    else:  # ksize == 5
        sobel_x = np.array([[-1, -2, 0, 2, 1],
                           [-4, -8, 0, 8, 4],
                           [-6, -12, 0, 12, 6],
                           [-4, -8, 0, 8, 4],
                           [-1, -2, 0, 2, 1]], dtype=np.float32)
        sobel_y = sobel_x.T
    
    # Tính gradient
    gx = convolve2d_fast(gray, sobel_x)
    gy = convolve2d_fast(gray, sobel_y)
    
    # Tính magnitude
    mag = np.sqrt(gx**2 + gy**2)
    
    # Normalize về 0-255
    mag = np.clip(mag * 255.0 / (mag.max() + 1e-8), 0, 255)
    
    return mag.astype(np.uint8)

def laplacian_edges(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Phát hiện biên Laplacian"""
    gray = to_gray(img)
    gray = gray.astype(np.float32)
    
    # Laplacian kernel - TỰ ĐỊNH NGHĨA
    if ksize == 3:
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=np.float32)
    else:  # ksize == 5
        laplacian_kernel = np.array([[0, 0, 1, 0, 0],
                                     [0, 1, 2, 1, 0],
                                     [1, 2, -16, 2, 1],
                                     [0, 1, 2, 1, 0],
                                     [0, 0, 1, 0, 0]], dtype=np.float32)
    
    lap = convolve2d_fast(gray, laplacian_kernel)
    lap = np.abs(lap)
    
    # Normalize
    lap = np.clip(lap * 255.0 / (lap.max() + 1e-8), 0, 255)
    
    return lap.astype(np.uint8)

def non_max_suppression(mag: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Non-maximum suppression"""
    h, w = mag.shape
    suppressed = np.zeros((h, w), dtype=np.float32)
    
    # Chuyển angle về degree
    angle = np.rad2deg(angle) % 180
    
    # Quantize angles to 4 directions
    angle_quant = np.zeros_like(angle, dtype=np.uint8)
    angle_quant[((angle >= 0) & (angle < 22.5)) | ((angle >= 157.5) & (angle <= 180))] = 0  # 0 độ
    angle_quant[(angle >= 22.5) & (angle < 67.5)] = 45  # 45 độ
    angle_quant[(angle >= 67.5) & (angle < 112.5)] = 90  # 90 độ
    angle_quant[(angle >= 112.5) & (angle < 157.5)] = 135  # 135 độ
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q, r = 255, 255
            
            # Get neighbors based on angle
            if angle_quant[i, j] == 0:
                q = mag[i, j + 1]
                r = mag[i, j - 1]
            elif angle_quant[i, j] == 45:
                q = mag[i + 1, j - 1]
                r = mag[i - 1, j + 1]
            elif angle_quant[i, j] == 90:
                q = mag[i + 1, j]
                r = mag[i - 1, j]
            elif angle_quant[i, j] == 135:
                q = mag[i - 1, j - 1]
                r = mag[i + 1, j + 1]
            
            # Suppress if not local maximum
            if mag[i, j] >= q and mag[i, j] >= r:
                suppressed[i, j] = mag[i, j]
    
    return suppressed

def hysteresis_threshold(img: np.ndarray, low: float, high: float) -> np.ndarray:
    """Double threshold và edge tracking - TỰ CODE"""
    strong = 255
    weak = 75
    
    result = np.zeros_like(img, dtype=np.uint8)
    
    # Classify pixels
    strong_mask = img >= high
    weak_mask = (img >= low) & (img < high)
    
    result[strong_mask] = strong
    result[weak_mask] = weak
    
    # Edge tracking - keep weak edges connected to strong edges
    h, w = img.shape
    changed = True
    
    while changed:
        changed = False
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if result[i, j] == weak:
                    # Check if connected to strong edge
                    if np.any(result[i-1:i+2, j-1:j+2] == strong):
                        result[i, j] = strong
                        changed = True
    
    # Remove remaining weak edges
    result[result == weak] = 0
    
    return result

def canny_edges(img: np.ndarray, low_thresh: int = 100, high_thresh: int = 200, 
                blur_ksize: int = 5, blur_sigma: float = 1.0) -> np.ndarray:
    """Thuật toán Canny - TỰ CODE TẤT CẢ CÁC BƯỚC"""
    gray = to_gray(img)
    
    # 1. Gaussian blur
    if blur_ksize and blur_ksize > 0:
        gray = gaussian_blur(gray, blur_ksize, blur_sigma)
    
    gray = gray.astype(np.float32)
    
    # 2. Sobel gradients
    sobel_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=np.float32)
    
    gx = convolve2d_fast(gray, sobel_x)
    gy = convolve2d_fast(gray, sobel_y)
    
    # 3. Gradient magnitude và direction
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    
    # 4. Non-maximum suppression
    suppressed = non_max_suppression(mag, angle)
    
    # 5. Double threshold và hysteresis
    edges = hysteresis_threshold(suppressed, low_thresh, high_thresh)
    
    return edges
