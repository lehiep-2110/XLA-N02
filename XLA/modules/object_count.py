import cv2
import numpy as np
from scipy import ndimage

def _binary_from_edges(edges: np.ndarray) -> np.ndarray:
    """Chuyển thành ảnh binary"""
    bin_img = (edges > 0).astype(np.uint8) * 255
    return bin_img

def morphology_close(img: np.ndarray, kernel_size: int = 5, iterations: int = 2) -> np.ndarray:
    """Morphological closing - sử dụng scipy để tăng tốc"""
    # Tạo circular kernel:
    #kernel =[[0, 0, 1, 0, 0],
             #[0, 1, 1, 1, 0],
             #[1, 1, 1, 1, 1],
             #[0, 1, 1, 1, 0],
             #[0, 0, 1, 0, 0]]

    y, x = np.ogrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]
    kernel = x**2 + y**2 <= (kernel_size//2)**2
    
    # CHUYỂN SANG BINARY TRƯỚC KHI XỬ LÝ
    binary = (img > 0).astype(bool)
    
    # Dilation rồi erosion - DÙNG BINARY OPERATIONS
    result = binary
    for _ in range(iterations):
        result = ndimage.binary_dilation(result, structure=kernel)  
    for _ in range(iterations):
        result = ndimage.binary_erosion(result, structure=kernel) 
    
    # Chuyển về uint8 * 255
    return result.astype(np.uint8) * 255

def find_contours(binary_img: np.ndarray):
    # Label connected components
    labeled, num_features = ndimage.label(binary_img > 0)
    
    contours = []
    for label_id in range(1, num_features + 1):
        # Tìm tất cả pixels thuộc component này
        points = np.argwhere(labeled == label_id)
        if len(points) > 0:
            contours.append(points[:, [1, 0]])
    
    return contours

def contour_area(contour: np.ndarray) -> float:
    # Với connected components, diện tích ~ số pixels
    return len(contour)

def bounding_rect(contour: np.ndarray) -> tuple:
    """Tìm bounding rectangle"""
    x_coords = contour[:, 0]
    y_coords = contour[:, 1]
    
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)
    
    return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))

def draw_rectangle(img: np.ndarray, x: int, y: int, w: int, h: int, 
                   color: tuple = (0, 255, 0), thickness: int = 2):
    """Vẽ rectangle - tối ưu"""
    result = img.copy()
    h_img, w_img = img.shape[:2]
    
    # Clamp coordinates
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    
    # Draw horizontal lines (top and bottom)
    for t in range(min(thickness, y2 - y1)):
        if y1 + t < h_img:
            result[y1 + t, x1:x2] = color
        if y2 - t - 1 >= 0 and y2 - t - 1 < h_img:
            result[y2 - t - 1, x1:x2] = color
    
    # Draw vertical lines (left and right)
    for t in range(min(thickness, x2 - x1)):
        if x1 + t < w_img:
            result[y1:y2, x1 + t] = color
        if x2 - t - 1 >= 0 and x2 - t - 1 < w_img:
            result[y1:y2, x2 - t - 1] = color
    
    return result

def count_objects_from_edges(orig_bgr: np.ndarray, edges: np.ndarray, 
                             min_area: int = 50, morph_iter: int = 4):
    
    # Convert edges to binary
    bin_img = _binary_from_edges(edges)
    
    # Morphological closing - DÙNG THAM SỐ TỪ GUI
    closed = morphology_close(bin_img, kernel_size=7, iterations=morph_iter)
    
    # Tìm contours - NHANH
    contours = find_contours(closed)

    
    count = 0
    annotated = orig_bgr.copy()
    contours_data = []
    
    for idx, cnt in enumerate(contours):
        area = contour_area(cnt)
        
        #Kiểm tra diện tích
        if area < min_area:
            continue
        
        x, y, w, h = bounding_rect(cnt)
        
        annotated = draw_rectangle(annotated, x, y, w, h, (0, 255, 0), 2)
        contours_data.append((x, y, w, h))
        count += 1
    
    return count, annotated, contours_data
