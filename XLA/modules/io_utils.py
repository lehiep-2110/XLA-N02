import numpy as np
import cv2
import csv

def read_image_any(path: str):
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return read_csv_as_image(path)
    # Read with cv2 (BGR)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read file: {path}")
    # If grayscale, expand to 3 channels for consistent handling
    if len(img.shape) == 2:
        img_gray = img
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        return img_bgr
    elif img.shape[2] == 4:
        # Drop alpha for processing
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return bgr
    return img

def read_csv_as_image(path: str):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            row_vals = []
            for v in r:
                try:
                    row_vals.append(float(v))
                except Exception:
                    row_vals.append(0.0)
            if row_vals:
                rows.append(row_vals)
    if not rows:
        raise ValueError("CSV appears empty or non-numeric.")
    mat = np.array(rows, dtype=np.float32)
    # Normalize to 0-255 if needed
    if mat.max() <= 1.0:
        mat = mat * 255.0
    mat = np.clip(mat, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(mat, cv2.COLOR_GRAY2BGR)
    return bgr

def save_image(path: str, img) -> None:
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png" or ext == ".bmp" or ext == ".jpg" or ext == ".jpeg":
        cv2.imwrite(path, img)
    else:
        # default to png if unknown
        cv2.imwrite(path + ".png", img)
