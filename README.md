# Edge Detection & Object Counting (Tkinter)

This project implements **Sobel, Laplacian, and Canny** edge detection and an application to **count objects** using contours.
It includes a Tkinter GUI that allows users to load images (JPG/PNG/BMP) **or CSV matrices**, adjust parameters, visualize results, and save outputs.

## Features
- Load image or CSV (grayscale matrix). CSV may be 0–255 or 0–1.
- Preprocessing (grayscale, Gaussian blur).
- Edge detection: Sobel, Laplacian, Canny.
- Object counting via contour detection with basic morphology.
- Side-by-side display of original, edges, and annotated result.
- Save the processed images.

## Requirements
- Python 3.9+ recommended
- Packages: `opencv-python`, `numpy`, `Pillow`

Install:
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

## File Structure
```
edge_detection_tkinter_project/
├─ main.py
├─ requirements.txt
├─ README.md
└─ modules/
   ├─ edge_detection.py
   ├─ object_count.py
   └─ io_utils.py
```

## Notes
- CSV should be a numeric matrix; headers are not required. Non-numeric cells are ignored (treated as 0).
- For large images, resizing is applied for GUI display only; processing uses the full-resolution image internally.
