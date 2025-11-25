# =========================
# [LOGIC] Imports & Constants
# =========================
import os
import tkinter as tk
import threading
import cv2
import numpy as np
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont

from modules.io_utils import read_image_any, save_image
from modules.edge_detection import sobel_edges, laplacian_edges, canny_edges
from modules.object_count import count_objects_from_edges

APP_TITLE = "Phát Hiện Biên & Đếm Đối Tượng (Nâng Cao)"
MAX_PROCESS_SIZE = 800
LIVE_DEBOUNCE_MS = 300

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        # Tăng kích thước cửa sổ và maximize
        self.geometry("1920x1080")
        self.state('zoomed')  # Maximize on Windows
        
        # Biến trạng thái
        self.img_bgr = None
        self.img_bgr_original = None
        self.edges = None
        self.annotated = None
        self.contours_data = None
        self.live_timer = None
        self.processing = False
        self.counting = False
        
        self._build_ui()
        
    def _build_ui(self):
        """Xây dựng giao diện"""
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top: Controls - compact hơn
        self._build_controls(main_container)
        
        # Bottom: Display - chiếm phần lớn
        self._build_display(main_container)
        
        # Status bar
        self.status_label = ttk.Label(self, text="Sẵn sàng", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
    def _build_controls(self, parent):
        """Bảng điều khiển - compact"""
        ctrl_frame = ttk.LabelFrame(parent, text="⚙️ Điều Khiển", padding=5)
        ctrl_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Row 1: File operations + Live update
        row1 = ttk.Frame(ctrl_frame)
        row1.pack(fill=tk.X, pady=3)
        
        ttk.Button(row1, text="📂 Mở Ảnh", command=self.load_image, width=12).pack(side=tk.LEFT, padx=3)
        ttk.Button(row1, text="💾 Lưu Biên", command=self.save_edges, width=12).pack(side=tk.LEFT, padx=3)
        ttk.Button(row1, text="💾 Lưu Kết Quả", command=self.save_annotated, width=12).pack(side=tk.LEFT, padx=3)
        
        self.live_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="🔄 Cập Nhật Trực Tiếp", variable=self.live_var).pack(side=tk.LEFT, padx=15)
        
        # Row 2: Method + Process button
        row2 = ttk.Frame(ctrl_frame)
        row2.pack(fill=tk.X, pady=3)
        
        ttk.Label(row2, text="Phương Pháp:", width=12).pack(side=tk.LEFT, padx=3)
        self.method_var = tk.StringVar(value="canny")
        
        ttk.Radiobutton(row2, text="Canny", variable=self.method_var, value="canny",
                       command=self.on_method_change).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(row2, text="Sobel", variable=self.method_var, value="sobel",
                       command=self.on_method_change).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(row2, text="Laplacian", variable=self.method_var, value="laplacian",
                       command=self.on_method_change).pack(side=tk.LEFT, padx=3)
        
        ttk.Button(row2, text="▶️ Xử Lý", command=self.process, width=10).pack(side=tk.RIGHT, padx=3)
        
        # Row 3: Parameters - compact trong 1 hàng
        row3 = ttk.LabelFrame(ctrl_frame, text="Tham Số", padding=5)
        row3.pack(fill=tk.X, pady=3)
        
        # Canny parameters - horizontal layout
        self.canny_frame = ttk.Frame(row3)
        self.low_var = tk.IntVar(value=50)
        self.high_var = tk.IntVar(value=150)
        self.blur_ksize_var = tk.IntVar(value=5)
        self.blur_sigma_var = tk.DoubleVar(value=1.4)
        
        self._create_compact_param(self.canny_frame, "Ngưỡng Thấp:", self.low_var, 0, 255, width=150)
        self._create_compact_param(self.canny_frame, "Ngưỡng Cao:", self.high_var, 0, 255, width=150)
        self._create_compact_param(self.canny_frame, "Blur Size:", self.blur_ksize_var, 1, 15, step=2, width=120)
        self._create_compact_param(self.canny_frame, "Sigma:", self.blur_sigma_var, 0.1, 5.0, resolution=0.1, width=120)
        
        # Sobel parameters
        self.sobel_frame = ttk.Frame(row3)
        self.sobel_ksize_var = tk.IntVar(value=3)
        
        sobel_f = ttk.Frame(self.sobel_frame)
        sobel_f.pack(side=tk.LEFT, padx=5)
        ttk.Label(sobel_f, text="Kích Thước Kernel:").pack(side=tk.LEFT, padx=3)
        sobel_combo = ttk.Combobox(sobel_f, textvariable=self.sobel_ksize_var,
                                   values=[3, 5], state="readonly", width=8)
        sobel_combo.pack(side=tk.LEFT, padx=3)
        sobel_combo.bind("<<ComboboxSelected>>", lambda e: self.on_param_change())
        
        # Laplacian parameters
        self.laplacian_frame = ttk.Frame(row3)
        self.laplacian_ksize_var = tk.IntVar(value=3)
        
        lap_f = ttk.Frame(self.laplacian_frame)
        lap_f.pack(side=tk.LEFT, padx=5)
        ttk.Label(lap_f, text="Kích Thước Kernel:").pack(side=tk.LEFT, padx=3)
        lap_combo = ttk.Combobox(lap_f, textvariable=self.laplacian_ksize_var,
                                values=[3, 5], state="readonly", width=8)
        lap_combo.pack(side=tk.LEFT, padx=3)
        lap_combo.bind("<<ComboboxSelected>>", lambda e: self.on_param_change())
        
        # Show Canny by default
        self.canny_frame.pack(fill=tk.X)
        
        # Row 4: Object counting - compact
        row4 = ttk.Frame(ctrl_frame)
        row4.pack(fill=tk.X, pady=3)
        
        ttk.Label(row4, text="🎯 Đếm Đối Tượng:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Min area
        self.min_area_var = tk.IntVar(value=50)
        ttk.Label(row4, text="Diện tích min:").pack(side=tk.LEFT, padx=3)
        ttk.Scale(row4, from_=10, to=1000, variable=self.min_area_var, 
                 orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=3)
        self.min_area_label = ttk.Label(row4, text="50", width=4)
        self.min_area_label.pack(side=tk.LEFT, padx=3)
        
        def update_min_area(*args):
            self.min_area_label.config(text=str(self.min_area_var.get()))
        self.min_area_var.trace_add("write", update_min_area)
        
        # THÊM MORPHOLOGY ITERATIONS
        self.morph_iter_var = tk.IntVar(value=4)
        ttk.Label(row4, text="Morphology:").pack(side=tk.LEFT, padx=(10, 3))
        ttk.Scale(row4, from_=1, to=10, variable=self.morph_iter_var, 
                 orient=tk.HORIZONTAL, length=100).pack(side=tk.LEFT, padx=3)
        self.morph_iter_label = ttk.Label(row4, text="4", width=3)
        self.morph_iter_label.pack(side=tk.LEFT, padx=3)
        
        def update_morph_iter(*args):
            self.morph_iter_label.config(text=str(self.morph_iter_var.get()))
        self.morph_iter_var.trace_add("write", update_morph_iter)
        
        self.show_numbers_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row4, text="Hiển thị số", variable=self.show_numbers_var,
                       command=self.on_show_numbers_change).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(row4, text="🔍 Đếm", command=self.count_objects, width=8).pack(side=tk.LEFT, padx=5)
        
        self.count_label = ttk.Label(row4, text="Số Đối Tượng: 0", 
                                     font=("Arial", 10, "bold"), foreground="blue")
        self.count_label.pack(side=tk.LEFT, padx=10)
        
    def _create_compact_param(self, parent, label, variable, from_, to, step=1, resolution=1, width=150):
        """Tạo control compact cho parameter"""
        frame = ttk.Frame(parent)
        frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(frame, text=label, width=12).pack(side=tk.LEFT, padx=2)
        
        scale = ttk.Scale(frame, from_=from_, to=to, variable=variable, 
                         orient=tk.HORIZONTAL, length=width)
        scale.pack(side=tk.LEFT, padx=2)
        
        value_label = ttk.Label(frame, text=str(variable.get()), width=4)
        value_label.pack(side=tk.LEFT, padx=2)
        
        def update(*args):
            val = variable.get()
            if isinstance(variable, tk.IntVar):
                if step > 1:
                    val = round(val / step) * step
                    if val % 2 == 0 and step == 2:
                        val += 1
                    variable.set(val)
                value_label.config(text=str(val))
            else:
                value_label.config(text=f"{val:.1f}")
            self.on_param_change()
        
        variable.trace_add("write", update)
        
    def on_method_change(self):
        """Callback khi thay đổi phương pháp"""
        self.canny_frame.pack_forget()
        self.sobel_frame.pack_forget()
        self.laplacian_frame.pack_forget()
        
        method = self.method_var.get()
        if method == "canny":
            self.canny_frame.pack(fill=tk.X)
        elif method == "sobel":
            self.sobel_frame.pack(fill=tk.X)
        else:
            self.laplacian_frame.pack(fill=tk.X)
        
        self.on_param_change()
    
    def on_show_numbers_change(self):
        """Callback khi thay đổi hiển thị số"""
        if self.annotated is not None and self.contours_data is not None:
            self._redraw_annotated()
    
    def _build_display(self, parent):
        """Khu vực hiển thị ảnh - 3 canvas ngang, chiếm phần lớn"""
        display_frame = ttk.Frame(parent)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress = ttk.Progressbar(display_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(0, 5))
        
        # 3 canvas ngang
        canvas_container = ttk.Frame(display_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvases = {}
        titles = ["📷 Ảnh Gốc", "🔍 Biên Phát Hiện", "🎯 Đối Tượng Đếm"]
        keys = ["original", "edges", "objects"]
        
        for i, (title, key) in enumerate(zip(titles, keys)):
            frame = ttk.LabelFrame(canvas_container, text=title, padding=5)
            frame.grid(row=0, column=i, sticky="nsew", padx=5)
            
            canvas = tk.Canvas(frame, bg="#2b2b2b", highlightthickness=0)
            canvas.pack(fill=tk.BOTH, expand=True)
            self.canvases[key] = canvas
            
            canvas.bind("<Configure>", lambda e, k=key: self._on_canvas_resize(k))
        
        # Configure grid weights - 3 cột ngang đều nhau
        for i in range(3):
            canvas_container.columnconfigure(i, weight=1)
        canvas_container.rowconfigure(0, weight=1)
        
    def _on_canvas_resize(self, key):
        """Callback khi canvas resize"""
        if key == "original" and self.img_bgr is not None:
            self._refresh_original()
        elif key == "edges" and self.edges is not None:
            self._refresh_edges()
        elif key == "objects" and self.annotated is not None:
            self._refresh_annotated()
    
    def on_param_change(self):
        """Callback khi tham số thay đổi"""
        if self.live_var.get() and self.img_bgr is not None and not self.processing:
            self.schedule_live_update()
    
    def schedule_live_update(self):
        """Debounce cho cập nhật trực tiếp"""
        if self.live_timer:
            self.after_cancel(self.live_timer)
        self.live_timer = self.after(LIVE_DEBOUNCE_MS, self.process)
    
    def _resize_for_processing(self, img):
        """Resize ảnh nếu quá lớn"""
        h, w = img.shape[:2]
        if max(h, w) > MAX_PROCESS_SIZE:
            scale = MAX_PROCESS_SIZE / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            from PIL import Image as PILImage
            if len(img.shape) == 3:
                pil_img = PILImage.fromarray(img[:, :, ::-1])
            else:
                pil_img = PILImage.fromarray(img)
            
            pil_img = pil_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
            
            if len(img.shape) == 3:
                return np.array(pil_img)[:, :, ::-1]
            return np.array(pil_img)
        return img
    
    def load_image(self):
        """Tải ảnh"""
        path = filedialog.askopenfilename(
            title="Chọn Ảnh",
            filetypes=[
                ("Tất cả ảnh", "*.png *.jpg *.jpeg *.bmp *.csv"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("BMP", "*.bmp"),
                ("CSV", "*.csv")
            ]
        )
        if not path:
            return
        
        try:
            self.img_bgr_original = read_image_any(path)
            self.img_bgr = self._resize_for_processing(self.img_bgr_original)
            
            self.status(f"Đã tải: {path}")
            self._refresh_original()
            
            if self.live_var.get():
                self.process()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải ảnh:\n{e}")
    
    def process(self):
        """Xử lý ảnh"""
        if self.img_bgr is None:
            messagebox.showwarning("Chưa Có Ảnh", "Vui lòng tải ảnh trước.")
            return
        
        if self.processing:
            return
        
        thread = threading.Thread(target=self._do_process_thread, daemon=True)
        thread.start()
    
    def _do_process_thread(self):
        """Xử lý trong thread"""
        self.processing = True
        self.after(0, self.progress.start)
        self.after(0, lambda: self.status("Đang xử lý..."))
        
        try:
            method = self.method_var.get()
            
            if method == "canny":
                self.edges = canny_edges(
                    self.img_bgr,
                    low_thresh=self.low_var.get(),
                    high_thresh=self.high_var.get(),
                    blur_ksize=self.blur_ksize_var.get(),
                    blur_sigma=self.blur_sigma_var.get()
                )
            elif method == "sobel":
                self.edges = sobel_edges(self.img_bgr, ksize=self.sobel_ksize_var.get())
            else:
                self.edges = laplacian_edges(self.img_bgr, ksize=self.laplacian_ksize_var.get())
            
            self.after(0, self._refresh_edges)
            self.after(0, lambda: self.status(f"Đã xử lý bằng {method.upper()}"))
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Lỗi", f"Xử lý thất bại:\n{e}"))
            self.after(0, lambda: self.status("Lỗi"))
        finally:
            self.after(0, self.progress.stop)
            self.processing = False
    
    def count_objects(self):
        """Đếm đối tượng - chạy trong thread"""
        if self.edges is None:
            messagebox.showwarning("Chưa Có Biên", "Vui lòng xử lý ảnh trước.")
            return
        
        if self.counting:
            return
        
        thread = threading.Thread(target=self._do_count_thread, daemon=True)
        thread.start()
    
    def _do_count_thread(self):
        """Đếm trong thread"""
        self.counting = True
        self.after(0, self.progress.start)
        self.after(0, lambda: self.status("Đang đếm đối tượng..."))
        
        try:
            count, annotated, contours_data = count_objects_from_edges(
                self.img_bgr, self.edges, 
                min_area=self.min_area_var.get(),
                morph_iter=self.morph_iter_var.get()  # ← THÊM THAM SỐ
            )
            
            self.annotated = annotated
            self.contours_data = contours_data
            
            self.after(0, lambda: self.count_label.config(text=f"Số Đối Tượng: {count}"))
            self.after(0, self._redraw_annotated)
            self.after(0, lambda: self.status(f"Tìm thấy {count} đối tượng"))
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Lỗi", f"Đếm thất bại:\n{e}"))
            self.after(0, lambda: self.status("Lỗi"))
        finally:
            self.after(0, self.progress.stop)
            self.counting = False
    
    def _redraw_annotated(self):
        """Vẽ lại annotated với/không có số"""
        if self.annotated is None or self.contours_data is None:
            return
        
        if self.show_numbers_var.get():
            result = self._draw_numbers_on_image(self.annotated.copy(), self.contours_data)
        else:
            result = self.annotated.copy()
        
        self.annotated_display = result
        self._refresh_annotated()
    
    def _draw_numbers_on_image(self, img_bgr, contours_data):
        """Vẽ số thứ tự lên ảnh"""
        img_rgb = img_bgr[:, :, ::-1]
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font_size = max(20, min(img_bgr.shape[0], img_bgr.shape[1]) // 30)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        for idx, (x, y, w, h) in enumerate(contours_data, 1):
            center_x = x + w // 2
            center_y = y + h // 2
            
            text = str(idx)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            circle_radius = max(text_w, text_h) // 2 + 10
            draw.ellipse(
                [center_x - circle_radius, center_y - circle_radius,
                 center_x + circle_radius, center_y + circle_radius],
                fill=(255, 255, 0),
                outline=(0, 0, 0),
                width=3
            )
            
            draw.text(
                (center_x - text_w // 2, center_y - text_h // 2),
                text,
                fill=(255, 0, 0),
                font=font
            )
        
        return np.array(pil_img)[:, :, ::-1]
    
    def save_edges(self):
        """Lưu ảnh biên"""
        if self.edges is None:
            messagebox.showwarning("Chưa Có Biên", "Không có gì để lưu.")
            return
        path = filedialog.asksaveasfilename(
            title="Lưu Ảnh Biên",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")]
        )
        if path:
            save_image(path, self.edges)
            self.status(f"Đã lưu: {path}")
    
    def save_annotated(self):
        """Lưu ảnh kết quả"""
        img_to_save = self.annotated_display if hasattr(self, 'annotated_display') else self.annotated
        
        if img_to_save is None:
            messagebox.showwarning("Chưa Có Kết Quả", "Không có gì để lưu.")
            return
        path = filedialog.asksaveasfilename(
            title="Lưu Kết Quả",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")]
        )
        if path:
            save_image(path, img_to_save)
            self.status(f"Đã lưu: {path}")
    
    def _fit_image_to_canvas(self, img_array, canvas):
        """Fit ảnh vào canvas"""
        if img_array is None:
            return None
        
        canvas.update_idletasks()
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        
        if canvas_w <= 1 or canvas_h <= 1:
            return None
        
        if len(img_array.shape) == 2:
            img_rgb = np.stack([img_array]*3, axis=2)
        else:
            img_rgb = img_array[:, :, ::-1]
        
        img_h, img_w = img_array.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        return ImageTk.PhotoImage(pil_img)
    
    def _refresh_original(self):
        """Cập nhật canvas ảnh gốc"""
        canvas = self.canvases["original"]
        photo = self._fit_image_to_canvas(self.img_bgr, canvas)
        if photo:
            canvas.delete("all")
            canvas.create_image(canvas.winfo_width()//2, canvas.winfo_height()//2,
                              image=photo, anchor=tk.CENTER)
            canvas.image = photo
    
    def _refresh_edges(self):
        """Cập nhật canvas ảnh biên"""
        canvas = self.canvases["edges"]
        photo = self._fit_image_to_canvas(self.edges, canvas)
        if photo:
            canvas.delete("all")
            canvas.create_image(canvas.winfo_width()//2, canvas.winfo_height()//2,
                              image=photo, anchor=tk.CENTER)
            canvas.image = photo
    
    def _refresh_annotated(self):
        """Cập nhật canvas kết quả"""
        img_to_show = self.annotated_display if hasattr(self, 'annotated_display') else self.annotated
        
        canvas = self.canvases["objects"]
        photo = self._fit_image_to_canvas(img_to_show, canvas)
        if photo:
            canvas.delete("all")
            canvas.create_image(canvas.winfo_width()//2, canvas.winfo_height()//2,
                              image=photo, anchor=tk.CENTER)
            canvas.image = photo
    
    def status(self, text):
        """Cập nhật trạng thái"""
        self.status_label.config(text=text)

if __name__ == "__main__":
    app = App()
    app.mainloop()
