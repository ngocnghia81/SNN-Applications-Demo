"""
Ứng dụng GUI cho mô hình Spiking Neural Network
Tận dụng mô hình SNN đã huấn luyện để nhận dạng chữ số viết tay
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk, ImageDraw, ImageOps
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import threading
import queue
import time
import cv2  # Thêm import cho OpenCV

# Đường dẫn đến file inference
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Import từ file inference đã có
    from spikingjelly_inference import Config, SNNMLP, SNNCNN, load_model, predict, visualize_prediction
    print("Đã import thành công từ file inference!")
except ImportError as e:
    print(f"Lỗi khi import: {e}")
    print("Đảm bảo file spikingjelly_inference.py tồn tại trong cùng thư mục với ứng dụng!")
    sys.exit(1)

class SNNApp:
    def __init__(self, root):
        # Khởi tạo cửa sổ chính
        self.root = root
        self.root.title("SNN - Nhận dạng chữ số viết tay")
        self.root.geometry("1200x800")
        self.root.configure(background="#f0f0f0")
        self.root.resizable(True, True)
        
        # Các biến của ứng dụng
        self.model = None
        self.model_type = None
        self.config = Config()
        self.canvas_width = 280  # 10x kích thước ảnh MNIST
        self.canvas_height = 280
        self.brush_size = 15
        self.drawing = False
        self.last_x = 0
        self.last_y = 0
        self.processing_queue = queue.Queue()
        self.photo_image = None  # Biến lưu ảnh gốc khi tải từ file
        
        # Biến cho camera
        self.camera_active = False
        self.camera_thread = None
        self.camera_frame = None
        self.camera_id = 0
        self.is_processing_frame = False
        
        # Biến cho bộ lọc ổn định kết quả
        self.prediction_history = []  # Lịch sử dự đoán
        self.last_prediction = None   # Dự đoán cuối cùng
        self.prediction_stability_threshold = 5  # Số frame giữ nguyên dự đoán để xác nhận
        self.min_confidence_threshold = 0.6      # Ngưỡng tin cậy tối thiểu
        self.no_digit_frames = 0       # Số frame không phát hiện chữ số liên tiếp
        self.no_digit_threshold = 10   # Ngưỡng để xóa kết quả khi không phát hiện
        
        # Tải model trong luồng riêng để không làm đứng giao diện
        self.status_var = tk.StringVar()
        self.status_var.set("Đang khởi tạo...")
        
        # Tạo các thành phần của giao diện
        self.create_widgets()
        
        # Tải model trong luồng riêng
        self.load_model_thread = threading.Thread(target=self.load_model_in_thread)
        self.load_model_thread.daemon = True
        self.load_model_thread.start()
        
        # Chạy một luồng để xử lý các dự đoán
        self.processing_thread = threading.Thread(target=self.process_predictions)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Cập nhật trạng thái
        self.root.after(100, self.update_status)
        
        # Đảm bảo giải phóng tài nguyên camera khi đóng ứng dụng
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        # Tạo frame chính
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame bên trái: Canvas để vẽ + các nút điều khiển
        left_frame = ttk.Frame(main_frame, padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas để vẽ
        canvas_frame = ttk.LabelFrame(left_frame, text="Vẽ chữ số / Camera", padding="10")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height, 
                               bg="black", highlightthickness=1, highlightbackground="gray")
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_drawing)
        
        # Các nút điều khiển
        controls_frame = ttk.Frame(left_frame, padding="5")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Xóa", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Nhận dạng", command=self.recognize).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Tải ảnh", command=self.load_image).pack(side=tk.LEFT, padx=5)
        
        # Nút camera
        self.camera_button = ttk.Button(controls_frame, text="Bật Camera", command=self.toggle_camera)
        self.camera_button.pack(side=tk.LEFT, padx=5)
        
        # Nút chụp ảnh từ camera
        self.capture_button = ttk.Button(controls_frame, text="Chụp ảnh", command=self.capture_frame, state=tk.DISABLED)
        self.capture_button.pack(side=tk.LEFT, padx=5)
        
        # Nút thiết lập camera
        self.settings_button = ttk.Button(controls_frame, text="Cài đặt", command=self.show_settings)
        self.settings_button.pack(side=tk.LEFT, padx=5)
        
        # Cài đặt độ dày bút vẽ
        brush_frame = ttk.Frame(controls_frame)
        brush_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(brush_frame, text="Kích thước bút:").pack(side=tk.LEFT)
        self.brush_scale = ttk.Scale(brush_frame, from_=1, to=40, orient="horizontal", 
                                    value=self.brush_size, length=150,
                                    command=self.update_brush_size)
        self.brush_scale.pack(side=tk.LEFT, padx=5)
        
        # Frame bên phải: Hiển thị kết quả
        right_frame = ttk.Frame(main_frame, padding="5")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Frame hiển thị kết quả nhận dạng
        result_frame = ttk.LabelFrame(right_frame, text="Kết quả nhận dạng", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Hiển thị chữ số được nhận dạng
        self.result_label = ttk.Label(result_frame, text="?", font=("Arial", 120, "bold"), 
                                     anchor="center")
        self.result_label.pack(pady=10)
        
        # Hiển thị độ tin cậy
        confidence_frame = ttk.Frame(result_frame)
        confidence_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(confidence_frame, text="Độ tin cậy:").pack(side=tk.LEFT)
        self.confidence_var = tk.StringVar(value="0%")
        ttk.Label(confidence_frame, textvariable=self.confidence_var, font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Biểu đồ thanh cho các lớp
        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Phân phối xác suất các lớp")
        self.ax.set_xlabel("Lớp")
        self.ax.set_ylabel("Xác suất")
        self.ax.set_xticks(range(10))
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, result_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Hiển thị đồ thị hoạt động neuron theo thời gian
        self.time_fig = Figure(figsize=(6, 3), dpi=100)
        self.time_ax = self.time_fig.add_subplot(111)
        self.time_ax.set_title("Hoạt động neuron theo thời gian")
        self.time_ax.set_xlabel("Bước thời gian")
        self.time_ax.set_ylabel("Hoạt động")
        
        self.time_canvas_plot = FigureCanvasTkAgg(self.time_fig, result_frame)
        self.time_canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_model_in_thread(self):
        try:
            self.status_var.set("Đang tải model...")
            self.model, self.model_type = load_model()
            self.status_var.set(f"Model đã được tải thành công! Loại: {self.model_type.upper()}")
        except Exception as e:
            self.status_var.set(f"Lỗi khi tải model: {str(e)}")
            messagebox.showerror("Lỗi", f"Không thể tải model: {str(e)}")
    
    def update_status(self):
        # Cập nhật status bar định kỳ
        self.root.after(100, self.update_status)
    
    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                  fill="white", width=self.brush_size,
                                  capstyle=tk.ROUND, smooth=True)
            self.last_x = x
            self.last_y = y
    
    def end_drawing(self, event):
        self.drawing = False
    
    def update_brush_size(self, val):
        self.brush_size = int(float(val))
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_label.config(text="?")
        self.confidence_var.set("0%")
        self.ax.clear()
        self.ax.set_title("Phân phối xác suất các lớp")
        self.ax.set_xlabel("Lớp")
        self.ax.set_ylabel("Xác suất")
        self.ax.set_xticks(range(10))
        self.canvas_plot.draw()
        
        self.time_ax.clear()
        self.time_ax.set_title("Hoạt động neuron theo thời gian")
        self.time_ax.set_xlabel("Bước thời gian")
        self.time_ax.set_ylabel("Hoạt động")
        self.time_canvas_plot.draw()
    
    def get_image_from_canvas(self):
        # Thay thế phương thức cũ bằng cách sử dụng PIL trực tiếp
        # Tạo một ảnh trống với kích thước canvas
        image = Image.new("RGB", (self.canvas_width, self.canvas_height), "black")
        draw = ImageDraw.Draw(image)
        
        # Lấy tất cả các đối tượng được vẽ trên canvas
        items = self.canvas.find_all()
        
        # Vẽ lại các đường thẳng từ canvas vào ảnh PIL
        for item in items:
            if self.canvas.type(item) == "line":
                # Lấy tọa độ của đường thẳng
                coords = self.canvas.coords(item)
                # Lấy thuộc tính của đường thẳng
                width = self.canvas.itemcget(item, "width")
                # Vẽ lại đường thẳng trên ảnh PIL
                for i in range(0, len(coords), 2):
                    if i+2 < len(coords):
                        draw.line([coords[i], coords[i+1], coords[i+2], coords[i+3]], 
                                  fill="white", width=int(float(width)))
            elif self.canvas.type(item) == "image":
                # Nếu là ảnh được tải lên, lấy ảnh đó trực tiếp
                return self.photo_image.resize((28, 28), Image.LANCZOS)
        
        # Chuyển đổi sang ảnh xám
        image = image.convert("L")
        
        # Chỉnh lại kích thước thành 28x28 (kích thước MNIST)
        image = image.resize((28, 28), Image.LANCZOS)
        
        # Đảo ngược màu (MNIST có nền đen, chữ trắng) nếu cần
        # Ở đây không cần đảo ngược vì đã vẽ chữ trắng trên nền đen
        
        return image
    
    def preprocess_image(self, image):
        # Chuyển đổi ảnh PIL thành tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        tensor = transform(image).unsqueeze(0)
        return tensor
    
    def recognize(self):
        if self.model is None:
            messagebox.showinfo("Thông báo", "Model chưa được tải xong. Vui lòng đợi...")
            return
        
        # Lấy ảnh từ canvas
        image = self.get_image_from_canvas()
        
        # Tiền xử lý ảnh
        tensor = self.preprocess_image(image)
        
        # Đưa vào hàng đợi để xử lý
        self.status_var.set("Đang nhận dạng...")
        self.processing_queue.put((tensor, image))
    
    def process_predictions(self):
        while True:
            try:
                # Lấy ảnh từ hàng đợi
                tensor, image = self.processing_queue.get(timeout=0.1)
                
                # Dự đoán
                predicted, confidence, spikes = predict(self.model, tensor, self.model_type)
                
                # Nếu đang sử dụng camera, áp dụng bộ lọc ổn định kết quả
                if image is None and self.camera_active:
                    # Kiểm tra ngưỡng tin cậy
                    if confidence < self.min_confidence_threshold:
                        # Nếu độ tin cậy thấp, bỏ qua kết quả này
                        continue
                    
                    # Thêm dự đoán mới vào lịch sử
                    self.prediction_history.append((predicted, confidence))
                    
                    # Giới hạn kích thước lịch sử
                    max_history = 7
                    if len(self.prediction_history) > max_history:
                        self.prediction_history = self.prediction_history[-max_history:]
                    
                    # Chỉ khi có đủ lịch sử mới phân tích
                    if len(self.prediction_history) >= 3:
                        # Đếm số lần xuất hiện của mỗi chữ số trong lịch sử
                        pred_counts = {}
                        for pred, conf in self.prediction_history:
                            pred_counts[pred] = pred_counts.get(pred, 0) + 1
                        
                        # Tìm dự đoán phổ biến nhất
                        most_common_pred, count = max(pred_counts.items(), key=lambda x: x[1])
                        
                        # Nếu dự đoán phổ biến nhất xuất hiện ít nhất 60% trong lịch sử
                        stability_ratio = count / len(self.prediction_history)
                        if stability_ratio >= 0.6:
                            # Tính độ tin cậy trung bình cho dự đoán phổ biến nhất
                            avg_conf = sum([conf for pred, conf in self.prediction_history if pred == most_common_pred]) / count
                            
                            # Nếu chữ số thay đổi hoặc lần đầu dự đoán
                            if self.last_prediction != most_common_pred:
                                self.last_prediction = most_common_pred
                                # Cập nhật UI với kết quả mới
                                self.root.after(0, lambda p=most_common_pred, c=avg_conf, s=spikes: self.update_ui(p, c, s, tensor))
                else:
                    # Nếu không phải camera (vẽ tay hoặc tải ảnh), hiển thị kết quả ngay
                    self.root.after(0, lambda: self.update_ui(predicted, confidence, spikes, tensor))
                
            except queue.Empty:
                # Không có gì trong hàng đợi, tiếp tục chờ
                time.sleep(0.1)
            except Exception as e:
                # Xử lý lỗi
                print(f"Lỗi khi xử lý dự đoán: {e}")
                self.root.after(0, lambda: self.status_var.set(f"Lỗi: {str(e)}"))
    
    def update_ui(self, predicted, confidence, spikes, tensor):
        # Cập nhật kết quả
        self.result_label.config(text=str(predicted))
        self.confidence_var.set(f"{confidence*100:.2f}%")
        
        # Cập nhật biểu đồ
        self.ax.clear()
        output_mean = spikes.mean(0).squeeze(0).cpu().numpy()
        bars = self.ax.bar(range(10), output_mean)
        bars[predicted].set_color('red')
        self.ax.set_title("Phân phối xác suất các lớp")
        self.ax.set_xlabel("Lớp")
        self.ax.set_ylabel("Xác suất")
        self.ax.set_xticks(range(10))
        self.canvas_plot.draw()
        
        # Cập nhật đồ thị theo thời gian
        self.time_ax.clear()
        spike_pattern = spikes.squeeze(1).cpu().numpy()
        self.time_ax.plot(spike_pattern[:, predicted], 'r-', linewidth=2, label=f'Lớp {predicted}')
        for i in range(10):
            if i != predicted:
                self.time_ax.plot(spike_pattern[:, i], 'b-', alpha=0.3, label=f'Lớp {i}' if i == 0 else None)
        self.time_ax.set_title("Hoạt động neuron theo thời gian")
        self.time_ax.set_xlabel("Bước thời gian")
        self.time_ax.set_ylabel("Hoạt động")
        self.time_ax.legend()
        self.time_canvas_plot.draw()
        
        # Cập nhật trạng thái
        self.status_var.set(f"Đã nhận dạng: {predicted} (Độ tin cậy: {confidence*100:.2f}%)")
        
        # Lưu kết quả visualization
        try:
            os.makedirs("results", exist_ok=True)
            timestamp = int(time.time())
            
            # Lưu ảnh biểu đồ
            self.fig.savefig(f"results/probs_{timestamp}.png")
            self.time_fig.savefig(f"results/activity_{timestamp}.png")
        except Exception as e:
            print(f"Lỗi khi lưu kết quả: {e}")
    
    def load_image(self):
        # Mở hộp thoại chọn file
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")],
            title="Chọn ảnh chữ số"
        )
        
        if not file_path:
            return
        
        try:
            # Tải ảnh
            image = Image.open(file_path).convert("L")
            
            # Lưu ảnh gốc để sử dụng sau này
            self.photo_image = image
            
            # Resize về kích thước canvas
            display_image = image.resize((self.canvas_width, self.canvas_height), Image.LANCZOS)
            
            # Hiển thị trên canvas
            self.clear_canvas()
            self.photo = ImageTk.PhotoImage(display_image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
            # Tiền xử lý cho model
            tensor = self.preprocess_image(image.resize((28, 28), Image.LANCZOS))
            
            # Dự đoán
            self.processing_queue.put((tensor, image))
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải ảnh: {str(e)}")
    
    def toggle_camera(self):
        """Bật/tắt camera"""
        if self.camera_active:
            # Tắt camera
            self.camera_active = False
            self.camera_button.config(text="Bật Camera")
            self.capture_button.config(state=tk.DISABLED)  # Vô hiệu hóa nút chụp ảnh
            if self.camera_thread is not None and self.camera_thread.is_alive():
                # Đợi thread kết thúc
                self.camera_thread.join(timeout=1.0)
            # Hiển thị lại canvas vẽ
            self.clear_canvas()
        else:
            # Bật camera
            self.camera_active = True
            self.camera_button.config(text="Tắt Camera")
            self.capture_button.config(state=tk.NORMAL)  # Kích hoạt nút chụp ảnh
            # Chạy camera trong thread riêng
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
    
    def camera_loop(self):
        """Vòng lặp xử lý camera"""
        try:
            # Thử tìm camera có sẵn
            available_camera = False
            for i in range(3):  # Thử với các chỉ số camera từ 0 đến 2
                self.camera_id = i
                cap = cv2.VideoCapture(self.camera_id)
                if cap.isOpened():
                    available_camera = True
                    self.status_var.set(f"Đã tìm thấy camera ở vị trí {self.camera_id}")
                    break
                cap.release()
            
            if not available_camera:
                # Nếu không tìm thấy camera nào, thông báo lỗi
                self.camera_active = False
                self.camera_button.config(text="Bật Camera")
                messagebox.showerror("Lỗi Camera", "Không thể tìm thấy webcam nào. Vui lòng kiểm tra kết nối hoặc quyền truy cập.")
                return
            
            # Thiết lập kích thước
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            except:
                # Một số camera không hỗ trợ thiết lập kích thước
                pass
            
            # Hiển thị thông tin camera
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.status_var.set(f"Camera đang hoạt động: {frame_width}x{frame_height} @ {fps:.1f}fps")
            
            frame_count_err = 0
            max_frame_errors = 5
            
            while self.camera_active:
                # Đọc frame từ camera
                ret, frame = cap.read()
                if not ret:
                    frame_count_err += 1
                    if frame_count_err > max_frame_errors:
                        # Nếu nhiều lỗi liên tiếp, dừng camera
                        self.status_var.set("Mất kết nối với camera")
                        break
                    # Ngủ một chút rồi thử lại
                    time.sleep(0.1)
                    continue
                
                # Reset lỗi nếu đọc frame thành công
                frame_count_err = 0
                
                # Xử lý frame
                self.process_camera_frame(frame)
                
                # Ngủ một chút để giảm CPU
                time.sleep(0.04)  # khoảng 25fps
                
            # Hiển thị thông báo camera đã tắt
            self.status_var.set("Camera đã tắt")
            
            # Giải phóng camera
            cap.release()
            
        except Exception as e:
            error_msg = f"Lỗi khi sử dụng camera: {str(e)}"
            print(error_msg)
            self.status_var.set(error_msg)
            messagebox.showerror("Lỗi Camera", error_msg)
            self.camera_active = False
            self.camera_button.config(text="Bật Camera")
    
    def process_camera_frame(self, frame):
        """Xử lý frame từ camera và hiển thị lên canvas"""
        if self.is_processing_frame:
            return  # Tránh xử lý quá nhiều frames
            
        self.is_processing_frame = True
        
        try:
            # Lưu frame hiện tại để nút chụp ảnh có thể sử dụng
            self.current_frame = frame.copy()
            
            # Hiển thị frame gốc
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_pil = img_pil.resize((self.canvas_width, self.canvas_height), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.root.after(0, lambda: self.update_camera_display(img_tk, img_pil))
            
            # Xử lý nhận dạng định kỳ (không phải mỗi frame)
            if not hasattr(self, 'frame_count'):
                self.frame_count = 0
            
            self.frame_count += 1
            if self.frame_count % 5 == 0 and self.model is not None:  # Xử lý mỗi 5 frames
                # Tiền xử lý ảnh từ camera
                processed_img = self.preprocess_camera_image(frame)
                
                # Nếu tensor không phải toàn 0, có nghĩa là đã phát hiện được chữ số
                if torch.sum(processed_img) > 0:
                    # Thêm vào queue để xử lý nhận dạng
                    self.processing_queue.put((processed_img, None))
            
        finally:
            self.is_processing_frame = False
    
    def update_camera_display(self, img_tk, img_pil):
        """Cập nhật hiển thị camera trên canvas"""
        self.camera_photo = img_tk  # Giữ tham chiếu
        self.photo_image = img_pil  # Lưu ảnh gốc
        
        # Xóa canvas và hiển thị ảnh mới
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.camera_photo, anchor=tk.NW)
    
    def preprocess_camera_image(self, frame):
        """Tiền xử lý ảnh từ camera để phát hiện và chuẩn bị cho mô hình SNN"""
        try:
            # Chuyển thành ảnh xám
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Lưu ảnh gốc để debug
            debug_dir = "debug_frames"
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = int(time.time())
            cv2.imwrite(f"{debug_dir}/original_{timestamp}.png", gray)
            
            # Thử phương pháp 1: Xử lý cơ bản với cân bằng histogram
            gray1 = cv2.equalizeHist(gray)
            blurred1 = cv2.GaussianBlur(gray1, (5, 5), 0)
            thresh1 = cv2.adaptiveThreshold(blurred1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # Thử phương pháp 2: Ngưỡng đơn giản với Otsu trên ảnh gốc
            _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            
            # Thử phương pháp 3: Tăng độ tương phản trước rồi mới ngưỡng
            # Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray3 = clahe.apply(gray)
            blurred3 = cv2.GaussianBlur(gray3, (5, 5), 0)
            _, thresh3 = cv2.threshold(blurred3, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            
            # Kết hợp các kết quả
            combined = cv2.bitwise_or(thresh1, cv2.bitwise_or(thresh2, thresh3))
            
            # Loại bỏ nhiễu
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Lưu ảnh đã xử lý để debug
            cv2.imwrite(f"{debug_dir}/processed_{timestamp}.png", closing)
            
            # Tìm contours
            contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Nếu không tìm thấy contour nào
            if not contours or len(contours) == 0:
                self.no_digit_frames += 1
                if self.no_digit_frames >= self.no_digit_threshold:
                    # Xóa kết quả nếu không phát hiện đủ số frame liên tiếp
                    self.root.after(0, self.clear_results)
                return torch.zeros((1, 1, 28, 28), dtype=torch.float32)
            
            # Đặt lại bộ đếm khi tìm thấy contour
            self.no_digit_frames = 0
            
            # Lấy tất cả các contour có diện tích lớn hơn ngưỡng
            min_area = 100  # Giảm diện tích tối thiểu xuống để bắt các chữ số nhỏ hơn
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            # Nếu không tìm thấy contour hợp lệ nào
            if not valid_contours:
                return torch.zeros((1, 1, 28, 28), dtype=torch.float32)
            
            # Sắp xếp các contour theo diện tích giảm dần
            sorted_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
            
            # Lặp qua tất cả các contour hợp lệ để tìm contour phù hợp nhất
            for contour in sorted_contours[:3]:  # Chỉ xem xét top 3 contour lớn nhất
                # Lấy hình chữ nhật bao quanh contour
                x, y, w, h = cv2.boundingRect(contour)
                
                # Kiểm tra tỷ lệ khung hình
                aspect_ratio = float(w) / h if h > 0 else 0
                if not (0.2 < aspect_ratio < 2.5):  # Nới lỏng ràng buộc về tỷ lệ
                    continue
                
                # Đơn giản hóa kiểm tra độ phức tạp để giảm false negative
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                complexity = perimeter * perimeter / (4 * np.pi * area) if area > 0 else 0
                
                if complexity > 50:  # Quá phức tạp, có thể không phải chữ số
                    continue
                
                # Cắt vùng chứa contour, thêm padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(gray.shape[1] - x, w + 2*padding)
                h = min(gray.shape[0] - y, h + 2*padding)
                
                # Vẽ hình chữ nhật viền đỏ lên frame gốc để hiển thị vùng được phát hiện
                frame_with_roi = frame.copy()
                cv2.rectangle(frame_with_roi, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Đổi sang viền đỏ
                cv2.putText(frame_with_roi, f"AR: {aspect_ratio:.2f}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Hiển thị frame với ROI lên canvas
                roi_frame_rgb = cv2.cvtColor(frame_with_roi, cv2.COLOR_BGR2RGB)
                roi_img_pil = Image.fromarray(roi_frame_rgb)
                roi_img_pil = roi_img_pil.resize((self.canvas_width, self.canvas_height), Image.LANCZOS)
                self.root.after(0, lambda: self.update_camera_display_with_roi(ImageTk.PhotoImage(roi_img_pil), roi_img_pil))
                
                # Cắt vùng chứa chữ số
                digit_roi = closing[y:y+h, x:x+w]
                
                # Lưu ảnh ROI để debug
                cv2.imwrite(f"{debug_dir}/roi_{timestamp}.png", digit_roi)
                
                # Căn giữa chữ số trong hình vuông
                square_size = max(h, w) + 2*padding
                digit_img = np.zeros((square_size, square_size), dtype=np.uint8)
                x_offset = (square_size - w) // 2
                y_offset = (square_size - h) // 2
                digit_img[y_offset:y_offset+h, x_offset:x_offset+w] = digit_roi
                
                # Resize về 28x28
                digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
                
                # Đảm bảo độ tương phản cao - ngưỡng chặt với Otsu
                _, digit_img = cv2.threshold(digit_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                
                # Lưu ảnh đã xử lý để kiểm tra
                os.makedirs("processed_frames", exist_ok=True)
                cv2.imwrite(f"processed_frames/digit_{timestamp}.png", digit_img)
                
                # Chuyển thành tensor
                tensor = torch.tensor(digit_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
                
                # Áp dụng các biến đổi tương tự như trên MNIST dataset
                tensor = transforms.Normalize((0.1307,), (0.3081,))(tensor)
                
                return tensor
            
            # Nếu không tìm thấy contour phù hợp nào
            self.status_var.set("Không phát hiện được chữ số")
            return torch.zeros((1, 1, 28, 28), dtype=torch.float32)
        
        except Exception as e:
            # Log lỗi nếu có
            print(f"Lỗi khi xử lý ảnh camera: {e}")
            self.status_var.set(f"Lỗi xử lý ảnh: {str(e)}")
            return torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    
    def update_camera_display_with_roi(self, img_tk, img_pil):
        """Cập nhật hiển thị camera với vùng ROI được đánh dấu"""
        self.camera_photo = img_tk  # Giữ tham chiếu
        self.photo_image = img_pil  # Lưu ảnh gốc
        
        # Xóa canvas và hiển thị ảnh mới
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.camera_photo, anchor=tk.NW)
        
        # Hiển thị thông báo giúp người dùng
        self.status_var.set("Đang quét chữ số từ camera...")
    
    def on_closing(self):
        """Xử lý khi đóng ứng dụng"""
        # Dừng camera nếu đang chạy
        self.camera_active = False
        if self.camera_thread is not None and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        
        # Đóng ứng dụng
        self.root.destroy()

    def clear_results(self):
        """Xóa kết quả khi không phát hiện chữ số"""
        if self.last_prediction is not None:
            self.last_prediction = None
            self.prediction_history = []
            self.result_label.config(text="?")
            self.confidence_var.set("0%")
            self.ax.clear()
            self.ax.set_title("Phân phối xác suất các lớp")
            self.ax.set_xlabel("Lớp")
            self.ax.set_ylabel("Xác suất")
            self.ax.set_xticks(range(10))
            self.canvas_plot.draw()
            
            self.time_ax.clear()
            self.time_ax.set_title("Hoạt động neuron theo thời gian")
            self.time_ax.set_xlabel("Bước thời gian")
            self.time_ax.set_ylabel("Hoạt động")
            self.time_canvas_plot.draw()
            
            # Cập nhật status
            self.status_var.set("Đang chờ phát hiện chữ số...")

    def show_settings(self):
        """Hiện cửa sổ thiết lập cho camera"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Cài đặt Camera")
        settings_window.geometry("400x350")
        settings_window.resizable(False, False)
        settings_window.transient(self.root)  # Liên kết với cửa sổ chính
        
        # Tạo frame cho các điều khiển
        settings_frame = ttk.Frame(settings_window, padding="20")
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Ngưỡng tin cậy
        ttk.Label(settings_frame, text="Ngưỡng tin cậy:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=10)
        conf_frame = ttk.Frame(settings_frame)
        conf_frame.grid(row=0, column=1, sticky=tk.W+tk.E, pady=10)
        
        conf_scale = ttk.Scale(conf_frame, from_=0.0, to=1.0, orient="horizontal", 
                              value=self.min_confidence_threshold, length=200)
        conf_scale.pack(side=tk.LEFT, padx=5)
        conf_value = ttk.Label(conf_frame, text=f"{self.min_confidence_threshold:.2f}")
        conf_value.pack(side=tk.LEFT, padx=5)
        
        # Độ ổn định dự đoán
        ttk.Label(settings_frame, text="Độ ổn định dự đoán:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=10)
        stab_frame = ttk.Frame(settings_frame)
        stab_frame.grid(row=1, column=1, sticky=tk.W+tk.E, pady=10)
        
        stability_threshold_scale = ttk.Scale(stab_frame, from_=0.3, to=0.9, orient="horizontal", 
                                             value=0.6, length=200)
        stability_threshold_scale.pack(side=tk.LEFT, padx=5)
        stability_value = ttk.Label(stab_frame, text="0.60")
        stability_value.pack(side=tk.LEFT, padx=5)
        
        # Kích thước tối thiểu chữ số
        ttk.Label(settings_frame, text="Kích thước tối thiểu:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=10)
        size_frame = ttk.Frame(settings_frame)
        size_frame.grid(row=2, column=1, sticky=tk.W+tk.E, pady=10)
        
        size_scale = ttk.Scale(size_frame, from_=50, to=500, orient="horizontal", 
                              value=200, length=200)
        size_scale.pack(side=tk.LEFT, padx=5)
        size_value = ttk.Label(size_frame, text="200")
        size_value.pack(side=tk.LEFT, padx=5)
        
        # Thời gian xóa kết quả
        ttk.Label(settings_frame, text="Frame không phát hiện:", font=('Arial', 10, 'bold')).grid(row=3, column=0, sticky=tk.W, pady=10)
        clear_frame = ttk.Frame(settings_frame)
        clear_frame.grid(row=3, column=1, sticky=tk.W+tk.E, pady=10)
        
        clear_scale = ttk.Scale(clear_frame, from_=3, to=30, orient="horizontal", 
                               value=self.no_digit_threshold, length=200)
        clear_scale.pack(side=tk.LEFT, padx=5)
        clear_value = ttk.Label(clear_frame, text=str(self.no_digit_threshold))
        clear_value.pack(side=tk.LEFT, padx=5)
        
        # Số lượng mẫu lịch sử
        ttk.Label(settings_frame, text="Kích thước lịch sử:", font=('Arial', 10, 'bold')).grid(row=4, column=0, sticky=tk.W, pady=10)
        history_frame = ttk.Frame(settings_frame)
        history_frame.grid(row=4, column=1, sticky=tk.W+tk.E, pady=10)
        
        history_scale = ttk.Scale(history_frame, from_=3, to=15, orient="horizontal", 
                                 value=7, length=200)
        history_scale.pack(side=tk.LEFT, padx=5)
        history_value = ttk.Label(history_frame, text="7")
        history_value.pack(side=tk.LEFT, padx=5)
        
        # Cập nhật giá trị hiển thị khi di chuyển thanh trượt
        def update_conf_value(val):
            value = float(val)
            conf_value.config(text=f"{value:.2f}")
            
        def update_stability_value(val):
            value = float(val)
            stability_value.config(text=f"{value:.2f}")
            
        def update_size_value(val):
            value = int(float(val))
            size_value.config(text=str(value))
            
        def update_clear_value(val):
            value = int(float(val))
            clear_value.config(text=str(value))
            
        def update_history_value(val):
            value = int(float(val))
            history_value.config(text=str(value))
        
        conf_scale.config(command=update_conf_value)
        stability_threshold_scale.config(command=update_stability_value)
        size_scale.config(command=update_size_value)
        clear_scale.config(command=update_clear_value)
        history_scale.config(command=update_history_value)
        
        # Nút Áp dụng và Hủy
        buttons_frame = ttk.Frame(settings_window)
        buttons_frame.pack(pady=20, padx=20)
        
        def apply_settings():
            self.min_confidence_threshold = float(conf_scale.get())
            self.prediction_stability_threshold = float(stability_threshold_scale.get())
            self.no_digit_threshold = int(clear_scale.get())
            min_area = int(size_scale.get())
            max_history = int(history_scale.get())
            
            # Cập nhật lịch sử dự đoán với kích thước mới
            if len(self.prediction_history) > max_history:
                self.prediction_history = self.prediction_history[-max_history:]
            
            # Thông báo cập nhật
            self.status_var.set(f"Đã cập nhật cài đặt camera")
            settings_window.destroy()
        
        ttk.Button(buttons_frame, text="Áp dụng", command=apply_settings).pack(side=tk.LEFT, padx=10)
        ttk.Button(buttons_frame, text="Hủy", command=settings_window.destroy).pack(side=tk.LEFT, padx=10)
        
        # Tập trung vào cửa sổ cài đặt
        settings_window.focus_set()
        settings_window.grab_set()  # Khóa các tương tác với cửa sổ chính
        settings_window.wait_window()  # Đợi cửa sổ này đóng lại

    def capture_frame(self):
        """Chụp frame hiện tại từ camera và xử lý tĩnh"""
        if not self.camera_active or not hasattr(self, 'current_frame'):
            messagebox.showinfo("Thông báo", "Camera không hoạt động hoặc chưa có frame nào được chụp.")
            return
        
        try:
            # Lưu frame hiện tại
            frame = self.current_frame.copy()
            
            # Tạo thư mục lưu trữ nếu chưa có
            capture_dir = "captured_frames"
            os.makedirs(capture_dir, exist_ok=True)
            timestamp = int(time.time())
            filename = f"{capture_dir}/capture_{timestamp}.png"
            
            # Lưu ảnh gốc
            cv2.imwrite(filename, frame)
            
            # Tạm dừng camera để tập trung vào xử lý
            was_active = self.camera_active
            if was_active:
                # Tạm dừng camera nhưng không đổi trạng thái nút
                self.camera_active = False
                self.status_var.set(f"Đã chụp ảnh và lưu vào {filename}")
            
            # Xử lý ảnh tĩnh với nhiều thông số khác nhau
            self.process_static_frame(frame)
            
            # Khôi phục camera nếu cần
            if was_active:
                self.camera_active = True
            
            # Thông báo thành công
            messagebox.showinfo("Thành công", f"Đã chụp và lưu ảnh vào {filename}")
        
        except Exception as e:
            error_msg = f"Lỗi khi chụp ảnh: {str(e)}"
            print(error_msg)
            messagebox.showerror("Lỗi", error_msg)

    def process_static_frame(self, frame):
        """Xử lý frame tĩnh với nhiều thông số khác nhau để tăng khả năng nhận dạng"""
        # Tạo thư mục cho các ảnh đã xử lý
        processed_dir = "static_processed"
        os.makedirs(processed_dir, exist_ok=True)
        timestamp = int(time.time())
        
        # Chuyển về ảnh xám
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Lưu ảnh gốc xám
        cv2.imwrite(f"{processed_dir}/gray_{timestamp}.png", gray)
        
        # Thử với nhiều phương pháp tiền xử lý khác nhau
        results = []
        
        # Phương pháp 1: Cân bằng histogram
        img1 = cv2.equalizeHist(gray)
        cv2.imwrite(f"{processed_dir}/histeq_{timestamp}.png", img1)
        
        # Phương pháp 2: CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img2 = clahe.apply(gray)
        cv2.imwrite(f"{processed_dir}/clahe_{timestamp}.png", img2)
        
        # Phương pháp 3: Lọc Gaussian
        img3 = cv2.GaussianBlur(gray, (5, 5), 0)
        cv2.imwrite(f"{processed_dir}/gaussian_{timestamp}.png", img3)
        
        # Các phương pháp ngưỡng khác nhau
        methods = [
            ("otsu", lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]),
            ("adaptive", lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                        cv2.THRESH_BINARY_INV, 11, 2))
        ]
        
        # Áp dụng các phương pháp ngưỡng cho mỗi ảnh đã tiền xử lý
        base_images = [gray, img1, img2, img3]
        base_names = ["gray", "histeq", "clahe", "gaussian"]
        
        for i, (base_img, base_name) in enumerate(zip(base_images, base_names)):
            for method_name, method_func in methods:
                try:
                    # Áp dụng phương pháp ngưỡng
                    thresh_img = method_func(base_img)
                    
                    # Lưu ảnh sau ngưỡng
                    output_name = f"{processed_dir}/{base_name}_{method_name}_{timestamp}.png"
                    cv2.imwrite(output_name, thresh_img)
                    
                    # Tìm contours
                    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Vẽ contours lên ảnh gốc
                    if contours:
                        viz_img = cv2.cvtColor(thresh_img.copy(), cv2.COLOR_GRAY2BGR)
                        cv2.drawContours(viz_img, contours, -1, (0, 255, 0), 2)
                        cv2.imwrite(f"{processed_dir}/{base_name}_{method_name}_contours_{timestamp}.png", viz_img)
                    
                    # Tìm contour lớn nhất
                    if contours:
                        max_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(max_contour)
                        
                        # Cắt vùng chữ số
                        padding = 10
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(thresh_img.shape[1] - x, w + 2*padding)
                        h = min(thresh_img.shape[0] - y, h + 2*padding)
                        
                        # Cắt và chuẩn bị cho model
                        digit_img = thresh_img[y:y+h, x:x+w]
                        
                        # Resize về kích thước MNIST
                        if digit_img.size > 0:
                            # Căn giữa trong khung vuông
                            square_size = max(h, w)
                            squared_img = np.zeros((square_size, square_size), dtype=np.uint8)
                            x_offset = (square_size - w) // 2
                            y_offset = (square_size - h) // 2
                            squared_img[y_offset:y_offset+h, x_offset:x_offset+w] = digit_img
                            
                            # Resize về 28x28
                            mnist_img = cv2.resize(squared_img, (28, 28), interpolation=cv2.INTER_AREA)
                            
                            # Lưu ảnh đã xử lý
                            cv2.imwrite(f"{processed_dir}/{base_name}_{method_name}_mnist_{timestamp}.png", mnist_img)
                            
                            # Chuyển thành tensor
                            tensor = torch.tensor(mnist_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
                            tensor = transforms.Normalize((0.1307,), (0.3081,))(tensor)
                            
                            # Thực hiện dự đoán
                            if self.model is not None:
                                predicted, confidence, spikes = predict(self.model, tensor, self.model_type)
                                
                                # Thêm vào danh sách kết quả
                                results.append((predicted, confidence, tensor, spikes, 
                                             f"{base_name}_{method_name}", mnist_img))
                except Exception as e:
                    print(f"Lỗi khi xử lý {base_name}_{method_name}: {e}")
        
        # Nếu có kết quả, hiển thị kết quả với độ tin cậy cao nhất
        if results:
            # Sắp xếp theo độ tin cậy giảm dần
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Lấy kết quả tốt nhất
            best_pred, best_conf, best_tensor, best_spikes, best_method, best_img = results[0]
            
            # Cập nhật UI
            self.root.after(0, lambda: self.update_ui(best_pred, best_conf, best_spikes, best_tensor))
            
            # Hiển thị thông báo
            self.status_var.set(f"Dự đoán từ ảnh tĩnh: {best_pred} (Độ tin cậy: {best_conf*100:.2f}%, phương pháp: {best_method})")
            
            # Lưu top 3 kết quả tốt nhất vào file text
            with open(f"{processed_dir}/results_{timestamp}.txt", "w") as f:
                f.write(f"Top kết quả cho ảnh chụp:\n")
                for i, (pred, conf, _, _, method, _) in enumerate(results[:3]):
                    f.write(f"{i+1}. Dự đoán: {pred}, Độ tin cậy: {conf*100:.2f}%, Phương pháp: {method}\n")
        else:
            messagebox.showinfo("Thông báo", "Không thể phát hiện được chữ số trong ảnh.")

def main():
    root = tk.Tk()
    app = SNNApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 