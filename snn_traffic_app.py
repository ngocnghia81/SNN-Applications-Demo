"""
Ứng dụng phân tích giao thông thông minh sử dụng SNN
Giám sát phương tiện, phát hiện vi phạm và trực quan hóa hoạt động neuron
"""

import os
import sys
import time
import threading
import queue
import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import argparse  # Thêm thư viện xử lý tham số command line
from collections import defaultdict
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
# Khắc phục lỗi hiển thị font tiếng Việt trên matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

# Import các module đã tạo
from snn_traffic_model import Config, load_model, preprocess_image, predict, visualize_neuron_activity, generate_sample_spike_activity
from snn_traffic_detector import TrafficDetector, TrafficViolationDetector

class TrafficMonitorApp:
    """Ứng dụng giám sát giao thông thông minh với SNN"""
    
    def __init__(self, root, video_source=None):
        """Khởi tạo ứng dụng
        
        Args:
            root: Cửa sổ chính
            video_source: Đường dẫn đến file video (tùy chọn)
        """
        self.root = root
        self.root.title("Hệ thống Giám sát Giao thông SNN")
        self.root.geometry("1200x800")
        self.root.configure(background="#f0f0f0")
        
        # Cấu hình
        self.config = Config()
        self.detector = None
        self.violation_detector = None
        self.model = None
        
        # Bộ theo dõi thời gian thực
        self.camera_source = 0  # Webcam mặc định
        self.video_source = video_source  # Lưu nguồn video từ tham số
        self.is_running = False
        self.cap = None
        
        # Xử lý đa luồng
        self.processing_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue()
        
        # Cờ điều khiển
        self.show_stats = True
        self.show_tracking = True
        self.show_violations = True
        self.show_neuron_activity = False
        
        # Cờ trạng thái
        self.is_loading = True
        self.status_message = "Đang khởi tạo..."
        
        # Cờ báo đóng ứng dụng
        self.is_closing = False
        
        # Neuron Activities
        self.neuron_figures = {}
        
        # Tạo giao diện người dùng
        self.create_widgets()
        
        # Tải mô hình và khởi tạo bộ phát hiện trong luồng riêng
        self.load_thread = threading.Thread(target=self.load_models)
        self.load_thread.daemon = True
        self.load_thread.start()
        
        # Luồng xử lý khung hình
        self.process_thread = threading.Thread(target=self.process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Tạo thư mục lưu trữ
        os.makedirs("captures", exist_ok=True)
        os.makedirs("detection_results", exist_ok=True)
        
        # Thiết lập handler khi đóng cửa sổ
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Nếu có nguồn video, tự động mở khi khởi động
        if self.video_source:
            # Khởi chạy sau 1 giây để đảm bảo các models đã được tải
            self.root.after(1000, self.auto_start_video)
    
    def auto_start_video(self):
        """Tự động bắt đầu phát video nếu được chỉ định qua tham số"""
        if not self.is_loading and self.video_source:
            self.load_video(self.video_source)
    
    def create_widgets(self):
        """Tạo các thành phần giao diện người dùng"""
        # Main layout
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame bên trái (Hiển thị video)
        self.left_frame = ttk.Frame(self.main_paned, padding=5)
        self.main_paned.add(self.left_frame, weight=3)
        
        # Frame bên phải (Kết quả phân tích và điều khiển)
        self.right_frame = ttk.Frame(self.main_paned, padding=5)
        self.main_paned.add(self.right_frame, weight=1)
        
        # === Frame bên trái (Hiển thị video và camera) ===
        # Frame video
        self.video_frame = ttk.LabelFrame(self.left_frame, text="Nguồn Video")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas hiển thị video
        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Khung điều khiển video
        self.video_controls = ttk.Frame(self.left_frame)
        self.video_controls.pack(fill=tk.X, padx=5, pady=5)
        
        # Nút điều khiển camera
        ttk.Button(self.video_controls, text="Bật Camera", command=self.toggle_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.video_controls, text="Tải Video", command=self.load_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.video_controls, text="Chụp Ảnh", command=self.capture_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.video_controls, text="Reset Ứng Dụng", command=self.reset_app).pack(side=tk.LEFT, padx=5)
        
        # === Frame bên phải (Kết quả và điều khiển) ===
        # Frame điều khiển
        self.control_frame = ttk.LabelFrame(self.right_frame, text="Điều Khiển")
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Checkboxes for display options
        self.show_stats_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Hiển thị Thống kê", 
                       variable=self.show_stats_var, 
                       command=self.update_display_options).pack(anchor=tk.W, padx=5, pady=2)
        
        self.show_tracking_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Hiển thị Theo dõi", 
                       variable=self.show_tracking_var,
                       command=self.update_display_options).pack(anchor=tk.W, padx=5, pady=2)
        
        self.show_violations_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Hiển thị Vi phạm", 
                       variable=self.show_violations_var,
                       command=self.update_display_options).pack(anchor=tk.W, padx=5, pady=2)
        
        self.show_neuron_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="Hiển thị Hoạt động Neuron", 
                       variable=self.show_neuron_var,
                       command=self.update_display_options).pack(anchor=tk.W, padx=5, pady=2)
        
        # Frame speed limit
        speed_frame = ttk.Frame(self.control_frame)
        speed_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(speed_frame, text="Giới hạn Tốc độ (km/h):").pack(side=tk.LEFT)
        
        self.speed_limit_var = tk.IntVar(value=50)
        self.speed_scale = ttk.Scale(speed_frame, from_=20, to=120, 
                                   orient=tk.HORIZONTAL, variable=self.speed_limit_var,
                                   command=self.update_speed_limit)
        self.speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(speed_frame, textvariable=self.speed_limit_var).pack(side=tk.LEFT)
        
        # Frame kết quả phân tích
        self.result_frame = ttk.LabelFrame(self.right_frame, text="Kết Quả Phân Tích")
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tạo notebook cho các tab kết quả
        self.result_notebook = ttk.Notebook(self.result_frame)
        self.result_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Thống kê
        self.stats_frame = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.stats_frame, text="Thống Kê")
        
        # Plot thống kê
        self.stats_fig = Figure(figsize=(4, 3), dpi=100)
        self.stats_ax = self.stats_fig.add_subplot(111)
        self.stats_ax.set_title("Thống kê Phương tiện")
        self.stats_canvas = FigureCanvasTkAgg(self.stats_fig, self.stats_frame)
        self.stats_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 2: Vi phạm
        self.violations_frame = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.violations_frame, text="Vi Phạm")
        
        # Treeview cho vi phạm
        columns = ("time", "vehicle", "type", "details")
        self.violations_tree = ttk.Treeview(self.violations_frame, columns=columns, show="headings")
        self.violations_tree.heading("time", text="Thời gian")
        self.violations_tree.heading("vehicle", text="Phương tiện")
        self.violations_tree.heading("type", text="Loại Vi phạm")
        self.violations_tree.heading("details", text="Chi tiết")
        
        self.violations_tree.column("time", width=100)
        self.violations_tree.column("vehicle", width=80)
        self.violations_tree.column("type", width=100)
        self.violations_tree.column("details", width=120)
        
        self.violations_tree.pack(fill=tk.BOTH, expand=True)
        
        # Tab 3: Hoạt động neuron
        self.neuron_frame = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.neuron_frame, text="Hoạt động Neuron")
        
        # Tạo canvas cho hoạt động neuron
        self.neuron_fig = Figure(figsize=(4, 3), dpi=100)
        self.neuron_ax = self.neuron_fig.add_subplot(111)
        self.neuron_ax.set_title("Hoạt động Neuron")
        self.neuron_canvas = FigureCanvasTkAgg(self.neuron_fig, self.neuron_frame)
        self.neuron_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Thanh trạng thái
        self.status_var = tk.StringVar(value="Đang khởi tạo...")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_models(self):
        """Tải mô hình và khởi tạo các bộ phát hiện"""
        try:
            self.status_var.set("Đang tải mô hình YOLO...")
            
            # Tạo bộ phát hiện phương tiện giao thông với YOLOv5
            self.detector = TrafficDetector(yolo_model='yolov5s')
            
            # Tạo bộ phát hiện vi phạm
            self.violation_detector = TrafficViolationDetector(speed_limit=self.speed_limit_var.get())
            
            # Tạo vùng đèn đỏ mẫu (chỉ để demo)
            width, height = 640, 480  # Kích thước mặc định
            red_light_zone = np.array([
                [width//2 - 100, height//2 - 50],
                [width//2 + 100, height//2 - 50],
                [width//2 + 100, height//2 + 50],
                [width//2 - 100, height//2 + 50]
            ], np.int32)
            self.violation_detector.set_red_light_zone(red_light_zone)
            
            self.status_var.set("Đang tải mô hình SNN...")
            
            # Tải mô hình SNN (nếu đã có)
            model_path = self.config.get_model_path()
            if os.path.exists(model_path):
                self.model = load_model(model_path, num_classes=self.config.num_classes, device=self.config.device)
                self.status_var.set("Tải mô hình hoàn tất")
            else:
                self.status_var.set("Không tìm thấy mô hình SNN đã huấn luyện. Chỉ sử dụng phát hiện YOLO.")
            
            # Cập nhật trạng thái
            self.is_loading = False
            self.status_var.set("Sẵn sàng")
            
            # Khởi tạo dữ liệu biểu đồ trống
            self.update_stats_plot()
            
            # Khởi tạo biểu đồ neuron trống
            self.update_neuron_plot([np.zeros(10)])
        
        except Exception as e:
            self.is_loading = False
            error_msg = f"Lỗi khi tải mô hình: {str(e)}"
            self.status_var.set(error_msg)
            messagebox.showerror("Lỗi", error_msg)
            
            import traceback
            traceback.print_exc()
    
    def toggle_camera(self):
        """Bật/tắt camera"""
        if self.is_running:
            self.stop_video()
            self.video_controls.winfo_children()[0].configure(text="Bật Camera")
        else:
            self.start_camera()
            self.video_controls.winfo_children()[0].configure(text="Tắt Camera")
    
    def start_camera(self):
        """Bắt đầu luồng lấy khung hình từ camera"""
        if self.is_running:
            return
        
        try:
            self.cap = cv2.VideoCapture(self.camera_source)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Không thể mở camera {self.camera_source}")
                return
            
            # Lấy kích thước khung hình
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            # Cập nhật vùng đèn đỏ theo kích thước khung hình
            red_light_zone = np.array([
                [width//2 - 100, height//2 - 50],
                [width//2 + 100, height//2 - 50],
                [width//2 + 100, height//2 + 50],
                [width//2 - 100, height//2 + 50]
            ], np.int32)
            self.violation_detector.set_red_light_zone(red_light_zone)
            
            # Bắt đầu luồng lấy khung hình
            self.is_running = True
            self.update_frame()
            self.status_var.set("Đang chạy camera")
        
        except Exception as e:
            messagebox.showerror("Error", f"Lỗi khi khởi động camera: {str(e)}")
    
    def load_video(self, video_path=None):
        """Tải video từ file
        
        Args:
            video_path: Đường dẫn đến file video (tùy chọn)
        """
        if self.is_running:
            self.stop_video()
        
        # Reset thống kê và theo dõi khi tải video mới
        if hasattr(self, 'detector') and self.detector is not None:
            self.detector.tracked_objects.clear()
            self.detector.next_obj_id = 0
            
            # Xóa thống kê cũ nhưng giữ lại cấu trúc
            for class_name in list(self.detector.traffic_stats["by_class"].keys()):
                self.detector.traffic_stats["by_class"][class_name] = 0
            self.detector.traffic_stats["total_count"] = 0
            self.detector.traffic_stats["avg_speed"] = 0
            self.detector.traffic_stats["violations"] = 0
        
        if hasattr(self, 'violation_detector') and self.violation_detector is not None:
            self.violation_detector.violations.clear()
        
        # Xóa danh sách vi phạm
        for item in self.violations_tree.get_children():
            self.violations_tree.delete(item)
        
        # Reset biểu đồ
        self.update_stats_plot()
        self.update_neuron_plot([np.zeros(10)])
            
        # Nếu không có đường dẫn, mở hộp thoại chọn file
        if not video_path:
            video_path = filedialog.askopenfilename(
                title="Chọn video",
                filetypes=[("Tập tin video", "*.mp4 *.avi *.mov"), ("Tất cả tập tin", "*.*")]
            )
        
        if not video_path:
            return
        
        try:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", f"Không thể mở video: {video_path}")
                return
            
            # Lấy kích thước khung hình
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            # Cập nhật vùng đèn đỏ
            red_light_zone = np.array([
                [width//2 - 100, height//2 - 50],
                [width//2 + 100, height//2 - 50],
                [width//2 + 100, height//2 + 50],
                [width//2 - 100, height//2 + 50]
            ], np.int32)
            
            if self.violation_detector:
                self.violation_detector.set_red_light_zone(red_light_zone)
            
            # Bắt đầu
            self.video_source = video_path
            self.is_running = True
            
            # Cập nhật nút camera
            self.video_controls.winfo_children()[0].configure(text="Tắt Video")
            
            self.update_frame()
            self.status_var.set(f"Đang phát video: {os.path.basename(video_path)}")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi tải video: {str(e)}")
    
    def update_frame(self):
        """Cập nhật khung hình từ camera/video và gửi đến hàng đợi xử lý"""
        if not self.is_running:
            return
        
        if self.cap is None or not self.cap.isOpened():
            self.stop_video()
            messagebox.showerror("Lỗi", "Mất kết nối với camera/video")
            return
        
        ret, frame = self.cap.read()
        
        if not ret:
            # Đã đến cuối video
            self.stop_video()
            self.status_var.set("Kết thúc video")
            return
        
        # Thay đổi kích thước khung hình nếu cần
        if frame.shape[1] > 800:
            scale = 800 / frame.shape[1]
            width = int(frame.shape[1] * scale)
            height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (width, height))
        
        # Gửi khung hình đến hàng đợi xử lý nếu hàng đợi không đầy
        try:
            if not self.processing_queue.full():
                self.processing_queue.put(frame, block=False)
        except queue.Full:
            pass  # Bỏ qua nếu hàng đợi đầy
        
        # Hiển thị khung hình gốc hoặc đã xử lý
        if not self.result_queue.empty():
            try:
                processed_frame = self.result_queue.get(block=False)
                self.show_frame(processed_frame)
                
                # Kích hoạt cập nhật biểu đồ thống kê định kỳ (10 khung hình cập nhật 1 lần)
                if hasattr(self, 'frame_count'):
                    self.frame_count += 1
                else:
                    self.frame_count = 0
                
                if self.frame_count % 10 == 0:
                    self.update_stats_plot()
                    
                # Cập nhật vi phạm nếu có
                self.update_violations_list()
            except queue.Empty:
                self.show_frame(frame)
        else:
            self.show_frame(frame)
        
        # Lặp lại sau một thời gian
        self.root.after(15, self.update_frame)
    
    def process_frames(self):
        """Luồng xử lý khung hình"""
        while not self.is_closing:
            try:
                # Lấy khung hình từ hàng đợi
                frame = self.processing_queue.get(timeout=0.5)
                
                if self.is_loading or self.detector is None:
                    # Chưa tải xong mô hình, gửi lại khung hình gốc
                    self.result_queue.put(frame)
                    continue
                
                # Phát hiện phương tiện
                detections, processed_frame = self.detector.detect(frame)
                
                # Phát hiện vi phạm nếu được bật
                if self.show_violations_var.get():
                    processed_frame = self.violation_detector.detect_violations(
                        self.detector.tracked_objects, processed_frame)
                
                # Vẽ biểu đồ thống kê nếu được bật
                if self.show_stats_var.get():
                    processed_frame = self.detector.draw_traffic_stats(processed_frame)
                
                # Trích xuất phương tiện để phân loại SNN (nếu có mô hình)
                if self.model is not None and self.show_neuron_var.get():
                    # Định kỳ cập nhật biểu đồ hoạt động neuron
                    if hasattr(self, 'neuron_update_counter'):
                        self.neuron_update_counter += 1
                    else:
                        self.neuron_update_counter = 0
                    
                    # Cập nhật biểu đồ neuron mỗi 20 khung hình hoặc khi có phương tiện mới
                    if self.neuron_update_counter % 20 == 0 or len(detections) > 0:
                        try:
                            if len(detections) > 0:
                                # Có phương tiện được phát hiện, sử dụng để phân tích
                                vehicles = self.detector.extract_vehicles(frame, detections)
                                if vehicles and len(vehicles) > 0:
                                    # Lấy phương tiện đầu tiên để phân tích
                                    vehicle_img, class_id, _ = vehicles[0]
                                    
                                    # Kiểm tra phương tiện có hợp lệ không
                                    if vehicle_img is not None and vehicle_img.size > 0 and vehicle_img.shape[0] > 0 and vehicle_img.shape[1] > 0:
                                        # Tiền xử lý và dự đoán
                                        vehicle_tensor = preprocess_image(vehicle_img)
                                        pred_class, confidence, spike_activities = predict(
                                            self.model, vehicle_tensor, self.config.device, self.config)
                                        
                                        # Trực quan hóa hoạt động neuron
                                        if spike_activities:
                                            self.root.after(0, lambda: self.update_neuron_plot(spike_activities))
                                    else:
                                        # Nếu vehicle_img không hợp lệ, tạo dữ liệu mẫu
                                        self._update_neuron_with_sample_data()
                                else:
                                    # Không thể trích xuất phương tiện, tạo dữ liệu mẫu
                                    self._update_neuron_with_sample_data()
                            else:
                                # Không có phương tiện, tạo dữ liệu mẫu nếu đến thời điểm cập nhật
                                self._update_neuron_with_sample_data()
                        except Exception as e:
                            print(f"Lỗi khi xử lý phương tiện: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            # Tạo dữ liệu mẫu khi có lỗi
                            self._update_neuron_with_sample_data()
                
                # Gửi khung hình đã xử lý
                self.result_queue.put(processed_frame)
                
                # Làm trống hàng đợi xử lý
                self.processing_queue.task_done()
            
            except queue.Empty:
                # Không có khung hình mới để xử lý
                pass
            except Exception as e:
                print(f"Lỗi khi xử lý khung hình: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def _update_neuron_with_sample_data(self):
        """Cập nhật biểu đồ neuron với dữ liệu mẫu"""
        from snn_traffic_model import generate_sample_spike_activity
        spike_activities = generate_sample_spike_activity(3)
        self.root.after(0, lambda: self.update_neuron_plot(spike_activities))
    
    def show_frame(self, frame):
        """Hiển thị khung hình lên canvas"""
        # Chuyển từ BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Chuyển sang định dạng Tkinter
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Cập nhật kích thước canvas
        self.canvas.config(width=img.width, height=img.height)
        
        # Hiển thị hình ảnh
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk  # Giữ tham chiếu
    
    def update_neuron_plot(self, spike_activities):
        """Cập nhật biểu đồ hoạt động neuron"""
        if not self.show_neuron_var.get():
            return
        
        if not spike_activities:
            # Tạo dữ liệu mẫu nếu không có hoạt động
            sample_activity = np.random.rand(10, 10) > 0.7
            spike_activities = [sample_activity.astype(float)]
        
        self.neuron_ax.clear()
        
        # Tìm lớp có dữ liệu để hiển thị
        selected_layer = None
        for i, activity in enumerate(spike_activities):
            if activity.size > 0:
                selected_layer = (i, activity)
                break
        
        # Nếu không tìm thấy lớp nào có dữ liệu
        if selected_layer is None:
            # Tạo dữ liệu mẫu từ hàm generate_sample_spike_activity
            from snn_traffic_model import generate_sample_spike_activity
            sample_data = generate_sample_spike_activity(1)[0]
            
            # Sử dụng dữ liệu mẫu mới tạo
            self._plot_neuron_heatmap(sample_data)
            self.neuron_ax.set_title("Dữ liệu Neuron - Mẫu", fontsize=12, fontweight='bold')
        else:
            layer_idx, activity = selected_layer
            
            # Nếu hoạt động có nhiều chiều, lấy chiều đầu tiên
            if activity.ndim > 1:
                data = activity
            else:
                # Nếu chỉ có 1 chiều, mở rộng thành 2 chiều để vẽ heatmap
                data = np.expand_dims(activity, axis=0)
            
            # Vẽ heatmap hoạt động neuron
            self._plot_neuron_heatmap(data)
            self.neuron_ax.set_title(f"Hoạt động Neuron - Lớp {layer_idx}", fontsize=12, fontweight='bold')
        
        # Cập nhật canvas
        self.neuron_canvas.draw()
    
    def _plot_neuron_heatmap(self, activity_data):
        """Vẽ biểu đồ heatmap hoạt động neuron"""
        # Đảm bảo dữ liệu là 2D
        if activity_data.ndim == 1:
            activity_data = np.expand_dims(activity_data, axis=0)
        
        # Tạo dữ liệu x-axis (thời gian) và y-axis (neuron)
        time_steps = activity_data.shape[0]
        num_neurons = min(activity_data.shape[1], 20)  # Giới hạn số neuron hiển thị
        
        # Chọn subset neuron nếu có quá nhiều
        if activity_data.shape[1] > num_neurons:
            # Chọn các neuron có hoạt động nhiều nhất
            activity_sums = np.sum(activity_data, axis=0)
            top_indices = np.argsort(activity_sums)[-num_neurons:]
            activity_data = activity_data[:, top_indices]
        
        # Tùy chọn 1: Heatmap - hiển thị hoạt động của tất cả neuron theo thời gian
        im = self.neuron_ax.imshow(
            activity_data.T,  # Chuyển vị để neuron nằm trên y-axis
            cmap='viridis',   # Colormap
            aspect='auto',    # Tự động điều chỉnh tỷ lệ
            interpolation='nearest'
        )
        
        # Thêm colorbar
        cbar = plt.colorbar(im, ax=self.neuron_ax)
        cbar.set_label('Mức độ kích hoạt')
        
        # Thiết lập trục
        self.neuron_ax.set_xlabel('Thời gian', fontsize=10)
        self.neuron_ax.set_ylabel('Neuron', fontsize=10)
        
        # Thiết lập ticks cho trục thời gian
        time_indices = np.linspace(0, time_steps-1, min(time_steps, 10), dtype=int)
        self.neuron_ax.set_xticks(time_indices)
        
        # Thiết lập ticks cho trục neuron
        neuron_indices = np.linspace(0, num_neurons-1, min(num_neurons, 10), dtype=int)
        self.neuron_ax.set_yticks(neuron_indices)
        
        # Thêm lưới
        self.neuron_ax.grid(False)
        
        # Làm đẹp
        self.neuron_fig.tight_layout()
    
    def update_stats_plot(self):
        """Cập nhật biểu đồ thống kê"""
        if self.detector is None:
            return
        
        stats = self.detector.get_traffic_stats()
        
        self.stats_ax.clear()
        labels = []
        values = []
        
        for class_name, count in stats['by_class'].items():
            if count > 0:
                labels.append(class_name)
                values.append(count)
        
        if not labels:
            # Thêm dữ liệu mẫu nếu không có dữ liệu thực
            if not hasattr(self, 'dummy_stats_shown') or not self.dummy_stats_shown:
                labels = ["Ô tô", "Xe máy", "Xe buýt", "Xe tải"]
                values = [0, 0, 0, 0]
                self.dummy_stats_shown = True
        
        # Tạo màu sắc cho các cột
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        # Vẽ biểu đồ cột với màu sắc
        bars = self.stats_ax.bar(labels, values, color=colors[:len(labels)])
        
        # Thêm giá trị lên đầu mỗi cột
        for bar in bars:
            height = bar.get_height()
            self.stats_ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                              f'{int(height)}', ha='center', va='bottom')
        
        self.stats_ax.set_title("Thống kê Phương tiện", fontsize=12, fontweight='bold')
        self.stats_ax.set_ylabel("Số lượng", fontsize=10)
        
        # Chỉnh màu nền và lưới
        self.stats_ax.set_facecolor('#f8f9fa')
        self.stats_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Cập nhật canvas
        self.stats_canvas.draw()
    
    def capture_frame(self):
        """Chụp khung hình hiện tại"""
        if not self.is_running or self.cap is None:
            messagebox.showinfo("Info", "Không có khung hình nào để chụp")
            return
        
        # Lấy khung hình hiện tại
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Không thể chụp khung hình")
            return
        
        # Tạo tên file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"captures/frame_{timestamp}.jpg"
        
        # Lưu khung hình
        cv2.imwrite(filename, frame)
        
        # Phát hiện phương tiện
        if self.detector:
            detections, annotated_frame = self.detector.detect(frame)
            result_filename = f"detection_results/detection_{timestamp}.jpg"
            cv2.imwrite(result_filename, annotated_frame)
            
            self.status_var.set(f"Đã lưu khung hình vào {filename} và kết quả phát hiện vào {result_filename}")
        else:
            self.status_var.set(f"Đã lưu khung hình vào {filename}")
    
    def update_display_options(self):
        """Cập nhật các tùy chọn hiển thị"""
        self.show_stats = self.show_stats_var.get()
        self.show_tracking = self.show_tracking_var.get()
        self.show_violations = self.show_violations_var.get()
        self.show_neuron_activity = self.show_neuron_var.get()
    
    def update_speed_limit(self, value=None):
        """Cập nhật giới hạn tốc độ"""
        if self.violation_detector:
            self.violation_detector.speed_limit = self.speed_limit_var.get()
    
    def stop_video(self):
        """Dừng video/camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status_var.set("Đã dừng")
        
        # Cập nhật nút camera
        self.video_controls.winfo_children()[0].configure(text="Bật Camera")
    
    def on_closing(self):
        """Xử lý khi đóng ứng dụng"""
        self.is_closing = True
        self.stop_video()
        
        # Dừng các luồng
        if self.processing_queue:
            # Xóa tất cả các mục trong hàng đợi
            while not self.processing_queue.empty():
                try:
                    self.processing_queue.get_nowait()
                    self.processing_queue.task_done()
                except:
                    pass
        
        # Đóng cửa sổ
        self.root.destroy()

    def update_violations_list(self):
        """Cập nhật danh sách vi phạm trên giao diện"""
        if not hasattr(self, 'violation_detector') or self.violation_detector is None:
            return
            
        violations = self.violation_detector.get_violations()
        
        # Xóa danh sách cũ
        for item in self.violations_tree.get_children():
            self.violations_tree.delete(item)
            
        # Thêm vi phạm mới
        for v in violations:
            time_str = time.strftime("%H:%M:%S", time.localtime(v.timestamp))
            self.violations_tree.insert("", tk.END, values=(time_str, v.vehicle_class, v.violation_type, v.details))

    def reset_app(self):
        """Reset ứng dụng về trạng thái ban đầu"""
        # Dừng video hiện tại nếu đang chạy
        self.stop_video()
        
        # Xóa dữ liệu hiện tại
        if hasattr(self, 'detector') and self.detector is not None:
            self.detector.tracked_objects.clear()
            self.detector.traffic_stats = {
                "by_class": defaultdict(int),
                "total_count": 0,
                "avg_speed": 0,
                "violations": 0
            }
            self.detector.next_obj_id = 0
        
        if hasattr(self, 'violation_detector') and self.violation_detector is not None:
            self.violation_detector.violations.clear()
        
        # Reset biểu đồ thống kê
        self.stats_ax.clear()
        self.stats_ax.set_title("Thống kê Phương tiện")
        self.stats_canvas.draw()
        
        # Reset biểu đồ neuron
        self.neuron_ax.clear()
        self.neuron_ax.set_title("Hoạt động Neuron")
        self.neuron_canvas.draw()
        
        # Xóa danh sách vi phạm
        for item in self.violations_tree.get_children():
            self.violations_tree.delete(item)
        
        # Reset chỉ số khung hình
        if hasattr(self, 'frame_count'):
            self.frame_count = 0
        
        # Hiển thị hình ảnh trống trên canvas
        empty_image = np.ones((480, 640, 3), dtype=np.uint8) * 0  # Ảnh đen
        self.show_frame(empty_image)
        
        # Cập nhật trạng thái
        self.status_var.set("Đã reset ứng dụng. Sẵn sàng.")

def parse_args():
    """Xử lý tham số dòng lệnh"""
    parser = argparse.ArgumentParser(description="SNN Traffic Monitoring System")
    parser.add_argument("--video", type=str, help="Đường dẫn đến file video để phân tích")
    parser.add_argument("--camera", type=int, default=0, help="ID của camera để sử dụng (mặc định: 0)")
    return parser.parse_args()

if __name__ == "__main__":
    # Xử lý tham số dòng lệnh
    args = parse_args()
    
    # Khởi tạo ứng dụng
    root = tk.Tk()
    app = TrafficMonitorApp(root, video_source=args.video)
    
    # Nếu được chỉ định camera khác
    if args.camera != 0:
        app.camera_source = args.camera
    
    root.mainloop() 