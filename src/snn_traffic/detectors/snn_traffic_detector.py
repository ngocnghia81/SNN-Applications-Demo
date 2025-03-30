"""
Module phát hiện giao thông và vi phạm giao thông
"""

import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
import random
from collections import defaultdict, deque

class TrafficViolation:
    """Lớp đại diện cho một vi phạm giao thông"""
    def __init__(self, vehicle_id, vehicle_class, violation_type, details="", timestamp=None):
        self.vehicle_id = vehicle_id
        self.vehicle_class = vehicle_class
        self.violation_type = violation_type
        self.details = details
        self.timestamp = timestamp or time.time()

class TrackedObject:
    """Lớp theo dõi đối tượng qua các khung hình"""
    def __init__(self, obj_id, class_id, class_name, bbox, confidence, color=None):
        self.id = obj_id
        self.class_id = class_id
        self.class_name = class_name
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.color = color or (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Lịch sử vị trí
        self.positions = [(time.time(), self.bbox_center())]
        
        # Tính toán tốc độ
        self.speed = 0  # km/h
        self.last_speed_update = time.time()
        
        # Trạng thái
        self.violations = []
    
    def update(self, bbox, confidence):
        """Cập nhật thông tin đối tượng"""
        self.bbox = bbox
        self.confidence = confidence
        self.positions.append((time.time(), self.bbox_center()))
        
        # Giới hạn số lượng vị trí lưu trữ
        if len(self.positions) > 20:
            self.positions.pop(0)
        
        # Cập nhật tốc độ định kỳ
        current_time = time.time()
        if current_time - self.last_speed_update > 0.5:  # Cập nhật mỗi 0.5 giây
            self.calculate_speed()
            self.last_speed_update = current_time
    
    def bbox_center(self):
        """Tính toán tâm của bbox"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def calculate_speed(self):
        """Tính toán tốc độ dựa trên thay đổi vị trí"""
        if len(self.positions) < 2:
            return
        
        # Lấy vị trí đầu và cuối
        start_time, start_pos = self.positions[0]
        end_time, end_pos = self.positions[-1]
        
        # Tính khoảng cách (pixels)
        distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # Tính thời gian (giây)
        time_diff = end_time - start_time
        
        if time_diff > 0:
            # Chuyển đổi pixels/second sang "km/h" (giả lập)
            # Trong thực tế, cần phép hiệu chỉnh phù hợp với góc nhìn camera
            pixels_per_meter = 20  # Giả sử 20 pixels = 1 meter
            self.speed = (distance / pixels_per_meter) * (3.6 / time_diff)
        
        # Thêm nhiễu ngẫu nhiên cho tính chân thực
        self.speed += random.uniform(-5, 5)
        self.speed = max(0, self.speed)  # Không cho phép tốc độ âm

class TrafficDetector:
    """Bộ phát hiện và theo dõi phương tiện giao thông"""
    
    def __init__(self, yolo_model='yolov5s'):
        """Khởi tạo bộ phát hiện
        
        Args:
            yolo_model: Tên mô hình YOLOv5 (mặc định: 'yolov5s')
        """
        # Tải mô hình YOLO
        try:
            self.model = YOLO(f"{yolo_model}.pt")
            print(f"Đã tải YOLOv5 ({yolo_model}) thành công.")
        except Exception as e:
            print(f"Lỗi khi tải mô hình YOLOv5: {str(e)}")
            raise
        
        # Ánh xạ chỉ số lớp sang tên
        self.traffic_classes = {
            2: "Ô tô",
            3: "Xe máy",
            5: "Xe buýt",
            7: "Xe tải",
            1: "Xe đạp"
        }
        
        # Ánh xạ tên lớp sang chỉ số
        self.class_name_to_id = {v: k for k, v in self.traffic_classes.items()}
        
        # Đối tượng theo dõi
        self.tracked_objects = {}
        self.next_obj_id = 0
        
        # Thống kê giao thông
        self.traffic_stats = {
            "by_class": defaultdict(int),
            "total_count": 0,
            "avg_speed": 0,
            "violations": 0
        }
        
        # Bộ đếm khung hình
        self.frame_count = 0
    
    def detect(self, frame, conf_threshold=0.35):
        """Phát hiện phương tiện trong khung hình
        
        Args:
            frame: Khung hình cần phát hiện
            conf_threshold: Ngưỡng độ tin cậy (mặc định: 0.35)
        
        Returns:
            Tuple (detections, annotated_frame)
        """
        # Tăng bộ đếm
        self.frame_count += 1
        
        # Thực hiện phát hiện
        results = self.model(frame, verbose=False)
        
        # Lấy khung hình có chú thích
        annotated_frame = results[0].plot() if self.frame_count % 2 == 0 else frame.copy()
        
        # Các lớp phương tiện cần theo dõi
        traffic_class_ids = list(self.traffic_classes.keys())
        
        # Danh sách đối tượng đã phát hiện trong khung hình này
        current_objects = {}
        
        detections = []
        
        # Xử lý từng phát hiện
        for detection in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, class_id = detection
            
            # Kiểm tra lớp và độ tin cậy
            if int(class_id) in traffic_class_ids and conf >= conf_threshold:
                # Thêm vào danh sách phát hiện
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': int(class_id),
                    'class_name': self.traffic_classes.get(int(class_id), "Unknown"),
                    'confidence': conf
                })
        
        # Cập nhật đối tượng theo dõi
        self._update_tracked_objects(detections)
        
        # Vẽ thông tin nếu có đối tượng được theo dõi
        if self.tracked_objects:
            self._draw_tracking_info(annotated_frame)
        
        return detections, annotated_frame
    
    def _update_tracked_objects(self, detections):
        """Cập nhật đối tượng theo dõi dựa trên phát hiện mới"""
        # Đánh dấu tất cả đối tượng hiện tại là chưa được cập nhật
        updated_ids = set()
        
        for detection in detections:
            bbox = detection['bbox']
            class_id = detection['class_id']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Tìm đối tượng phù hợp nhất để cập nhật
            matched_id = self._find_matching_object(bbox)
            
            if matched_id is not None:
                # Cập nhật đối tượng đã tồn tại
                self.tracked_objects[matched_id].update(bbox, confidence)
                updated_ids.add(matched_id)
            else:
                # Tạo đối tượng theo dõi mới
                new_obj = TrackedObject(
                    self.next_obj_id, class_id, class_name, bbox, confidence)
                self.tracked_objects[self.next_obj_id] = new_obj
                updated_ids.add(self.next_obj_id)
                self.next_obj_id += 1
                
                # Cập nhật thống kê
                self.traffic_stats["by_class"][class_name] += 1
                self.traffic_stats["total_count"] += 1
        
        # Xóa đối tượng cũ không còn cập nhật
        current_ids = list(self.tracked_objects.keys())
        for obj_id in current_ids:
            if obj_id not in updated_ids:
                del self.tracked_objects[obj_id]
        
        # Cập nhật tốc độ trung bình
        speeds = [obj.speed for obj in self.tracked_objects.values() if obj.speed > 0]
        if speeds:
            self.traffic_stats["avg_speed"] = sum(speeds) / len(speeds)
    
    def _find_matching_object(self, new_bbox, iou_threshold=0.3):
        """Tìm đối tượng hiện có phù hợp nhất với bbox mới"""
        best_match_id = None
        best_iou = iou_threshold
        
        new_center = ((new_bbox[0] + new_bbox[2]) / 2, (new_bbox[1] + new_bbox[3]) / 2)
        
        for obj_id, obj in self.tracked_objects.items():
            current_bbox = obj.bbox
            current_center = obj.bbox_center()
            
            # Tính IoU (Intersection over Union)
            iou = self._calculate_iou(new_bbox, current_bbox)
            
            # Tính khoảng cách giữa các tâm
            center_distance = np.sqrt((new_center[0] - current_center[0])**2 + 
                                     (new_center[1] - current_center[1])**2)
            
            # Ưu tiên cho cả IoU cao và khoảng cách tâm nhỏ
            if iou > best_iou or (iou > 0.1 and center_distance < 50):
                best_match_id = obj_id
                best_iou = iou
        
        return best_match_id
    
    def _calculate_iou(self, bbox1, bbox2):
        """Tính toán IoU giữa hai bbox"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Tính diện tích giao
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Tính diện tích từng bbox
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Tính IoU
        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        return iou
    
    def _draw_tracking_info(self, frame):
        """Vẽ thông tin theo dõi lên khung hình"""
        for obj_id, obj in self.tracked_objects.items():
            x1, y1, x2, y2 = [int(c) for c in obj.bbox]
            
            # Vẽ đường viền bao quanh đối tượng
            cv2.rectangle(frame, (x1, y1), (x2, y2), obj.color, 2)
            
            # Hiển thị ID, lớp và tốc độ - Sử dụng font tốt hơn để tránh lỗi hiển thị tiếng Việt
            text = f"ID:{obj_id} {obj.class_name} {obj.speed:.1f}km/h"
            
            # Sử dụng font mặc định của OpenCV (FONT_HERSHEY_SIMPLEX) với kích thước lớn hơn
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.7
            thickness = 2
            
            # Tạo background cho text để dễ đọc hơn
            (text_width, text_height), baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
            cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), obj.color, -1)
            
            # Vẽ text với màu trắng trên nền màu
            cv2.putText(frame, text, (x1, y1 - 5), fontFace, fontScale, (255, 255, 255), thickness)
            
            # Vẽ vết (trajectory)
            points = [pos[1] for pos in obj.positions]
            for i in range(1, len(points)):
                cv2.line(frame, 
                       (int(points[i-1][0]), int(points[i-1][1])),
                       (int(points[i][0]), int(points[i][1])),
                       obj.color, 2)
    
    def extract_vehicles(self, frame, detections):
        """Trích xuất hình ảnh phương tiện từ khung hình
        
        Args:
            frame: Khung hình gốc
            detections: Danh sách các phát hiện
        
        Returns:
            Danh sách các tuple (image, class_id, confidence)
        """
        vehicles = []
        
        for det in detections:
            bbox = det['bbox']
            class_id = det['class_id']
            confidence = det['confidence']
            
            x1, y1, x2, y2 = [int(c) for c in bbox]
            
            # Tránh chỉ số ngoài phạm vi
            y1 = max(0, y1)
            x1 = max(0, x1)
            y2 = min(frame.shape[0], y2)
            x2 = min(frame.shape[1], x2)
            
            # Trích xuất ROI
            vehicle_img = frame[y1:y2, x1:x2]
            
            # Thêm vào danh sách
            vehicles.append((vehicle_img, class_id, confidence))
        
        return vehicles
    
    def draw_traffic_stats(self, frame):
        """Vẽ thống kê giao thông lên khung hình
        
        Args:
            frame: Khung hình gốc
        
        Returns:
            Khung hình với thống kê
        """
        # Vẽ nền
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Thiết lập font và màu sắc
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7
        thickness = 2
        textColor = (255, 255, 255)
        
        # Hiển thị thống kê
        cv2.putText(frame, f"Tổng phương tiện: {self.traffic_stats['total_count']}", 
                   (20, 40), fontFace, fontScale, textColor, thickness)
        
        cv2.putText(frame, f"Tốc độ TB: {self.traffic_stats['avg_speed']:.1f} km/h", 
                   (20, 70), fontFace, fontScale, textColor, thickness)
        
        # Hiển thị phương tiện hiện tại
        current_count = len(self.tracked_objects)
        cv2.putText(frame, f"Phương tiện hiện tại: {current_count}", 
                   (20, 100), fontFace, fontScale, textColor, thickness)
        
        # Hiển thị vi phạm
        cv2.putText(frame, f"Vi phạm: {self.traffic_stats['violations']}", 
                   (20, 130), fontFace, fontScale, textColor, thickness)
        
        return frame
    
    def get_traffic_stats(self):
        """Lấy thống kê giao thông hiện tại
        
        Returns:
            Dictionary chứa thống kê
        """
        return self.traffic_stats

class TrafficViolationDetector:
    """Bộ phát hiện vi phạm giao thông"""
    
    def __init__(self, speed_limit=50):
        """Khởi tạo bộ phát hiện vi phạm
        
        Args:
            speed_limit: Giới hạn tốc độ (km/h)
        """
        self.speed_limit = speed_limit
        self.violations = []
    
    def set_speed_limit(self, speed_limit):
        """Đặt giới hạn tốc độ mới
        
        Args:
            speed_limit: Giới hạn tốc độ mới (km/h)
        """
        self.speed_limit = speed_limit
    
    def set_red_light_zone(self, polygon):
        """
        Phương thức giả để tương thích với code cũ.
        Không còn sử dụng nhưng giữ lại để không phải sửa code khác.
        """
        pass
    
    def detect_violations(self, tracked_objects, frame):
        """Phát hiện vi phạm trong danh sách đối tượng theo dõi
        
        Args:
            tracked_objects: Từ điển các đối tượng theo dõi
            frame: Khung hình hiện tại
        
        Returns:
            Khung hình có chú thích vi phạm
        """
        if not tracked_objects:
            return frame
        
        # Kiểm tra vi phạm cho từng đối tượng
        for obj_id, obj in tracked_objects.items():
            # Phát hiện vi phạm tốc độ
            if obj.speed > self.speed_limit:
                # Kiểm tra xem đã có vi phạm chưa
                speed_violation = any(v.violation_type == "Vượt tốc độ" for v in obj.violations)
                
                if not speed_violation:
                    violation = TrafficViolation(
                        obj_id, obj.class_name, "Vượt tốc độ", 
                        f"{obj.speed:.1f} km/h (Giới hạn: {self.speed_limit} km/h)")
                    obj.violations.append(violation)
                    self.violations.append(violation)
                
                # Hiển thị cảnh báo trên đối tượng
                self._draw_speed_violation(frame, obj)
        
        return frame
    
    def _draw_speed_violation(self, frame, obj):
        """Vẽ cảnh báo vi phạm tốc độ"""
        x1, y1, x2, y2 = [int(c) for c in obj.bbox]
        
        # Vẽ đường viền cảnh báo
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Thiết lập font và màu sắc
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        thickness = 2
        
        # Tạo text cảnh báo vi phạm
        warning_text = f"VƯỢT TỐC ĐỘ: {obj.speed:.1f} km/h"
        
        # Tạo background cho text để dễ đọc hơn
        (text_width, text_height), baseline = cv2.getTextSize(warning_text, fontFace, fontScale, thickness)
        cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 0, 255), -1)
        
        # Hiển thị cảnh báo
        cv2.putText(frame, warning_text, 
                   (x1, y1 - 5), fontFace, fontScale, (255, 255, 255), thickness)
    
    def get_violations(self):
        """Lấy danh sách vi phạm
        
        Returns:
            Danh sách các vi phạm
        """
        return self.violations 