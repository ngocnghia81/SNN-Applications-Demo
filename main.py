#!/usr/bin/env python3
"""
SNN Traffic Monitoring - Ứng dụng chính
Khởi chạy ứng dụng giám sát giao thông sử dụng SNN
"""

import sys
import os

# Thêm thư mục src vào đường dẫn Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from snn_traffic.app.snn_traffic_app import TrafficMonitorApp
import tkinter as tk
import argparse

def main():
    """Hàm chính để chạy ứng dụng"""
    # Xử lý tham số dòng lệnh
    parser = argparse.ArgumentParser(description='SNN Traffic Monitoring App')
    parser.add_argument('--video', type=str, help='Đường dẫn đến file video')
    parser.add_argument('--camera', type=int, default=0, help='Chỉ số camera (mặc định: 0)')
    parser.add_argument('--speed-limit', type=int, default=30, help='Giới hạn tốc độ (km/h, mặc định: 30)')
    
    args = parser.parse_args()
    
    # Khởi tạo ứng dụng Tkinter
    root = tk.Tk()
    app = TrafficMonitorApp(root, video_source=args.video)
    
    # Thiết lập giới hạn tốc độ
    if hasattr(app, 'speed_limit_var'):
        app.speed_limit_var.set(args.speed_limit)
    
    # Chạy ứng dụng
    root.mainloop()

if __name__ == "__main__":
    main()
