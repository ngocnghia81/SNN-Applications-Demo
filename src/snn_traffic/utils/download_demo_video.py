#!/usr/bin/env python3
"""
Tải video demo giao thông để kiểm tra ứng dụng phân tích giao thông SNN
"""

import os
import sys
import requests
from tqdm import tqdm

# URL các video demo
DEMO_VIDEOS = {
    "highway": "https://github.com/MaheshGowda1/Traffic_flow_monitoring/raw/master/Highway.mp4",
    "intersection": "https://github.com/MaheshGowda1/Traffic_flow_monitoring/raw/master/Intersection.mp4",
    "traffic_cam": "https://github.com/intel-iot-devkit/sample-videos/raw/master/traffic.mp4",
    "city_traffic": "https://github.com/MaheshGowda1/Traffic_flow_monitoring/raw/master/City.mp4"
}

def download_file(url, output_dir, filename):
    """
    Tải file từ URL với thanh tiến trình
    
    Args:
        url: URL của file cần tải
        output_dir: Thư mục đích
        filename: Tên file
    
    Returns:
        Path tới file đã tải
    """
    output_path = os.path.join(output_dir, filename)
    
    # Kiểm tra file đã tồn tại chưa
    if os.path.exists(output_path):
        print(f"File {filename} đã tồn tại. Bỏ qua tải xuống.")
        return output_path
    
    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tải file với thanh tiến trình
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    
    return output_path

def main():
    """Hàm chính"""
    # Tạo thư mục đích
    output_dir = "demo_videos"
    
    print("Tải video demo giao thông:")
    for name, url in DEMO_VIDEOS.items():
        filename = f"{name}.mp4"
        try:
            path = download_file(url, output_dir, filename)
            print(f"Đã tải: {path}")
        except Exception as e:
            print(f"Lỗi khi tải {name}: {str(e)}")
    
    # Hướng dẫn sử dụng
    print("\nĐể chạy ứng dụng với video demo, sử dụng lệnh:")
    for name in DEMO_VIDEOS.keys():
        print(f"python snn_traffic_app.py --video demo_videos/{name}.mp4")

if __name__ == "__main__":
    main() 