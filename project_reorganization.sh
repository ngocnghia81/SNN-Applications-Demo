#!/bin/bash

# Script tổ chức lại thư mục dự án SNN
echo "Bắt đầu tổ chức lại cấu trúc thư mục dự án SNN..."

# 1. Tạo cấu trúc thư mục mới
mkdir -p src/snn_traffic/{app,models,detectors,utils,visualization}
mkdir -p src/snn_examples/{bindsnet,nengo,pysnn,spikingjelly}
mkdir -p data/{mnist,captured,processed,demo_videos}
mkdir -p docs/images
mkdir -p models/{traffic,mnist,backups}
mkdir -p tests
mkdir -p notebooks
mkdir -p configs
mkdir -p utils

# 2. Di chuyển các file của ứng dụng traffic monitoring vào thư mục tương ứng
echo "Di chuyển các file ứng dụng traffic monitoring..."
cp snn_traffic_app.py src/snn_traffic/app/
cp snn_traffic_model.py src/snn_traffic/models/
cp snn_traffic_detector.py src/snn_traffic/detectors/
cp traffic_flow_visualization.py src/snn_traffic/visualization/
cp download_demo_video.py src/snn_traffic/utils/

# 3. Di chuyển các file của các ví dụ SNN khác nhau
echo "Di chuyển các file ví dụ SNN..."
# Bindsnet files
cp bindsnet_*.py src/snn_examples/bindsnet/
cp bindset_test.py src/snn_examples/bindsnet/
cp -r bindsnet_plots_mnist/ src/snn_examples/bindsnet/plots

# Nengo files
cp nengo_*.py src/snn_examples/nengo/
cp -r nengo_plots_mnist/ src/snn_examples/nengo/plots

# PySNN files
cp pysnn_*.py src/snn_examples/pysnn/
cp -r pysnn_plots/ src/snn_examples/pysnn/plots
cp -r pysnn_plots_mnist/ src/snn_examples/pysnn/plots_mnist

# SpikingJelly files
cp spikingjelly_*.py src/snn_examples/spikingjelly/
cp -r snn_plots/ src/snn_examples/spikingjelly/plots
cp -r snn_plots_mnist/ src/snn_examples/spikingjelly/plots_mnist

# 4. Di chuyển mô hình vào thư mục models
echo "Di chuyển các mô hình đã huấn luyện..."
mkdir -p models/traffic
mkdir -p models/mnist
cp models/*.pt models/mnist/
cp yolov5su.pt models/traffic/

# 5. Di chuyển các thư mục khác vào vị trí mới
echo "Di chuyển dữ liệu đã xử lý..."
cp -r captured_frames/ data/captured/original
cp -r static_processed/ data/captured/processed
cp -r flow_visualization/ docs/images/flow
cp -r demo_videos/ data/demo_videos
cp -r images/ docs/images/misc

# 6. Tạo file __init__.py trong các module
echo "Tạo các file __init__.py cho cấu trúc module..."
touch src/snn_traffic/__init__.py
touch src/snn_traffic/app/__init__.py
touch src/snn_traffic/models/__init__.py
touch src/snn_traffic/detectors/__init__.py
touch src/snn_traffic/utils/__init__.py
touch src/snn_traffic/visualization/__init__.py
touch src/snn_examples/__init__.py
touch src/snn_examples/bindsnet/__init__.py
touch src/snn_examples/nengo/__init__.py
touch src/snn_examples/pysnn/__init__.py
touch src/snn_examples/spikingjelly/__init__.py

# 7. Cập nhật requirements.txt vào thư mục gốc
echo "Cập nhật requirements.txt..."
cp requirements.txt ./

# 8. Tạo file README.md mới trong thư mục gốc
echo "Cập nhật README.md..."
cp README.md ./

# 9. Tạo file main.py để chạy ứng dụng chính
echo "Tạo file chạy ứng dụng chính..."
cat > main.py << 'EOF'
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
EOF

# 10. Tạo file setup.py để cài đặt gói
echo "Tạo file setup.py..."
cat > setup.py << 'EOF'
#!/usr/bin/env python3
"""
Setup script cho dự án SNN Traffic Monitoring
"""

from setuptools import setup, find_packages

setup(
    name="snn_traffic",
    version="0.1.0",
    description="Ứng dụng giám sát giao thông sử dụng Spiking Neural Networks",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "numpy==1.23.5",
        "matplotlib>=3.3.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "ultralytics>=8.0.0",
        "tqdm>=4.62.0",
        "scikit-learn<1.3.0",
        "scipy>=1.7.0",
        "seaborn>=0.11.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "snn-traffic=main:main",
        ],
    },
)
EOF

# 11. Tạo file cấu hình mẫu
echo "Tạo file cấu hình mẫu..."
cat > configs/default_config.json << 'EOF'
{
    "model": {
        "snn_model_path": "models/mnist/SNNMLP_best.pt",
        "yolo_model_path": "models/traffic/yolov5su.pt",
        "use_gpu": true
    },
    "camera": {
        "width": 640,
        "height": 480,
        "fps": 30
    },
    "detection": {
        "confidence_threshold": 0.4,
        "speed_limit": 30
    },
    "display": {
        "show_stats": true,
        "show_tracking": true,
        "show_violations": true,
        "show_neuron_activity": true
    },
    "paths": {
        "captures_dir": "data/captured/original",
        "processed_dir": "data/captured/processed",
        "results_dir": "data/results"
    }
}
EOF

# 12. Tạo file run.sh để dễ dàng chạy ứng dụng
echo "Tạo script chạy ứng dụng..."
cat > run.sh << 'EOF'
#!/bin/bash

# Script chạy ứng dụng SNN Traffic Monitoring

# Kiểm tra nếu Python venv đã được kích hoạt
if [[ -z "$VIRTUAL_ENV" ]]; then
    if [[ -d "myenv-new" ]]; then
        source myenv-new/bin/activate
        echo "Đã kích hoạt môi trường ảo myenv-new"
    else
        echo "Cảnh báo: Không tìm thấy môi trường ảo. Sẽ sử dụng Python hệ thống."
    fi
fi

# Chạy ứng dụng với các tham số
python main.py "$@"
EOF
chmod +x run.sh

echo "Cấu trúc thư mục mới đã được tạo!"
echo "Sử dụng lệnh './run.sh' để chạy ứng dụng." 