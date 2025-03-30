import os
import sys
import shutil
import subprocess
import pkg_resources
from pathlib import Path

# Kiểm tra và cài đặt các thư viện cần thiết
required_packages = ['PyInstaller', 'numpy', 'torch', 'torchvision', 'opencv-python', 'ultralytics', 'matplotlib']
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = [pkg for pkg in required_packages if pkg.lower() not in installed]

if missing:
    print(f"Đang cài đặt các thư viện cần thiết: {', '.join(missing)}")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)

# Chuẩn bị các file và thư mục cần thiết
def prepare_files():
    print("Đang chuẩn bị các file cho ứng dụng Traffic Monitoring...")
    
    # Tạo thư mục tạm thời cho việc đóng gói
    temp_dir = Path("traffic_app_temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Tạo thư mục cho dữ liệu đầu ra
    for folder in ["captures", "detection_results", "demo_videos"]:
        (temp_dir / folder).mkdir(exist_ok=True)
    
    # Sao chép các file cần thiết
    for file in ["snn_traffic_app.py", "snn_traffic_model.py", "snn_traffic_detector.py", "download_demo_video.py"]:
        if Path(file).exists():
            shutil.copy(file, temp_dir)
    
    # Sao chép mô hình YOLOv5 nếu có
    yolo_model = Path("yolov5su.pt")
    if yolo_model.exists():
        shutil.copy(yolo_model, temp_dir)
    
    # Tạo file main.py để chạy ứng dụng
    main_file = temp_dir / "main.py"
    with open(main_file, "w") as f:
        f.write("""
import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

def check_demo_videos():
    # Kiểm tra nếu thư mục demo_videos trống
    video_dir = Path("demo_videos")
    video_dir.mkdir(exist_ok=True)
    
    videos = list(video_dir.glob("*.mp4"))
    if not videos:
        return False
    return True

def download_videos():
    try:
        import download_demo_video
        download_demo_video.main()
        return True
    except Exception as e:
        print(f"Lỗi khi tải video demo: {e}")
        return False

def run_app():
    try:
        import snn_traffic_app
        app = snn_traffic_app.TrafficMonitorApp()
        app.run()
    except Exception as e:
        print(f"Lỗi khi chạy ứng dụng: {e}")

def main():
    root = tk.Tk()
    root.withdraw()
    
    # Đảm bảo các thư mục tồn tại
    for folder in ["captures", "detection_results", "demo_videos"]:
        os.makedirs(folder, exist_ok=True)
    
    # Kiểm tra videos demo
    has_videos = check_demo_videos()
    
    if not has_videos:
        result = messagebox.askyesno(
            "Tải video demo",
            "Không tìm thấy video demo. Bạn có muốn tải các video demo từ internet không?\\n\\n" + 
            "(Nếu chọn No, bạn sẽ cần phải sử dụng webcam hoặc tự thêm video vào thư mục demo_videos)"
        )
        
        if result:
            success = download_videos()
            if not success:
                messagebox.showerror(
                    "Lỗi tải video", 
                    "Không thể tải video demo. Vui lòng kiểm tra kết nối internet hoặc thêm video thủ công."
                )
    
    # Chạy ứng dụng chính
    run_app()

if __name__ == "__main__":
    main()
        """)
    
    return temp_dir

# Đóng gói ứng dụng sử dụng PyInstaller
def package_app(temp_dir):
    print("Đang đóng gói ứng dụng Traffic Monitoring...")
    
    # Chuẩn bị lệnh PyInstaller
    pyinstaller_cmd = [
        'pyinstaller',
        '--onefile',
        '--windowed',
        '--name', 'SNN_Traffic_Monitor',
        '--add-data', f'{temp_dir}/snn_traffic_app.py;.',
        '--add-data', f'{temp_dir}/snn_traffic_model.py;.',
        '--add-data', f'{temp_dir}/snn_traffic_detector.py;.',
        '--add-data', f'{temp_dir}/download_demo_video.py;.',
    ]
    
    # Thêm model file nếu tồn tại
    yolo_model = temp_dir / "yolov5su.pt"
    if yolo_model.exists():
        pyinstaller_cmd.extend(['--add-data', f'{yolo_model};.'])
    
    # Thêm file main.py
    pyinstaller_cmd.append(f'{temp_dir}/main.py')
    
    # Chạy PyInstaller
    subprocess.run(pyinstaller_cmd)
    
    # Di chuyển file thực thi và chuẩn bị các thư mục cần thiết
    dist_file = Path("dist/SNN_Traffic_Monitor.exe")
    if dist_file.exists():
        shutil.copy(dist_file, "SNN_Traffic_Monitor.exe")
        print(f"Đã tạo file thực thi: SNN_Traffic_Monitor.exe")
        
        # Tạo các thư mục cần thiết
        for folder in ["captures", "detection_results", "demo_videos"]:
            Path(folder).mkdir(exist_ok=True)
    else:
        print("Không thể tìm thấy file thực thi. Quá trình đóng gói có thể đã thất bại.")

# Dọn dẹp các file tạm thời
def cleanup(temp_dir):
    print("Đang dọn dẹp các file tạm thời...")
    
    # Xóa các thư mục tạm thời
    for path in [temp_dir, Path("build"), Path("dist")]:
        if path.exists():
            shutil.rmtree(path)
    
    # Xóa file spec
    spec_file = Path("SNN_Traffic_Monitor.spec")
    if spec_file.exists():
        spec_file.unlink()

# Tạo file README để hướng dẫn sử dụng
def create_readme():
    readme_content = """# SNN Traffic Monitor - Ứng dụng theo dõi giao thông sử dụng SNN

## Cách sử dụng

1. Chạy file `SNN_Traffic_Monitor.exe` để khởi động ứng dụng
2. Nếu chưa có video demo, ứng dụng sẽ hỏi bạn có muốn tải xuống từ internet hay không
3. Sau khi khởi động, có thể lựa chọn nguồn video (webcam hoặc video demo)
4. Sử dụng các nút điều khiển để bắt đầu/dừng phân tích giao thông
5. Xem kết quả phân tích và biểu đồ hoạt động neuron trong thời gian thực

## Yêu cầu hệ thống

- Windows 10/11 64-bit
- RAM tối thiểu 4GB (khuyến nghị 8GB)
- Card đồ họa hỗ trợ GPU (nếu có) sẽ tăng tốc độ xử lý
- Webcam (nếu muốn sử dụng camera trực tiếp)

## Giải quyết sự cố

Nếu ứng dụng không khởi động được:
1. Đảm bảo đã cài đặt Microsoft Visual C++ Redistributable
2. Kiểm tra quyền truy cập thư mục
3. Thử chạy với quyền admin

## Liên hệ hỗ trợ

Nếu gặp vấn đề, vui lòng liên hệ qua email: [your-email@example.com]
"""
    
    with open("SNN_Traffic_Monitor_README.txt", "w") as f:
        f.write(readme_content)
    
    print("Đã tạo file README hướng dẫn sử dụng.")

# Hàm main
def main():
    print("=== Đóng gói ứng dụng SNN Traffic Monitor ===")
    
    try:
        temp_dir = prepare_files()
        package_app(temp_dir)
        cleanup(temp_dir)
        create_readme()
        print("\nĐóng gói thành công! Bạn có thể chạy file SNN_Traffic_Monitor.exe")
        print("Vui lòng đọc file SNN_Traffic_Monitor_README.txt để biết hướng dẫn sử dụng.")
    except Exception as e:
        print(f"Lỗi trong quá trình đóng gói: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 