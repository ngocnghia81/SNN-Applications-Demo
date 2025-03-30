import os
import sys
import shutil
import subprocess
import pkg_resources
from pathlib import Path

# Kiểm tra và cài đặt các thư viện cần thiết
required_packages = ['PyInstaller', 'numpy', 'matplotlib', 'torch', 'torchvision', 'spikingjelly']
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = [pkg for pkg in required_packages if pkg.lower() not in installed]

if missing:
    print(f"Đang cài đặt các thư viện cần thiết: {', '.join(missing)}")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)

# Đảm bảo thư mục output tồn tại
output_dir = Path("mnist_visualization/output")
output_dir.mkdir(parents=True, exist_ok=True)

# Chuẩn bị các file và thư mục cần thiết
def prepare_files():
    print("Đang chuẩn bị các file cho ứng dụng MNIST visualization...")
    
    # Tạo thư mục tạm thời cho việc đóng gói
    temp_dir = Path("mnist_app_temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Sao chép các file cần thiết
    shutil.copy("mnist_flow_visualization.py", temp_dir)
    
    # Tạo file main.py để chạy ứng dụng
    main_file = temp_dir / "main.py"
    with open(main_file, "w") as f:
        f.write("""
import os
import sys
import mnist_flow_visualization

# Đảm bảo thư mục output tồn tại
os.makedirs("output", exist_ok=True)

# Chạy tất cả các hàm tạo biểu đồ
print("Đang tạo trực quan hóa luồng SNN cho MNIST...")
mnist_flow_visualization.create_flowchart()
mnist_flow_visualization.create_mnist_preprocessing_visualization()
mnist_flow_visualization.create_spike_encoding_visualization()
mnist_flow_visualization.create_snn_architecture_visualization()
mnist_flow_visualization.create_lif_neuron_visualization()
mnist_flow_visualization.create_classification_visualization()
mnist_flow_visualization.create_training_accuracy_visualization()

print("\\nHoàn thành! Các hình ảnh đã được lưu trong thư mục 'output'.")
print("Bạn có thể tìm thấy các hình ảnh trực quan hóa trong thư mục này.")
input("\\nNhấn Enter để đóng ứng dụng...")
        """)
    
    return temp_dir

# Đóng gói ứng dụng sử dụng PyInstaller
def package_app(temp_dir):
    print("Đang đóng gói ứng dụng MNIST visualization...")
    
    # Chuẩn bị lệnh PyInstaller
    pyinstaller_cmd = [
        'pyinstaller',
        '--onefile',
        '--windowed',
        '--name', 'MNIST_SNN_Visualization',
        '--add-data', f'{temp_dir}/mnist_flow_visualization.py;.',
        f'{temp_dir}/main.py'
    ]
    
    # Thêm các thư mục dữ liệu nếu cần
    
    # Chạy PyInstaller
    subprocess.run(pyinstaller_cmd)
    
    # Di chuyển file thực thi đến thư mục đích
    dist_file = Path("dist/MNIST_SNN_Visualization.exe")
    if dist_file.exists():
        shutil.copy(dist_file, "MNIST_SNN_Visualization.exe")
        print(f"Đã tạo file thực thi: MNIST_SNN_Visualization.exe")
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
    spec_file = Path("MNIST_SNN_Visualization.spec")
    if spec_file.exists():
        spec_file.unlink()

# Hàm main
def main():
    print("=== Đóng gói ứng dụng MNIST SNN Visualization ===")
    
    try:
        temp_dir = prepare_files()
        package_app(temp_dir)
        cleanup(temp_dir)
        print("\nĐóng gói thành công! Bạn có thể chạy file MNIST_SNN_Visualization.exe")
    except Exception as e:
        print(f"Lỗi trong quá trình đóng gói: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 