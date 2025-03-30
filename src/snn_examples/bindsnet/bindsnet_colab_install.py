"""
Cài đặt thư viện cần thiết cho BindsNET trên Google Colab
"""

import os
import subprocess
import sys
import platform
import shutil

# Hàm cài đặt gói
def install_package(package_name):
    print(f"Đang cài đặt {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--ignore-installed", package_name])
        print(f"Đã cài đặt xong {package_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"Không thể cài đặt {package_name}")
        return False

print("Cài đặt các phiên bản tương thích cho BindsNET...")
print(f"Python version: {platform.python_version()}")

# Thử cài đặt PyTorch phiên bản mới hơn (phù hợp với Python 3.11)
try:
    print("Cài đặt PyTorch và Torchvision...")
    # Sử dụng phiên bản mới nhất tương thích với Python 3.11
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--ignore-installed", "torch", "torchvision"])
    print("Đã cài đặt PyTorch và Torchvision")
except subprocess.CalledProcessError:
    print("Không thể cài đặt PyTorch và Torchvision tự động")
    print("Thử sử dụng cài đặt thủ công:")
    print("!pip install torch torchvision")

# Cài đặt các thư viện hỗ trợ
packages = ["tqdm", "matplotlib"]
for package in packages:
    try:
        __import__(package)
        print(f"{package} đã được cài đặt sẵn")
    except ImportError:
        install_package(package)

# Kiểm tra và cài đặt BindsNET
try:
    # Thử import BindsNET
    import bindsnet
    print("BindsNET đã được cài đặt sẵn")
except ImportError:
    print("Tiến hành cài đặt BindsNET...")
    
    # Kiểm tra torch
    try:
        import torch
        print(f"Đã cài đặt PyTorch phiên bản: {torch.__version__}")
        
        # Kiểm tra xem module torch._six có tồn tại không
        has_torch_six = False
        try:
            from torch._six import string_classes
            has_torch_six = True
            print("PyTorch có module torch._six (tương thích với BindsNET)")
        except ImportError:
            print("PyTorch không có module torch._six (cần sửa đổi BindsNET)")
        
        # Phương pháp 1: Cài đặt trực tiếp từ GitHub
        try:
            print("Thử cài đặt BindsNET từ GitHub...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--ignore-installed",
                "git+https://github.com/BindsNET/bindsnet.git@master"
            ])
            print("Đã cài đặt BindsNET từ GitHub")
        except subprocess.CalledProcessError:
            print("Không thể cài đặt BindsNET trực tiếp từ GitHub")
            
            # Phương pháp 2: Tải và sửa đổi mã nguồn
            try:
                print("Thử tải và sửa đổi mã nguồn BindsNET...")
                
                # Dọn dẹp thư mục cũ nếu có
                if os.path.exists("bindsnet"):
                    shutil.rmtree("bindsnet")
                
                # Kiểm tra xem git có sẵn không
                try:
                    subprocess.check_call(["git", "--version"])
                    has_git = True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    has_git = False
                
                if has_git:
                    # Tải về bằng git
                    subprocess.check_call(["git", "clone", "https://github.com/BindsNET/bindsnet.git"])
                else:
                    # Tải về bằng wget nếu không có git
                    print("Git không khả dụng, thử tải bằng wget...")
                    subprocess.check_call([
                        "wget", "https://github.com/BindsNET/bindsnet/archive/refs/heads/master.zip"
                    ])
                    subprocess.check_call(["unzip", "master.zip"])
                    os.rename("bindsnet-master", "bindsnet")
                
                # Di chuyển vào thư mục bindsnet
                os.chdir("bindsnet")
                
                # Sửa mã nguồn nếu cần
                if not has_torch_six:
                    print("Sửa đổi mã nguồn BindsNET để tương thích với PyTorch mới...")
                    
                    # Sửa file collate.py
                    collate_path = "bindsnet/datasets/collate.py"
                    if os.path.exists(collate_path):
                        print(f"Sửa file {collate_path}...")
                        with open(collate_path, "r") as f:
                            content = f.read()
                        
                        # Thay thế import từ torch._six
                        content = content.replace(
                            "from torch._six import container_abcs, string_classes, int_classes", 
                            "import collections.abc as container_abcs\nstring_classes = str\nint_classes = int"
                        )
                        
                        with open(collate_path, "w") as f:
                            f.write(content)
                        
                        print("Đã sửa file collate.py")
                
                # Cài đặt từ mã nguồn
                print("Cài đặt từ mã nguồn...")
                subprocess.check_call([sys.executable, "setup.py", "install"])
                
                # Trở lại thư mục gốc
                os.chdir("..")
                print("Đã cài đặt BindsNET từ mã nguồn đã sửa đổi")
                
            except Exception as e:
                print(f"Lỗi khi tải và sửa đổi BindsNET: {e}")
                print("\nPhương pháp thủ công:")
                print("1. Tải BindsNET: !git clone https://github.com/BindsNET/bindsnet.git")
                print("2. Sửa file bindsnet/datasets/collate.py")
                print('3. Thay dòng "from torch._six import container_abcs, string_classes, int_classes" bằng:')
                print('   "import collections.abc as container_abcs"')
                print('   "string_classes = str"')
                print('   "int_classes = int"')
                print("4. Cài đặt: !cd bindsnet && python setup.py install")
    
    except ImportError as e:
        print(f"Lỗi khi kiểm tra PyTorch: {e}")

# Kiểm tra cài đặt
try:
    import torch
    import torchvision
    import bindsnet
    print("\nKiểm tra phiên bản cuối cùng:")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"BindsNET đã được cài đặt thành công")
except ImportError as e:
    print(f"Lỗi: {e}")
    print("Một số gói không được cài đặt thành công")
    print("Vui lòng làm theo hướng dẫn thủ công để hoàn tất cài đặt")

print("\nĐã hoàn thành quá trình cài đặt.")
print("Bạn có thể chạy file 'bindsnet_colab_main.py' để huấn luyện mạng SNN") 