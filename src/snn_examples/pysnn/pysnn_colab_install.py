"""
PySNN Installation Script for Google Colab
"""

import sys
import subprocess
import importlib.util
import os


def install_package(package_name):
    """
    Install a package using pip and provide feedback
    """
    print(f"Đang cài đặt {package_name}...", end="")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
        print(" ✓")
        return True
    except Exception as e:
        print(f" ✗ (Lỗi: {e})")
        return False


def check_package_installed(package_name):
    """
    Check if a package is installed
    """
    return importlib.util.find_spec(package_name) is not None


print("=== Thiết lập môi trường cho PySNN trên Google Colab ===\n")

# Kiểm tra phiên bản Python
python_version = sys.version.split()[0]
print(f"Python version: {python_version}")

# Cài đặt PyTorch và Torchvision nếu chưa có
if not (check_package_installed("torch") and check_package_installed("torchvision")):
    print("\n1. Cài đặt PyTorch và Torchvision")
    try:
        # Cài đặt PyTorch với CUDA 11.8
        install_command = "torch torchvision"
        install_package(install_command)
    except Exception as e:
        print(f"Lỗi khi cài đặt PyTorch: {e}")
        print("Hãy thử cài đặt thủ công:")
        print("!pip install torch torchvision")
else:
    import torch
    print(f"\n1. PyTorch đã được cài đặt (phiên bản {torch.__version__})")

# Cài đặt các gói phụ thuộc khác
print("\n2. Cài đặt các gói phụ thuộc")
dependencies = ["tqdm", "matplotlib", "numpy"]

for dep in dependencies:
    if not check_package_installed(dep):
        install_package(dep)
    else:
        print(f"{dep} đã được cài đặt ✓")

# Kiểm tra và cài đặt PySNN
print("\n3. Kiểm tra và cài đặt PySNN")
if not check_package_installed("pysnn"):
    print("PySNN chưa được cài đặt, đang clone từ GitHub...")
    
    # Tạo thư mục temp nếu chưa tồn tại
    if not os.path.exists("./temp"):
        os.makedirs("./temp")
    
    # Clone repository
    try:
        subprocess.check_call(["git", "clone", "https://github.com/BasBuller/PySNN.git", "./temp/PySNN"], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
        print("Đã clone PySNN repository thành công.")
        
        # Cài đặt PySNN từ source
        os.chdir("./temp/PySNN")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
        os.chdir("../..")
        print("Đã cài đặt PySNN từ source thành công ✓")
    except Exception as e:
        print(f"Lỗi khi cài đặt PySNN: {e}")
        print("Hãy thử cài đặt thủ công:")
        print("!git clone https://github.com/BasBuller/PySNN.git")
        print("!cd PySNN && pip install -e .")
else:
    print("PySNN đã được cài đặt ✓")

# Kiểm tra xem tất cả các gói đã được cài đặt thành công chưa
print("\n=== Kiểm tra các gói đã cài đặt ===")
all_packages_installed = True

check_packages = ["torch", "torchvision", "tqdm", "matplotlib", "numpy", "pysnn"]
for package in check_packages:
    if check_package_installed(package):
        # Hiển thị phiên bản nếu có thể
        try:
            if package == "pysnn":
                print(f"{package}: Đã cài đặt ✓")
            else:
                module = __import__(package)
                print(f"{package}: v{module.__version__} ✓")
        except (ImportError, AttributeError):
            print(f"{package}: Đã cài đặt ✓")
    else:
        print(f"{package}: Chưa cài đặt ✗")
        all_packages_installed = False

if all_packages_installed:
    print("\n✅ Tất cả các gói đã được cài đặt thành công!")
    print("Bạn có thể tiếp tục chạy file pysnn_colab_main.py để huấn luyện mạng SNN.")
else:
    print("\n❌ Một số gói chưa được cài đặt đúng cách.")
    print("Vui lòng cài đặt thủ công các gói còn thiếu trước khi tiếp tục.") 