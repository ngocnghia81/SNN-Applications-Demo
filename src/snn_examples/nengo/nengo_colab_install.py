"""
Nengo Installation Script for Google Colab
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


print("=== Thiết lập môi trường cho Nengo trên Google Colab ===\n")

# Gỡ cài đặt các gói hiện tại để tránh xung đột
print("Gỡ cài đặt các gói hiện tại để tránh xung đột...")
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "nengo", "nengo-dl", "nengo-extras"], 
                     stdout=subprocess.DEVNULL, 
                     stderr=subprocess.DEVNULL)

# Cài đặt các gói phụ thuộc
print("\n1. Cài đặt các gói phụ thuộc")
for dep in ["matplotlib", "tqdm"]:
    if not check_package_installed(dep):
        install_package(dep)
    else:
        print(f"{dep} đã được cài đặt ✓")

# Cài đặt TensorFlow với nhiều phiên bản khác nhau để tìm phiên bản phù hợp
print("\n2. Cài đặt TensorFlow (tùy chọn, cần thiết cho Nengo-DL)")
tf_installed = False

# Bỏ qua việc cài đặt TensorFlow, để Nengo hoạt động độc lập
print("Bỏ qua cài đặt TensorFlow, sẽ sử dụng Nengo cơ bản mà không cần Nengo-DL")
tf_installed = True  # Đánh dấu như đã cài đặt để tiếp tục

# Cài đặt Nengo với phiên bản cụ thể
print("\n3. Cài đặt Nengo cơ bản")
install_package("nengo==3.1.0")  # Sử dụng phiên bản 3.1.0

# Tạo monkey patch để sửa lỗi product trong NumPy
print("\n4. Tạo monkey patch cho NumPy")
patch_file = "monkey_patch.py"
with open(patch_file, "w") as f:
    f.write("""
# Monkey patch để thêm hàm product vào NumPy
import numpy as np
if not hasattr(np, 'product'):
    # Nếu NumPy không có hàm product, thêm một alias cho hàm prod
    np.product = np.prod
    print("Đã thêm monkey patch cho NumPy.product")

# Tạo file init để tự động áp dụng patch
with open("__init__.py", "w") as init_file:
    init_file.write("# Tự động import monkey patch khi import Nengo\\n")
    init_file.write("import sys, os\\n")
    init_file.write("import monkey_patch\\n")
""")

# Chạy monkey patch
print("Áp dụng monkey patch...")
exec(open(patch_file).read())

# Cài đặt Nengo-SPA
print("\n5. Cài đặt gói Nengo-SPA")
install_package("nengo-spa")

# Thêm hướng dẫn về monkey patch
print("\n=== QUAN TRỌNG ===")
print("Để sử dụng Nengo, vui lòng thêm dòng sau ở đầu file nengo_colab_main.py:")
print("import numpy as np")
print("if not hasattr(np, 'product'): np.product = np.prod")
print("\nHoặc chạy lại file này mỗi khi khởi động lại runtime.")

# Kiểm tra cài đặt
print("\n=== Kiểm tra cài đặt ===")
try:
    import numpy as np
    if not hasattr(np, 'product'):
        np.product = np.prod
    
    import nengo
    print(f"Nengo: v{nengo.__version__} ✓")
    
    # Kiểm tra nengo_dl
    try:
        import nengo_dl
        print(f"Nengo-DL: v{nengo_dl.__version__} ✓")
    except (ImportError, AttributeError):
        print("Nengo-DL: Không được cài đặt ✗")
        print("Sẽ sử dụng Nengo cơ bản mà không có Nengo-DL.")
        
except (ImportError, AttributeError) as e:
    print(f"Lỗi: {e}")
    print("Vui lòng khởi động lại runtime và chạy lại file này.")

# Install compatible versions
print("\nInstalling Nengo and dependencies...")
install_package("numpy==1.23.5")  # This version should have product function
install_package("nengo==3.1.0")   # Specify a stable version
install_package("nengo-dl==3.4.0") # Specify a stable version
install_package("tensorflow")

# Check if numpy has been installed
try:
    import numpy
    print(f"NumPy version: {numpy.__version__}")
    
    # Add monkey patch for product if it doesn't exist
    if not hasattr(numpy, 'product'):
        print("Adding monkey patch for numpy.product...")
        
        # Add product as an alias for prod
        numpy.product = numpy.prod
        
        print("Monkey patch applied. numpy.product is now available.")
    else:
        print("numpy.product is already available.")
        
    # Check if the patch worked
    print("Testing numpy.product function...")
    test_array = numpy.array([1, 2, 3, 4])
    result = numpy.product(test_array)
    print(f"Product of [1, 2, 3, 4] = {result}")
    
    # Import nengo to verify installation
    print("Importing Nengo to verify installation...")
    import nengo
    print(f"Nengo version: {nengo.__version__}")
    
    import nengo_dl
    print(f"Nengo-DL version: {nengo_dl.__version__}")
    
    print("Installation successful!")
    
except ImportError as e:
    print(f"Error importing packages: {e}")
    print("Please try running this script again, or restart your runtime.") 