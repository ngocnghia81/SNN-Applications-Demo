# Hướng dẫn đóng gói ứng dụng thành file EXE

Tài liệu này hướng dẫn đóng gói 2 ứng dụng SNN demo thành file EXE để có thể chạy trên các máy tính Windows mà không cần cài đặt Python.

## 1. Chuẩn bị môi trường

Trước khi bắt đầu, hãy đảm bảo bạn đã cài đặt các thư viện cần thiết:

```bash
pip install pyinstaller numpy matplotlib torch torchvision opencv-python ultralytics spikingjelly
```

## 2. Đóng gói ứng dụng MNIST SNN Visualization

### Bước 1: Chạy script đóng gói

Chạy script `package_mnist_app.py` để tạo file EXE:

```bash
python package_mnist_app.py
```

Script này sẽ tự động thực hiện các bước sau:

-   Cài đặt các thư viện cần thiết nếu chưa có
-   Chuẩn bị các file cần thiết
-   Đóng gói ứng dụng với PyInstaller
-   Tạo file thực thi `MNIST_SNN_Visualization.exe`

### Bước 2: Chuẩn bị thư mục đầu ra

Sau khi chạy script thành công, bạn sẽ thấy file `MNIST_SNN_Visualization.exe` trong thư mục làm việc.

### Bước 3: Sử dụng ứng dụng

-   Chạy file `MNIST_SNN_Visualization.exe`
-   Ứng dụng sẽ tạo ra 7 hình ảnh trực quan hóa trong thư mục `output`
-   Sau khi hoàn thành, nhấn Enter để đóng ứng dụng

## 3. Đóng gói ứng dụng Traffic Monitoring

### Bước 1: Chạy script đóng gói

Chạy script `package_traffic_app.py` để tạo file EXE:

```bash
python package_traffic_app.py
```

Script này sẽ tự động thực hiện các bước sau:

-   Cài đặt các thư viện cần thiết nếu chưa có
-   Chuẩn bị các file cần thiết bao gồm mô hình YOLOv5
-   Đóng gói ứng dụng với PyInstaller
-   Tạo file thực thi `SNN_Traffic_Monitor.exe` và các thư mục cần thiết

### Bước 2: Chuẩn bị phân phối

Sau khi chạy script thành công, bạn sẽ thấy:

-   File `SNN_Traffic_Monitor.exe`
-   File hướng dẫn `SNN_Traffic_Monitor_README.txt`
-   Các thư mục `captures`, `detection_results`, và `demo_videos`

### Bước 3: Sử dụng ứng dụng

-   Chạy file `SNN_Traffic_Monitor.exe`
-   Nếu chưa có video demo, ứng dụng sẽ hỏi bạn có muốn tải xuống không
-   Sau đó ứng dụng sẽ hiển thị giao diện giám sát giao thông

## 4. Phân phối cho người dùng

Để phân phối ứng dụng cho người dùng, bạn cần:

### Cho ứng dụng MNIST:

-   File `MNIST_SNN_Visualization.exe`
-   Tạo thư mục `output` trống

### Cho ứng dụng Traffic:

-   File `SNN_Traffic_Monitor.exe`
-   File hướng dẫn `SNN_Traffic_Monitor_README.txt`
-   Các thư mục `captures`, `detection_results`, và `demo_videos` (có thể để trống)
-   File mô hình `yolov5su.pt` (nếu không đóng gói cùng EXE)

## 5. Giải quyết sự cố

### Vấn đề thường gặp

1. **Thiếu thư viện**: Đảm bảo tất cả các thư viện phụ thuộc đã được cài đặt
2. **File EXE không chạy**:
    - Kiểm tra Microsoft Visual C++ Redistributable đã được cài đặt
    - Chạy với quyền Administrator
    - Kiểm tra Windows Defender hoặc phần mềm diệt virus có chặn không
3. **Lỗi thiếu model file**: Đảm bảo file `yolov5su.pt` nằm trong cùng thư mục với EXE

### Gỡ lỗi

-   Nếu gặp vấn đề khi đóng gói, thử chạy lệnh PyInstaller trực tiếp với cờ `--debug`
-   Kiểm tra logs trong cửa sổ console khi chạy ứng dụng
-   Xem file `SNN_Traffic_Monitor_README.txt` để biết thêm thông tin
