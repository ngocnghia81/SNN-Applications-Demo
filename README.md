# SNN Traffic Monitoring

Hệ thống giám sát giao thông thông minh sử dụng mạng nơ-ron xung (Spiking Neural Networks) kết hợp với phát hiện đối tượng để theo dõi, phân loại phương tiện và phát hiện vi phạm.

## Tính năng chính

-   **Phát hiện phương tiện:** Sử dụng YOLOv5 để phát hiện các loại phương tiện như ô tô, xe máy, xe buýt, xe tải, xe đạp.
-   **Theo dõi phương tiện:** Theo dõi quỹ đạo di chuyển của phương tiện qua các khung hình.
-   **Phân loại phương tiện:** Sử dụng SNN để phân loại chi tiết loại phương tiện.
-   **Phát hiện vi phạm:** Theo dõi và phát hiện vi phạm tốc độ.
-   **Trực quan hóa SNN:** Hiển thị hoạt động của các neuron trong mạng SNN theo thời gian thực.
-   **Thống kê giao thông:** Tạo báo cáo và biểu đồ thống kê về lưu lượng giao thông.
-   **Hỗ trợ nhiều nguồn dữ liệu:** Hoạt động với camera trực tiếp, video ghi sẵn và hình ảnh.

## Cấu trúc dự án

```
snn-traffic/
├── configs/                  # Cấu hình ứng dụng
├── data/                     # Dữ liệu
│   ├── captured/             # Khung hình đã chụp
│   ├── demo_videos/          # Video demo
│   ├── mnist/                # Dữ liệu MNIST (cho huấn luyện)
│   └── processed/            # Dữ liệu đã xử lý
├── docs/                     # Tài liệu
│   └── images/               # Hình ảnh, biểu đồ
│       └── flow/             # Trực quan hóa luồng hoạt động
├── models/                   # Mô hình đã huấn luyện
│   ├── mnist/                # Mô hình SNN cho MNIST
│   └── traffic/              # Mô hình cho phân tích giao thông
├── notebooks/                # Jupyter notebooks
├── src/                      # Mã nguồn
│   ├── snn_examples/         # Các ví dụ về SNN
│   │   ├── bindsnet/         # Ví dụ sử dụng BindsNET
│   │   ├── nengo/            # Ví dụ sử dụng Nengo
│   │   ├── pysnn/            # Ví dụ sử dụng PySNN
│   │   └── spikingjelly/     # Ví dụ sử dụng SpikingJelly
│   └── snn_traffic/          # Ứng dụng giám sát giao thông
│       ├── app/              # Ứng dụng chính
│       ├── detectors/        # Phát hiện đối tượng
│       ├── models/           # Mô hình SNN
│       ├── utils/            # Tiện ích
│       └── visualization/    # Trực quan hóa
├── tests/                    # Kiểm thử
├── utils/                    # Tiện ích chung
├── main.py                   # Script khởi chạy ứng dụng
├── requirements.txt          # Các thư viện phụ thuộc
├── run.sh                    # Script chạy ứng dụng
└── setup.py                  # Cấu hình cài đặt
```

## Yêu cầu hệ thống

-   Python 3.8 hoặc cao hơn
-   PyTorch 1.8.0 hoặc cao hơn
-   OpenCV 4.5.0 hoặc cao hơn
-   Thư viện YOLOv5 (Ultralytics)
-   SpikingJelly
-   Các thư viện khác được liệt kê trong `requirements.txt`

## Cài đặt

1. Clone repository:

    ```bash
    git clone https://github.com/yourusername/snn-traffic.git
    cd snn-traffic
    ```

2. Cài đặt môi trường ảo (tùy chọn nhưng khuyến nghị):

    ```bash
    python -m venv myenv-new
    source myenv-new/bin/activate  # Linux/Mac
    # hoặc
    myenv-new\Scripts\activate  # Windows
    ```

3. Cài đặt các thư viện cần thiết:

    ```bash
    pip install -r requirements.txt
    ```

4. Cài đặt gói:
    ```bash
    pip install -e .
    ```

## Sử dụng

### Chạy ứng dụng

Sử dụng script `run.sh` (khuyến nghị):

```bash
# Chạy với camera mặc định
./run.sh

# Chạy với camera cụ thể
./run.sh --camera 1

# Chạy với file video
./run.sh --video data/demo_videos/highway.mp4

# Thiết lập giới hạn tốc độ
./run.sh --speed-limit 40
```

Hoặc sử dụng Python trực tiếp:

```bash
python main.py --video data/demo_videos/highway.mp4
```

### Giao diện người dùng

Ứng dụng cung cấp giao diện người dùng đồ họa với các tính năng:

-   **Bảng điều khiển:** Điều khiển chức năng hiển thị và thiết lập tham số
-   **Khung video:** Hiển thị video từ camera hoặc file được xử lý
-   **Biểu đồ thống kê:** Hiển thị số lượng phương tiện theo loại
-   **Bảng vi phạm:** Hiển thị các vi phạm được phát hiện
-   **Biểu đồ neuron:** Hiển thị hoạt động của các neuron trong SNN

## Cách hoạt động

1. **Đầu vào:** Khung hình từ camera hoặc video được đưa vào hệ thống
2. **Phát hiện:** YOLOv5 phát hiện các phương tiện trong khung hình
3. **Theo dõi:** Hệ thống theo dõi các phương tiện qua các khung hình liên tiếp
4. **Trích xuất:** Cắt ra hình ảnh của từng phương tiện
5. **Phân loại SNN:** Mạng SNN phân loại chi tiết loại phương tiện
6. **Phát hiện vi phạm:** Phân tích hành vi và tốc độ để phát hiện vi phạm
7. **Hiển thị kết quả:** Kết quả được hiển thị trong giao diện đồ họa

## Tạo biểu đồ trực quan hóa

Để tạo các biểu đồ trực quan hóa luồng hoạt động của ứng dụng:

```bash
python src/snn_traffic/visualization/traffic_flow_visualization.py
```

Các biểu đồ sẽ được lưu trong thư mục `docs/images/flow`.

## Tác giả

-   Nguyễn Ngọc Nghĩa
-   Email: ngocnghia2004nn@gmail.com

## Giấy phép

MIT License
