"""
Trực quan hóa luồng hoạt động của ứng dụng giám sát giao thông SNN
Tạo biểu đồ cho từng bước xử lý để thêm vào slide trình bày
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
from PIL import Image

# Tạo thư mục đầu ra
OUTPUT_DIR = "flow_visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_flowchart():
    """Tạo biểu đồ luồng tổng quan của hệ thống"""
    plt.figure(figsize=(12, 8))
    
    # Định nghĩa vị trí các khối
    blocks = {
        "input": (0.5, 0.8, "Đầu vào\n(Camera/Video)"),
        "yolo": (0.5, 0.65, "YOLO Detection\nPhát hiện đối tượng"),
        "tracker": (0.5, 0.5, "Object Tracking\nTheo dõi đối tượng"),
        "vehicle_extraction": (0.5, 0.35, "Vehicle Extraction\nTrích xuất phương tiện"),
        "snn": (0.5, 0.2, "SNN\nPhân loại phương tiện"),
        "violation": (0.8, 0.5, "Violation Detection\nPhát hiện vi phạm"),
        "visualization": (0.2, 0.35, "Neuron Visualization\nTrực quan hóa hoạt động"),
        "output": (0.5, 0.05, "Hiển thị kết quả")
    }
    
    # Vẽ các khối
    for key, (x, y, label) in blocks.items():
        draw_block(x, y, label, key=="snn") # SNN là khối nổi bật
    
    # Vẽ các mũi tên
    draw_arrow(blocks["input"][0], blocks["input"][1], blocks["yolo"][0], blocks["yolo"][1])
    draw_arrow(blocks["yolo"][0], blocks["yolo"][1], blocks["tracker"][0], blocks["tracker"][1])
    draw_arrow(blocks["tracker"][0], blocks["tracker"][1], blocks["vehicle_extraction"][0], blocks["vehicle_extraction"][1])
    draw_arrow(blocks["vehicle_extraction"][0], blocks["vehicle_extraction"][1], blocks["snn"][0], blocks["snn"][1])
    draw_arrow(blocks["snn"][0], blocks["snn"][1], blocks["output"][0], blocks["output"][1])
    
    # Mũi tên phụ
    draw_arrow(blocks["tracker"][0], blocks["tracker"][1], blocks["violation"][0], blocks["violation"][1], style='dashed')
    draw_arrow(blocks["violation"][0], blocks["violation"][1], blocks["output"][0], blocks["output"][1], style='dashed')
    draw_arrow(blocks["snn"][0], blocks["snn"][1], blocks["visualization"][0], blocks["visualization"][1], style='dashed')
    draw_arrow(blocks["visualization"][0], blocks["visualization"][1], blocks["output"][0], blocks["output"][1], style='dashed')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_system_flowchart.png"), dpi=300)
    plt.close()

def draw_block(x, y, text, highlight=False):
    """Vẽ một khối trong biểu đồ luồng"""
    width, height = 0.25, 0.1
    rect = Rectangle((x-width/2, y-height/2), width, height, 
                   facecolor='lightblue' if not highlight else 'lightsalmon',
                   edgecolor='black', alpha=0.8)
    plt.gca().add_patch(rect)
    plt.text(x, y, text, ha='center', va='center', fontsize=9)

def draw_arrow(x1, y1, x2, y2, style='solid'):
    """Vẽ mũi tên giữa các khối"""
    plt.arrow(x1, y1-0.05, x2-x1, y2-y1+0.04, 
             head_width=0.02, head_length=0.02, fc='black', ec='black',
             length_includes_head=True, linestyle=style)

def create_yolo_detection_visualization():
    """Minh họa quá trình phát hiện đối tượng bằng YOLO"""
    plt.figure(figsize=(15, 6))
    
    # Mô phỏng hình ảnh đầu vào
    plt.subplot(1, 3, 1)
    img = np.random.rand(100, 100, 3)
    plt.imshow(img)
    plt.title("1. Đầu vào: Khung hình video")
    plt.axis('off')
    
    # Mô phỏng YOLO detection grid
    plt.subplot(1, 3, 2)
    grid_img = np.zeros((100, 100, 3))
    grid_size = 20
    for i in range(0, 100, grid_size):
        for j in range(0, 100, grid_size):
            if np.random.rand() > 0.8:
                confidence = np.random.rand()
                color = (0, confidence, 0)
                grid_img[i:i+grid_size, j:j+grid_size] = color
    plt.imshow(grid_img)
    plt.title("2. YOLO: Phát hiện đối tượng")
    plt.axis('off')
    
    # Mô phỏng kết quả phát hiện
    plt.subplot(1, 3, 3)
    result_img = img.copy()
    # Vẽ vài bounding box giả lập
    for _ in range(3):
        x, y = np.random.randint(10, 80), np.random.randint(10, 80)
        w, h = np.random.randint(10, 20), np.random.randint(10, 20)
        result_img[y:y+h, x:x+w, 0] = 0.9  # Đỏ cho bbox
    plt.imshow(result_img)
    plt.title("3. Kết quả: Bounding box đối tượng")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_yolo_detection_process.png"), dpi=300)
    plt.close()

def create_tracking_visualization():
    """Minh họa quá trình theo dõi đối tượng qua thời gian"""
    plt.figure(figsize=(15, 6))
    
    frames = 3
    for i in range(frames):
        plt.subplot(1, frames, i+1)
        img = np.random.rand(100, 100, 3)
        
        # Vẽ đối tượng di chuyển theo thời gian
        x = 20 + i*20
        y = 50
        w, h = 15, 10
        img[y:y+h, x:x+w, 0] = 0.9  # Đỏ cho bbox
        
        # Vẽ đường đi
        if i > 0:
            for j in range(i):
                prev_x = 20 + j*20 + w//2
                img[y+h//2, prev_x:x, 1] = 0.9  # Xanh lá cho đường đi
        
        plt.imshow(img)
        plt.title(f"Frame {i+1}: Theo dõi đối tượng")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_object_tracking.png"), dpi=300)
    plt.close()

def create_snn_process_visualization():
    """Minh họa quá trình xử lý của SNN"""
    plt.figure(figsize=(15, 10))
    
    # 1. Trích xuất phương tiện
    plt.subplot(2, 2, 1)
    vehicle_img = np.random.rand(50, 80, 3)
    plt.imshow(vehicle_img)
    plt.title("1. Phương tiện được trích xuất")
    plt.axis('off')
    
    # 2. Mã hóa thành chuỗi xung
    plt.subplot(2, 2, 2)
    time_steps = 10
    neurons = 20
    spike_data = np.random.rand(time_steps, neurons) > 0.8
    plt.imshow(spike_data.T, cmap='binary', aspect='auto')
    plt.title("2. Mã hóa thành chuỗi xung (spikes)")
    plt.xlabel("Thời gian")
    plt.ylabel("Neuron")
    
    # 3. Xử lý SNN qua các lớp
    plt.subplot(2, 2, 3)
    layers = 4
    neurons_per_layer = [20, 15, 10, 5]
    layer_labels = ["Input", "Conv1", "Conv2", "Output"]
    
    # Tạo dữ liệu hoạt động giả lập cho các lớp
    y_positions = []
    y_ticks = []
    total_neurons = 0
    
    for layer, n_neurons in enumerate(neurons_per_layer):
        y_ticks.append(total_neurons + n_neurons//2)
        for _ in range(n_neurons):
            y_positions.append(total_neurons)
            total_neurons += 1
    
    # Tạo dữ liệu spike ngẫu nhiên
    time_data = np.linspace(0, 1, 100)
    spike_times = []
    
    for y in range(total_neurons):
        neuron_spikes = time_data[np.random.rand(100) > 0.9]
        spike_times.extend([(t, y) for t in neuron_spikes])
    
    if spike_times:
        times, neurons = zip(*spike_times)
        plt.scatter(times, neurons, s=10, c='black', marker='|')
    
    plt.yticks(y_ticks, layer_labels)
    plt.title("3. Xử lý qua các lớp SNN")
    plt.xlabel("Thời gian")
    
    # 4. Kết quả phân loại
    plt.subplot(2, 2, 4)
    classes = ['Ô tô', 'Xe máy', 'Xe buýt', 'Xe tải', 'Xe đạp']
    confidences = [0.1, 0.7, 0.05, 0.1, 0.05]
    colors = ['gray', 'green' if max(confidences) == 0.7 else 'gray', 
             'gray', 'gray', 'gray']
    
    bars = plt.bar(classes, confidences, color=colors)
    plt.ylim(0, 1)
    plt.title("4. Kết quả phân loại")
    plt.ylabel("Độ tin cậy")
    plt.xticks(rotation=45)
    
    # Đánh dấu kết quả cao nhất
    idx = confidences.index(max(confidences))
    plt.text(idx, confidences[idx] + 0.05, "Phát hiện", 
            ha='center', va='bottom', color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_snn_processing.png"), dpi=300)
    plt.close()

def create_neuron_visualization():
    """Minh họa trực quan hóa hoạt động neuron"""
    plt.figure(figsize=(12, 8))
    
    # Tạo dữ liệu giả lập cho biểu đồ heatmap
    time_steps = 50
    num_neurons = 20
    
    # Tạo mẫu hoạt động thực tế
    activities = np.zeros((time_steps, num_neurons))
    
    # 1. Neuron theo dạng sóng
    for i in range(5):
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)
        amp = np.random.uniform(0.3, 0.9)
        activities[:, i] = amp * np.sin(freq * np.linspace(0, 4*np.pi, time_steps) + phase)
        activities[:, i] = np.maximum(0, activities[:, i])
    
    # 2. Neuron theo nhóm
    group_size = 3
    for i in range(5, 5+group_size):
        for t in [10, 25, 40]:
            activities[t:t+3, i] = np.random.uniform(0.5, 1.0, (3,))
    
    # 3. Neuron ngẫu nhiên
    for i in range(8, num_neurons):
        spike_prob = np.random.uniform(0.1, 0.3)
        random_mask = np.random.rand(time_steps) < spike_prob
        random_values = np.random.uniform(0.3, 1.0, size=time_steps)
        activities[:, i] = random_values * random_mask
    
    # Vẽ heatmap
    plt.imshow(activities.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Mức độ kích hoạt')
    plt.xlabel('Thời gian', fontsize=12)
    plt.ylabel('Neuron', fontsize=12)
    plt.title('Hoạt động Neuron - Lớp 0', fontsize=14, fontweight='bold')
    
    # Thêm chú thích
    plt.annotate('Neuron dạng sóng', xy=(25, 2), xytext=(30, 2), 
                arrowprops=dict(facecolor='white', shrink=0.05))
    
    plt.annotate('Neuron hoạt động theo nhóm', xy=(25, 6), xytext=(30, 6), 
                arrowprops=dict(facecolor='white', shrink=0.05))
    
    plt.annotate('Neuron hoạt động ngẫu nhiên', xy=(25, 15), xytext=(30, 15), 
                arrowprops=dict(facecolor='white', shrink=0.05))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_neuron_activity_visualization.png"), dpi=300)
    plt.close()

def create_violation_detection():
    """Minh họa quá trình phát hiện vi phạm tốc độ"""
    plt.figure(figsize=(15, 6))
    
    # Khởi tạo hình nền
    img = np.zeros((100, 300, 3))
    img[:, :, :] = 0.9  # Nền xám nhạt
    
    # Vẽ đường đi mô phỏng con đường
    road_color = (0.4, 0.4, 0.4)  # Màu xám đậm
    road_y = 40
    road_height = 20
    img[road_y:road_y+road_height, :, :] = road_color
    
    # Vẽ vạch kẻ đường
    for i in range(0, 300, 30):
        img[road_y+road_height//2-1:road_y+road_height//2+1, i:i+15, :] = 0.9  # Màu trắng
    
    # Vẽ xe di chuyển ở các vị trí khác nhau với tốc độ
    car_width, car_height = 20, 10
    positions = [50, 150, 250]
    speeds = [15, 30, 65]  # km/h
    
    for i, (pos, speed) in enumerate(zip(positions, speeds)):
        # Vẽ xe
        car_color = (0, 0.5, 1) if speed <= 30 else (1, 0, 0)  # Xanh nếu tốc độ ok, đỏ nếu vượt tốc
        img[road_y+road_height//2-car_height//2:road_y+road_height//2+car_height//2, 
           pos:pos+car_width, :] = car_color
        
        # Hiển thị tốc độ
        plt.text(pos+car_width//2, road_y+road_height//2, f"{speed}", 
                color='white', ha='center', va='center', fontsize=9)
        
        # Đánh dấu vi phạm
        if speed > 30:
            plt.text(pos+car_width//2, road_y-10, "VI PHẠM", 
                    color='red', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Hiển thị giới hạn tốc độ
    plt.text(10, 10, "Giới hạn tốc độ: 30 km/h", fontsize=12, color='black')
    
    plt.imshow(img)
    plt.title("Phát hiện vi phạm tốc độ", fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "6_speed_violation_detection.png"), dpi=300)
    plt.close()

def create_user_interface_visualization():
    """Minh họa giao diện người dùng của ứng dụng"""
    plt.figure(figsize=(15, 10))
    
    # Tạo mô phỏng giao diện
    # Khung chính
    plt.axhline(y=0.5, xmin=0.25, xmax=0.75, color='black', linestyle='-')
    plt.axvline(x=0.75, ymin=0, ymax=0.5, color='black', linestyle='-')
    plt.axvline(x=0.25, ymin=0, ymax=0.5, color='black', linestyle='-')
    plt.axhline(y=0, xmin=0.25, xmax=0.75, color='black', linestyle='-')
    
    # Khung video
    video_rect = Rectangle((0.3, 0.1), 0.4, 0.3, fill=False, color='black')
    plt.gca().add_patch(video_rect)
    plt.text(0.5, 0.25, "Hiển thị video", ha='center', va='center')
    
    # Panel điều khiển
    control_rect = Rectangle((0.8, 0.1), 0.15, 0.3, fill=False, color='black')
    plt.gca().add_patch(control_rect)
    plt.text(0.875, 0.4, "Điều khiển", ha='center')
    plt.text(0.875, 0.35, "[ ] Hiển thị thống kê", ha='center', fontsize=8)
    plt.text(0.875, 0.32, "[x] Hiển thị theo dõi", ha='center', fontsize=8)
    plt.text(0.875, 0.29, "[x] Hiển thị vi phạm", ha='center', fontsize=8)
    plt.text(0.875, 0.26, "[x] Hiển thị neuron", ha='center', fontsize=8)
    plt.text(0.875, 0.2, "Giới hạn tốc độ: 30", ha='center', fontsize=8)
    
    # Panel thống kê
    stats_rect = Rectangle((0.3, 0.6), 0.4, 0.3, fill=False, color='black')
    plt.gca().add_patch(stats_rect)
    plt.text(0.5, 0.75, "Biểu đồ thống kê", ha='center')
    
    # Bảng vi phạm
    violation_rect = Rectangle((0.8, 0.6), 0.15, 0.3, fill=False, color='black')
    plt.gca().add_patch(violation_rect)
    plt.text(0.875, 0.75, "Vi phạm giao thông", ha='center')
    
    # Panel neuron
    neuron_rect = Rectangle((0.05, 0.1), 0.15, 0.8, fill=False, color='black')
    plt.gca().add_patch(neuron_rect)
    plt.text(0.125, 0.5, "Biểu đồ\nhoạt động\nneuron", ha='center', va='center')
    
    plt.axis('off')
    plt.title("Giao diện ứng dụng giám sát giao thông SNN", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "7_user_interface.png"), dpi=300)
    plt.close()

def main():
    """Tạo tất cả các biểu đồ"""
    print("Đang tạo biểu đồ trực quan hóa...")
    create_flowchart()
    create_yolo_detection_visualization()
    create_tracking_visualization()
    create_snn_process_visualization()
    create_neuron_visualization()
    create_violation_detection()
    create_user_interface_visualization()
    print(f"Đã tạo xong tất cả biểu đồ trong thư mục '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main() 