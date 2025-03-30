"""
Trực quan hóa luồng hoạt động của mạng SNN trong nhận dạng MNIST
Tạo biểu đồ cho từng bước xử lý để thêm vào slide trình bày
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
from PIL import Image
import torch
import torch.nn.functional as F

# Tạo thư mục đầu ra
OUTPUT_DIR = "mnist_visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_snn_mnist_flowchart():
    """Tạo biểu đồ luồng tổng quan của quy trình xử lý MNIST với SNN"""
    plt.figure(figsize=(12, 8))
    
    # Định nghĩa vị trí các khối
    blocks = {
        "input": (0.5, 0.85, "Đầu vào\n(Ảnh MNIST)"),
        "preprocessing": (0.5, 0.7, "Tiền xử lý\nNormalization"),
        "encoding": (0.5, 0.55, "Mã hóa\nChuyển thành xung"),
        "snn": (0.5, 0.4, "SNN\nMạng neural xung"),
        "decoding": (0.5, 0.25, "Giải mã\nTích lũy xung"),
        "classification": (0.5, 0.1, "Phân loại\nChữ số 0-9")
    }
    
    # Vẽ các khối
    for key, (x, y, label) in blocks.items():
        draw_block(x, y, label, key=="snn") # SNN là khối nổi bật
    
    # Vẽ các mũi tên
    prev_key = None
    for key in blocks:
        if prev_key:
            draw_arrow(blocks[prev_key][0], blocks[prev_key][1], blocks[key][0], blocks[key][1])
        prev_key = key
        
    # Thêm chú thích về xung
    plt.text(0.8, 0.55, "Chuỗi xung (spike train)", fontsize=8, ha='center')
    plt.text(0.8, 0.4, "Neuron LIF\n(Leaky Integrate-and-Fire)", fontsize=8, ha='center')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_mnist_snn_flowchart.png"), dpi=300)
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

def create_mnist_preprocessing_visualization():
    """Minh họa quá trình tiền xử lý MNIST"""
    plt.figure(figsize=(15, 6))
    
    # Tạo chữ số thực tế - số 5
    digit = np.zeros((28, 28))
    
    # Vẽ số 5 rõ ràng hơn
    # Vẽ nét ngang trên cùng
    digit[3:5, 5:20] = 0.9
    # Vẽ nét dọc bên trái trên
    digit[5:13, 5:7] = 0.9
    # Vẽ nét ngang giữa
    digit[12:14, 5:20] = 0.9
    # Vẽ nét dọc bên phải dưới
    digit[14:22, 17:20] = 0.9
    # Vẽ nét ngang dưới cùng
    digit[21:23, 5:18] = 0.9
    
    # 1. Hình ảnh MNIST gốc
    plt.subplot(1, 3, 1)
    plt.imshow(digit, cmap='gray')
    plt.title("1. Đầu vào: Ảnh MNIST (28x28)")
    plt.axis('off')
    
    # 2. Mô phỏng quá trình chuẩn hóa
    plt.subplot(1, 3, 2)
    normalized = (digit - 0.1307) / 0.3081  # Giá trị chuẩn hóa MNIST thông thường
    plt.imshow(normalized, cmap='gray')
    plt.title("2. Chuẩn hóa: (x - 0.1307) / 0.3081")
    plt.axis('off')
    
    # 3. Mô phỏng chuyển đổi thành tensor
    plt.subplot(1, 3, 3)
    # Hiển thị ví dụ dạng tensor - lấy mẫu 5x5 gần tâm chữ số
    tensor_view = normalized[10:15, 10:15]
    plt.imshow(tensor_view, cmap='viridis')
    plt.title("3. Chuyển đổi: PyTorch Tensor")
    plt.colorbar(label='Giá trị Chuẩn hóa')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_mnist_preprocessing.png"), dpi=300)
    plt.close()

def create_spike_encoding_visualization():
    """Minh họa quá trình mã hóa hình ảnh MNIST thành xung"""
    plt.figure(figsize=(15, 8))
    
    # Tạo chữ số thực tế - số 5
    digit = np.zeros((28, 28))
    
    # Vẽ số 5 rõ ràng
    # Vẽ nét ngang trên cùng
    digit[3:5, 5:20] = 0.9
    # Vẽ nét dọc bên trái trên
    digit[5:13, 5:7] = 0.9
    # Vẽ nét ngang giữa
    digit[12:14, 5:20] = 0.9
    # Vẽ nét dọc bên phải dưới
    digit[14:22, 17:20] = 0.9
    # Vẽ nét ngang dưới cùng
    digit[21:23, 5:18] = 0.9
    
    # 1. Hình ảnh MNIST gốc
    plt.subplot(2, 2, 1)
    plt.imshow(digit, cmap='gray')
    plt.title("1. Hình ảnh MNIST")
    plt.axis('off')
    
    # 2. Mô phỏng mã hóa xung kiểu Rate coding
    plt.subplot(2, 2, 2)
    time_steps = 16
    # Chuyển đổi cường độ pixel thành tỷ lệ xung
    # pixel càng sáng, xác suất phát xung càng cao
    spike_rates = digit.flatten()
    
    # Tạo mô phỏng xung theo thời gian
    spike_train = np.random.rand(time_steps, len(spike_rates)) < np.tile(spike_rates, (time_steps, 1))
    
    # Tạo biểu đồ raster plot của xung - chỉ hiển thị 100 neuron đầu tiên để rõ ràng hơn
    plt.imshow(spike_train[:, :100], aspect='auto', cmap='binary', interpolation='none')
    plt.title("2. Rate Coding: Mã hóa cường độ -> xác suất xung")
    plt.xlabel("Pixel (Neuron), chỉ hiển thị 100 đầu tiên")
    plt.ylabel("Thời gian (time steps)")
    
    # 3. Mô phỏng mã hóa xung kiểu Temporal coding
    plt.subplot(2, 2, 3)
    
    # Trong Temporal coding, giá trị càng lớn thì phát xung càng sớm
    # Pixel càng sáng thì phát xung càng sớm
    # Tạo ma trận thời gian xung, -1 là không có xung
    temporal_spikes = np.ones((time_steps, len(spike_rates))) * -1
    
    # Ánh xạ cường độ pixel sang thời gian phát xung
    for i, intensity in enumerate(spike_rates):
        if intensity > 0.1:  # Pixel có cường độ đủ lớn
            # pixel càng sáng thì phát xung càng sớm
            spike_time = int((1 - intensity) * (time_steps - 1))
            if spike_time < time_steps:
                temporal_spikes[spike_time, i] = 1
    
    # Hiển thị chỉ 100 neuron đầu tiên cho rõ ràng
    plt.imshow(temporal_spikes[:, :100], aspect='auto', cmap='viridis', interpolation='none')
    plt.title("3. Temporal Coding: Cường độ -> thời gian xung")
    plt.xlabel("Pixel (Neuron), chỉ hiển thị 100 đầu tiên")
    plt.ylabel("Thời gian (time steps)")
    plt.colorbar(label='Hoạt động xung')
    
    # 4. Mô phỏng mã hóa xung kiểu Direct Input (lặp lại trên mỗi time step)
    plt.subplot(2, 2, 4)
    
    # Để minh họa rõ ràng, hiển thị lại hình ảnh số 5 được chuyển về dạng chuỗi thời gian
    # Reshape lại để thấy rõ hình dạng
    reshaped_digit = np.zeros((time_steps, 28, 28))
    for t in range(time_steps):
        reshaped_digit[t] = digit  # Mỗi time step đều có hình ảnh giống nhau
    
    # Hiển thị 3 time step đầu tiên
    plt.imshow(np.vstack([reshaped_digit[0], reshaped_digit[1], reshaped_digit[2]]), cmap='viridis')
    plt.title("4. Direct Input: Lặp lại hình ảnh theo thời gian")
    plt.xlabel("Chiều rộng pixel")
    plt.ylabel("Time steps × Chiều cao")
    plt.colorbar(label='Cường độ Pixel')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_spike_encoding.png"), dpi=300)
    plt.close()

def create_snn_network_visualization():
    """Minh họa kiến trúc mạng SNN cho MNIST"""
    plt.figure(figsize=(12, 8))
    
    # Vẽ kiến trúc mạng nơ-ron SNN
    layer_sizes = [784, 400, 10]  # Input, Hidden, Output
    layer_names = ["Input\n784 neurons", "Hidden\nLIF neurons", "Output\n10 neurons"]
    layer_colors = ['lightblue', 'lightsalmon', 'lightgreen']
    layer_positions = [0.2, 0.5, 0.8]
    
    # Vẽ các lớp
    for i, (size, name, color, pos) in enumerate(zip(layer_sizes, layer_names, layer_colors, layer_positions)):
        # Hiển thị một số lượng hạn chế neuron để rõ ràng
        display_neurons = min(size, 20)
        
        # Vẽ hình chữ nhật đại diện cho lớp
        rect = Rectangle((pos - 0.1, 0.1), 0.2, 0.6, facecolor=color, alpha=0.5, edgecolor='black')
        plt.gca().add_patch(rect)
        
        # Vẽ các neuron trong lớp
        neuron_y = np.linspace(0.15, 0.65, display_neurons)
        neuron_x = np.ones(display_neurons) * pos
        
        # Vẽ các neuron
        plt.scatter(neuron_x, neuron_y, s=100, color=color, edgecolor='black', zorder=3)
        
        # Chỉ hiển thị số lượng neuron thực tế trong lớp output
        if i == 2:  # Lớp output
            for j, y in enumerate(neuron_y):
                plt.text(pos, y, str(j), ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Thêm tên lớp
        plt.text(pos, 0.75, name, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Nếu không phải lớp cuối, vẽ kết nối đến lớp kế tiếp
        if i < len(layer_sizes) - 1:
            next_pos = layer_positions[i + 1]
            next_size = min(layer_sizes[i + 1], 20)
            next_neuron_y = np.linspace(0.15, 0.65, next_size)
            
            # Chỉ hiển thị một số kết nối để tránh quá rối
            for j in range(0, display_neurons, 4):
                for k in range(0, next_size, 4):
                    plt.plot([pos, next_pos], [neuron_y[j], next_neuron_y[k]], 
                           'k-', alpha=0.1, linewidth=0.5, zorder=1)
    
    # Mô tả kiểu neuron
    plt.text(0.5, 0.9, "Kiến trúc mạng SNN cho nhận dạng MNIST", 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Thêm chú thích về LIF neuron
    lif_detail = """
    LIF Neuron (Leaky Integrate-and-Fire):
    - Tích lũy điện thế theo thời gian
    - Phát xung khi vượt ngưỡng
    - Reset sau khi phát xung
    - Rò rỉ điện thế theo thời gian
    """
    plt.text(0.5, 0.02, lif_detail, ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_snn_architecture.png"), dpi=300)
    plt.close()

def create_lif_neuron_visualization():
    """Minh họa hoạt động của neuron LIF (Leaky Integrate-and-Fire)"""
    plt.figure(figsize=(15, 10))
    
    # Thời gian mô phỏng
    time = np.arange(0, 100)
    
    # Tham số neuron LIF
    threshold = 1.0  # Ngưỡng kích hoạt
    tau = 10.0       # Hằng số thời gian
    rest = 0.0       # Điện thế nghỉ
    reset = 0.0      # Điện thế sau khi phát xung
    
    # Tạo dữ liệu đầu vào
    input_spikes = np.zeros_like(time, dtype=float)
    input_spikes[[10, 15, 16, 17, 40, 42, 44, 70, 80, 81, 82]] = 0.5
    
    # Mô phỏng hoạt động của neuron LIF
    v_membrane = np.zeros_like(time, dtype=float)
    output_spikes = np.zeros_like(time, dtype=float)
    
    # Mô phỏng động học của neuron
    for i in range(1, len(time)):
        # Tính toán rò rỉ điện thế
        v_leak = -(v_membrane[i-1] - rest) / tau
        
        # Cập nhật điện thế màng
        v_membrane[i] = v_membrane[i-1] + v_leak + input_spikes[i]
        
        # Kiểm tra phát xung
        if v_membrane[i] >= threshold:
            output_spikes[i] = 1.0
            v_membrane[i] = reset
    
    # Vẽ 4 biểu đồ
    # 1. Xung đầu vào
    plt.subplot(4, 1, 1)
    plt.stem(time, input_spikes, linefmt='C0-', markerfmt='C0o', basefmt='k-')
    plt.title("1. Xung đầu vào")
    plt.ylabel("Cường độ")
    plt.ylim(-0.1, 0.6)
    
    # 2. Điện thế màng
    plt.subplot(4, 1, 2)
    plt.plot(time, v_membrane, 'C1-')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Ngưỡng')
    plt.title("2. Điện thế màng")
    plt.ylabel("Điện thế (V)")
    plt.legend()
    
    # 3. Xung đầu ra
    plt.subplot(4, 1, 3)
    plt.stem(time, output_spikes, linefmt='C2-', markerfmt='C2o', basefmt='k-')
    plt.title("3. Xung đầu ra")
    plt.ylabel("Xung")
    plt.ylim(-0.1, 1.1)
    
    # 4. Mô tả quá trình tích hợp và phóng xung
    plt.subplot(4, 1, 4)
    
    # Vẽ điện thế màng
    plt.plot(time, v_membrane, 'C1-', label='Điện thế màng')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Ngưỡng')
    
    # Đánh dấu xung đầu vào
    for i in np.where(input_spikes > 0)[0]:
        plt.axvline(x=i, color='C0', alpha=0.3)
    
    # Đánh dấu xung đầu ra
    for i in np.where(output_spikes > 0)[0]:
        plt.axvline(x=i, color='g', linestyle='-', linewidth=2, alpha=0.5)
        # Vẽ mũi tên chỉ điểm phóng xung
        plt.annotate('', xy=(i, threshold), xytext=(i, threshold+0.2),
                    arrowprops=dict(facecolor='g', shrink=0.05))
    
    plt.title("4. Quá trình tích hợp và phóng xung")
    plt.xlabel("Thời gian (time steps)")
    plt.ylabel("Điện thế (V)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_lif_neuron_dynamics.png"), dpi=300)
    plt.close()

def create_mnist_classification_visualization():
    """Minh họa quá trình phân loại MNIST từ mạng SNN"""
    plt.figure(figsize=(15, 10))
    
    # 1. Tạo dữ liệu mẫu hoạt động neuron theo thời gian
    plt.subplot(2, 2, 1)
    time_steps = 16
    neurons = 10  # 10 neuron cho 10 chữ số
    
    # Tạo dữ liệu hoạt động xung cho 10 neuron đầu ra
    output_spikes = np.zeros((time_steps, neurons))
    
    # Giả sử neuron 5 (ứng với chữ số 5) có nhiều xung nhất
    for i in range(neurons):
        # Tỷ lệ phát xung khác nhau cho mỗi neuron, neuron 5 cao nhất
        rate = 0.2 if i != 5 else 0.7
        output_spikes[:, i] = np.random.rand(time_steps) < rate
    
    # Vẽ hoạt động xung
    plt.imshow(output_spikes, aspect='auto', cmap='viridis')
    plt.colorbar(label='Phóng xung (0/1)')
    plt.title("1. Hoạt động của 10 neuron đầu ra theo thời gian")
    plt.xlabel("Neuron (0-9)")
    plt.ylabel("Thời gian (time steps)")
    
    # 2. Biểu đồ tích lũy xung
    plt.subplot(2, 2, 2)
    
    # Tính tổng số xung trên mỗi neuron
    spike_counts = output_spikes.sum(axis=0)
    
    # Vẽ biểu đồ tổng số xung
    classes = np.arange(10)
    colors = ['#1f77b4'] * 10
    colors[5] = '#d62728'  # Đánh dấu lớp được dự đoán
    
    barplot = plt.bar(classes, spike_counts, color=colors)
    plt.xlabel("Chữ số (0-9)")
    plt.ylabel("Tổng số xung")
    plt.title("2. Tích lũy xung trên mỗi neuron đầu ra")
    
    # 3. Xác suất dự đoán (softmax)
    plt.subplot(2, 2, 3)
    
    # Áp dụng hàm softmax để có xác suất
    softmax_probs = np.exp(spike_counts) / np.sum(np.exp(spike_counts))
    
    barplot = plt.bar(classes, softmax_probs, color=colors)
    plt.xlabel("Chữ số (0-9)")
    plt.ylabel("Xác suất")
    plt.title("3. Xác suất dự đoán (softmax)")
    
    # Đánh dấu giá trị lớn nhất
    max_idx = np.argmax(softmax_probs)
    plt.text(max_idx, softmax_probs[max_idx], f"Dự đoán: {max_idx}", 
             ha='center', va='bottom', fontweight='bold')
    
    # 4. Kết quả phân loại
    plt.subplot(2, 2, 4)
    
    # Tạo một ma trận đơn vị để hiển thị kết quả
    result_img = np.zeros((10, 10))
    result_img[5, 5] = 1  # Đánh dấu kết quả dự đoán
    
    plt.imshow(result_img, cmap='gist_yarg')
    plt.xticks([])
    plt.yticks([])
    plt.title(f"4. Kết quả phân loại: Chữ số 5")
    plt.axis('off')
    
    # Vẽ chữ số 5 lớn ở giữa
    plt.text(5, 5, "5", fontsize=50, ha='center', va='center', 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    # Thêm thông tin về độ chính xác
    plt.text(5, 9, f"Độ tin cậy: {softmax_probs[5]*100:.1f}%", 
            ha='center', va='top', fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "6_mnist_classification.png"), dpi=300)
    plt.close()

def create_training_accuracy_visualization():
    """Minh họa quá trình huấn luyện và độ chính xác của SNN trên MNIST"""
    plt.figure(figsize=(15, 6))
    
    # 1. Đồ thị huấn luyện
    plt.subplot(1, 2, 1)
    
    # Mô phỏng dữ liệu huấn luyện
    epochs = np.arange(1, 21)
    train_loss = 2.5 * np.exp(-0.15 * epochs) + 0.1 * np.random.randn(len(epochs))
    val_loss = 2.3 * np.exp(-0.12 * epochs) + 0.2 + 0.15 * np.random.randn(len(epochs))
    
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Quá trình huấn luyện SNN')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Độ chính xác trên tập kiểm tra
    plt.subplot(1, 2, 2)
    
    # Mô phỏng độ chính xác theo thời gian mô phỏng
    time_steps = np.arange(1, 17)
    accuracy = 100 * (1 - np.exp(-0.3 * time_steps)) + 5 * np.random.randn(len(time_steps))
    accuracy = np.clip(accuracy, 0, 100)
    
    plt.plot(time_steps, accuracy, 'g-o')
    plt.xlabel('Số bước thời gian')
    plt.ylabel('Độ chính xác (%)')
    plt.title('Độ chính xác trên tập kiểm tra theo thời gian mô phỏng')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Đánh dấu giá trị đặc biệt
    max_acc_idx = np.argmax(accuracy)
    plt.annotate(f'Tối đa: {accuracy[max_acc_idx]:.1f}%', 
                xy=(time_steps[max_acc_idx], accuracy[max_acc_idx]),
                xytext=(time_steps[max_acc_idx]-3, accuracy[max_acc_idx]+5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "7_training_accuracy.png"), dpi=300)
    plt.close()

def main():
    """Tạo tất cả các biểu đồ"""
    print("Đang tạo biểu đồ trực quan hóa MNIST...")
    create_snn_mnist_flowchart()
    create_mnist_preprocessing_visualization()
    create_spike_encoding_visualization()
    create_snn_network_visualization()
    create_lif_neuron_visualization()
    create_mnist_classification_visualization()
    create_training_accuracy_visualization()
    print(f"Đã tạo xong tất cả biểu đồ trong thư mục '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main() 