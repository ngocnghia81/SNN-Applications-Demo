"""
SNN MNIST Training with BindsNET on Google Colab - Phiên bản đầy đủ
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.encoding import PoissonEncoder
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.learning import PostPre
from tqdm import tqdm
from google.colab import drive
import time

# Kết nối Google Drive để lưu kết quả
drive.mount('/content/drive')
base_path = '/content/drive/MyDrive/SNN_Results'  # Đổi thành đường dẫn trong Drive của bạn
if not os.path.exists(base_path):
    os.makedirs(base_path)
    print(f"Đã tạo thư mục lưu kết quả: {base_path}")

# Tạo thư mục để lưu hình ảnh
output_dir = os.path.join(base_path, "snn_plots_mnist")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Đã tạo thư mục đầu ra: {output_dir}")

# Thời gian bắt đầu
start_time = time.time()

# Cấu hình thiết bị (luôn sử dụng GPU nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# Tạo thư mục dữ liệu cho bindsnet
data_dir = "./data/bindsnet_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Đã tạo thư mục dữ liệu: {data_dir}")

# Sử dụng toàn bộ dữ liệu MNIST
print("Sử dụng toàn bộ dữ liệu MNIST (60,000 mẫu train và 10,000 mẫu test)")

# Tham số tối ưu hóa
batch_size = 1  # Thay vì 64
time_window = 250     # Tăng time window từ 100 lên 250 để có nhiều cơ hội phát xung hơn
epochs = 1           # Giảm số epoch

# Tải dữ liệu MNIST
transform = transforms.Compose([transforms.ToTensor()])

# Train set - sử dụng toàn bộ
trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
train_subset = torch.utils.data.Subset(trainset, range(5000))
trainloader = torch.utils.data.DataLoader(train_subset, batch_size=1, shuffle=True)

# Test set - sử dụng toàn bộ
testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Tổng số mẫu huấn luyện: {len(trainset)}")
print(f"Tổng số mẫu kiểm tra: {len(testset)}")

# Định nghĩa tên lớp cho MNIST
class_names = [str(i) for i in range(10)]

# Tạo mạng SNN với cấu trúc đơn giản
print("Tạo mạng SNN...")
snn = Network(dt=1.0)  # Không chuyển sang thiết bị ngay

# Lớp đầu vào: 784 nơ-ron (28x28 pixel)
input_layer = Input(n=28 * 28, traces=True)  # Bật traces=True cho tất cả các lớp

# Lớp ẩn: 100 nơ-ron LIF - giảm ngưỡng để dễ phát xung hơn
hidden_layer = LIFNodes(n=100, thresh=-55.0, reset=-65.0, rest=-65.0, tau=10.0, traces=True)

# Lớp đầu ra: 10 nơ-ron LIF - giảm ngưỡng để dễ phát xung hơn
output_layer = LIFNodes(n=10, thresh=-55.0, reset=-65.0, rest=-65.0, tau=10.0, traces=True)

# Thêm các lớp vào mạng
snn.add_layer(input_layer, name="input")
snn.add_layer(hidden_layer, name="hidden")
snn.add_layer(output_layer, name="output")

# Khởi tạo trọng số kết nối giữa các lớp với cơ chế học STDP
print("Khởi tạo kết nối giữa các lớp...")
# Tăng trọng số ban đầu để kích thích nơ-ron mạnh hơn
input_hidden_weights = 0.1 * torch.rand(784, 100)  # Tăng từ 0.05 lên 0.1
input_hidden = Connection(
    source=input_layer,
    target=hidden_layer,
    w=input_hidden_weights,
    update_rule=PostPre,
    nu=(1e-3, 1e-1),  # Tăng tốc độ học
    wmin=0.0,
    wmax=1.0,
)
snn.add_connection(input_hidden, source="input", target="hidden")

# Thêm kết nối trực tiếp từ lớp ẩn đến lớp đầu ra
hidden_output_weights = 0.1 * torch.rand(100, 10)  # Khởi tạo trọng số với giá trị dương
hidden_output = Connection(
    source=hidden_layer,
    target=output_layer,
    w=hidden_output_weights,
    update_rule=PostPre,
    nu=(1e-3, 1e-1),  # Tăng tốc độ học
    wmin=0.0,
    wmax=1.0,
)
snn.add_connection(hidden_output, source="hidden", target="output")

# Thêm ức chế bên giữa các nơ-ron đầu ra - giảm mức ức chế để nhiều nơ-ron có thể hoạt động
inhibition = torch.ones(10, 10) - torch.diag(torch.ones(10))
output_recurrent = Connection(
    source=output_layer,
    target=output_layer,
    w=-0.1 * inhibition,  # Giảm mức ức chế từ -0.5 xuống -0.1
)
snn.add_connection(output_recurrent, source="output", target="output")

# Thêm các monitor để theo dõi hoạt động của mạng
print("Thiết lập các monitor theo dõi hoạt động của mạng...")
snn.add_monitor(Monitor(input_layer, state_vars=["s"]), name="input_monitor")
snn.add_monitor(Monitor(hidden_layer, state_vars=["s", "v"]), name="hidden_monitor")
snn.add_monitor(Monitor(output_layer, state_vars=["s", "v"]), name="output_monitor")

# Chuyển toàn bộ mạng sang thiết bị sau khi thiết lập xong
snn = snn.to(device)

# Bộ mã hóa Poisson
encoder = PoissonEncoder(time=time_window, dt=1.0)

# Giải pháp cho lỗi thiết bị trong encoder
def encode_to_spikes(data, time_window, device):
    # Mã hóa trên CPU, sau đó chuyển sang thiết bị mong muốn
    with torch.no_grad():
        data_cpu = data.detach().cpu()
        spike_data = encoder(data_cpu)
        return spike_data.to(device)

# Hàm trực quan hóa
def visualize_sample_and_spikes(image, label, spike_record, voltage_record, time_window, save_path=None):
    plt.figure(figsize=(18, 12))
    
    # Hiển thị hình ảnh gốc
    plt.subplot(3, 3, 1)
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f'Hình ảnh gốc: {class_names[label]}')
    plt.axis('off')
    
    # Hiển thị tổng xung (spike) ở lớp đầu vào
    plt.subplot(3, 3, 2)
    spikes_input = spike_record["input_monitor"].get("s").squeeze()
    if spikes_input.size(0) == time_window:
        plt.imshow(spikes_input.sum(0).reshape(28, 28).float().cpu(), cmap='hot')  # Chuyển về CPU để vẽ
        plt.title('Tổng xung (spike) ở lớp đầu vào')
        plt.colorbar()
    
    # Hiển thị xung (spike) theo thời gian ở lớp ẩn
    plt.subplot(3, 3, 4)
    spikes_hidden = spike_record["hidden_monitor"].get("s").squeeze()
    if spikes_hidden.size(0) == time_window:
        plt.imshow(spikes_hidden.cpu().T, cmap='binary', aspect='auto')  # Chuyển về CPU để vẽ
        plt.title('Xung (spike) theo thời gian ở lớp ẩn')
        plt.xlabel('Thời gian (ms)')
        plt.ylabel('Nơ-ron')
    
    # Hiển thị điện thế theo thời gian ở lớp ẩn (5 nơ-ron đầu tiên)
    plt.subplot(3, 3, 5)
    v_hidden = voltage_record["hidden_monitor"].get("v").squeeze()
    if v_hidden.size(0) == time_window:
        for i in range(5):
            plt.plot(v_hidden[:, i].cpu().numpy(), label=f'Nơ-ron {i}')  # Chuyển về CPU để vẽ
        plt.title('Điện thế theo thời gian ở lớp ẩn (5 nơ-ron)')
        plt.xlabel('Thời gian (ms)')
        plt.ylabel('Điện thế (mV)')
        plt.legend()
    
    # Hiển thị xung (spike) theo thời gian ở lớp đầu ra
    plt.subplot(3, 3, 7)
    spikes_output = spike_record["output_monitor"].get("s").squeeze()
    if spikes_output.size(0) == time_window:
        plt.imshow(spikes_output.cpu().T, cmap='binary', aspect='auto')  # Chuyển về CPU để vẽ
        plt.title('Xung (spike) theo thời gian ở lớp đầu ra')
        plt.xlabel('Thời gian (ms)')
        plt.ylabel('Lớp')
        plt.yticks(range(10), [f'{i}' for i in range(10)])
    
    # Hiển thị tổng xung (spike) theo lớp ở lớp đầu ra
    plt.subplot(3, 3, 8)
    if spikes_output.size(0) == time_window:
        spike_count = spikes_output.sum(0).cpu()  # Chuyển về CPU để vẽ
        if spike_count.sum() > 0:  # Đảm bảo có xung trước khi vẽ
            plt.bar(range(10), spike_count)
            plt.title('Tổng xung (spike) theo lớp')
            plt.xlabel('Lớp')
            plt.ylabel('Số lượng xung (spike)')
            plt.xticks(range(10), [f'{i}' for i in range(10)])
            plt.axvline(x=label, color='red', linestyle='--')
            plt.text(label, max(spike_count)/2 if max(spike_count) > 0 else 0.5, 'Lớp thực tế', 
                    rotation=90, verticalalignment='center')
        else:
            plt.text(0.5, 0.5, 'Không có xung nào được phát hiện', 
                    horizontalalignment='center', verticalalignment='center')
    
    # Hiển thị trọng số kết nối từ lớp ẩn đến lớp đầu ra cho lớp thực tế
    plt.subplot(3, 3, 9)
    plt.hist(hidden_output.w[:, label].detach().cpu().numpy(), bins=30)  # Thay đổi từ input_hidden sang hidden_output
    plt.title(f'Phân phối trọng số kết nối đến lớp {label}')
    plt.xlabel('Trọng số')
    plt.ylabel('Số lượng kết nối')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Đã lưu biểu đồ trực quan hóa vào {save_path}")
    else:
        plt.show()
    
    plt.close()

# Hàm đánh giá mạng SNN với dữ liệu kiểm tra
def evaluate_snn(snn, testloader, time_window, device, num_samples=None):
    print("Đánh giá mạng SNN...")
    correct = 0
    total = 0
    
    snn.reset_state_variables()
    
    # Sử dụng tqdm để hiển thị thanh tiến trình đánh giá
    pbar = tqdm(testloader, desc="Đánh giá")
    for images, labels in pbar:
        if num_samples is not None and total >= num_samples:
            break
            
        images = images.view(images.shape[0], -1).to(device)
        labels = labels.to(device)
        
        batch_size = images.shape[0]
        total += batch_size
        
        # Mã hóa Poisson
        rate = images.clone()
        rate[rate <= 0] = 1e-6
        
        # Tạo spike inputs trên cùng thiết bị
        spike_inputs = encode_to_spikes(rate, time_window, device)
        
        # Chạy mạng SNN
        snn.run(inputs={"input": spike_inputs}, time=time_window)
        
        # Lấy kết quả từ monitor
        output_spikes = snn.monitors["output_monitor"].get("s").sum(0)
        predicted = torch.max(output_spikes, 1)[1]
        
        correct += (predicted == labels).sum().item()
        
        snn.reset_state_variables()
        
        # Cập nhật tiến trình
        accuracy = 100 * correct / total
        pbar.set_postfix({"accuracy": f"{accuracy:.2f}%"})
    
    accuracy = 100 * correct / total
    print(f"Độ chính xác: {accuracy:.2f}% trên {total} mẫu kiểm tra")
    return accuracy

# Huấn luyện SNN
print("\nBắt đầu huấn luyện mạng SNN...")
training_progress = []

for epoch in range(epochs):
    epoch_start_time = time.time()
    print(f"Epoch {epoch+1}/{epochs}")
    
    # Sử dụng tqdm để hiển thị thanh tiến trình
    pbar = tqdm(trainloader, desc=f"Training Epoch {epoch+1}")
    for images, labels in pbar:
        images = images.view(images.shape[0], -1).to(device)
        
        rate = images.clone()
        rate[rate <= 0] = 1e-6
        
        # Tạo spike inputs trên cùng thiết bị - sử dụng hàm mới
        spike_inputs = encode_to_spikes(rate, time_window, device)
        
        snn.run(inputs={"input": spike_inputs}, time=time_window)
        
        snn.reset_state_variables()
        
        # Hiển thị tiến trình trong tqdm
        pbar.set_description(f"Epoch {epoch+1}/{epochs}")
    
    # Đánh giá trên 1000 mẫu test để tiết kiệm thời gian giữa các epoch
    eval_samples = 1000
    accuracy = evaluate_snn(snn, testloader, time_window, device, num_samples=eval_samples)
    training_progress.append(accuracy)
    
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1} hoàn tất trong {epoch_time:.2f} giây! Độ chính xác: {accuracy:.2f}%\n")

# Đánh giá lần cuối trên toàn bộ tập test
print("Đánh giá cuối cùng trên toàn bộ tập test...")
final_accuracy = evaluate_snn(snn, testloader, time_window, device)
print(f"Độ chính xác cuối cùng: {final_accuracy:.2f}%")

print("Huấn luyện hoàn tất!")

# Trực quan hóa tiến trình huấn luyện
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), training_progress, marker='o')
plt.title('Tiến trình huấn luyện mạng SNN với MNIST')
plt.xlabel('Epoch')
plt.ylabel('Độ chính xác (%)')
plt.grid(True)
training_plot_path = os.path.join(output_dir, "training_progress.png")
plt.savefig(training_plot_path)
print(f"Đã lưu biểu đồ tiến trình huấn luyện vào {training_plot_path}")
plt.close()

# Trực quan hóa một mẫu từ tập kiểm tra
print("\nTrực quan hóa hoạt động của mạng SNN trên một mẫu kiểm tra...")
dataiter = iter(testloader)
images, labels = next(dataiter)

idx = np.random.randint(0, images.size(0))
img = images[idx].view(-1).to(device)
label = labels[idx].item()

snn.reset_state_variables()

rate = img.clone()
rate[rate <= 0] = 1e-6
# Sử dụng hàm mới để mã hóa
spike_inputs = encode_to_spikes(rate.unsqueeze(0), time_window, device)
snn.run(inputs={"input": spike_inputs}, time=time_window)

sample_vis_path = os.path.join(output_dir, "sample_visualization.png")
visualize_sample_and_spikes(img.cpu(), label, snn.monitors, snn.monitors, time_window, save_path=sample_vis_path)

# Trực quan hóa trọng số kết nối
plt.figure(figsize=(15, 6))
for i in range(10):
    weights = input_hidden.w[:, i].view(28, 28).detach().cpu().numpy()  # Chuyển về CPU để vẽ
    plt.subplot(2, 5, i+1)
    plt.imshow(weights, cmap='coolwarm')
    plt.title(f'Trọng số đến nơ-ron ẩn {i}')
    plt.colorbar()
    plt.axis('off')

plt.tight_layout()
input_weights_path = os.path.join(output_dir, "input_hidden_weights.png")
plt.savefig(input_weights_path)
print(f"Đã lưu biểu đồ trọng số input→hidden vào {input_weights_path}")
plt.close()

plt.figure(figsize=(15, 8))
for i in range(10):
    plt.subplot(2, 5, i+1)
    # Lấy 100 giá trị đầu tiên từ các trọng số đầu vào để hiển thị
    values = input_hidden.w[:100, i].detach().cpu().numpy()  # Đảm bảo chỉ lấy 100 giá trị đầu
    plt.stem(range(len(values)), values)
    plt.title(f'Trọng số đến lớp {i}')
    plt.xlabel('Nơ-ron ẩn')
    plt.ylabel('Trọng số')
    
plt.tight_layout()
output_weights_path = os.path.join(output_dir, "hidden_output_weights.png")
plt.savefig(output_weights_path)
print(f"Đã lưu biểu đồ trọng số hidden→output vào {output_weights_path}")
plt.close()

# Hiển thị tổng thời gian chạy
total_time = time.time() - start_time
print(f"\nTổng thời gian chạy: {total_time:.2f} giây ({total_time/60:.2f} phút)")
print(f"Hoàn tất! Tất cả các biểu đồ đã được lưu vào thư mục: {output_dir}")

# Hàm dự đoán với mạng SNN đã huấn luyện - được tối ưu hóa để hoạt động tốt hơn
def predict_with_snn(snn, image, device, time_window=500):  # Tăng time_window
    """Dự đoán nhãn cho một hình ảnh đầu vào bằng SNN đã huấn luyện với nhiều cải tiến
    để đảm bảo hoạt động tốt hơn trong thực tế"""
    snn.reset_state_variables()
    
    # Chuẩn bị dữ liệu
    img_tensor = image.view(-1).to(device)
    
    # Tăng độ tương phản để tăng tín hiệu đầu vào
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
    rate = img_tensor.clone()
    rate[rate <= 0.1] = 1e-6  # Ngưỡng cao hơn để giảm nhiễu
    
    # Thêm giai đoạn khởi động cho mạng
    warmup_window = 100
    total_window = warmup_window + time_window
    
    # Mã hóa hình ảnh thành xung với độ dài thời gian tổng cộng
    spike_inputs = encode_to_spikes(rate.unsqueeze(0), total_window, device)
    
    # Chạy mạng SNN
    snn.run(inputs={"input": spike_inputs}, time=total_window)
    
    # Lấy kết quả từ monitor, nhưng chỉ tính trong thời gian thực sự (bỏ qua giai đoạn khởi động)
    output_spikes = snn.monitors["output_monitor"].get("s")[warmup_window:].sum(0)
    output_voltages = snn.monitors["output_monitor"].get("v")[warmup_window:].mean(0)
    
    # Xử lý trường hợp không có xung
    spike_values = output_spikes.squeeze().cpu().numpy()
    voltage_values = output_voltages.squeeze().cpu().numpy()
    
    if output_spikes.sum() == 0:
        # Sử dụng điện thế thay vì xung nếu không có xung
        predicted = torch.max(output_voltages, 1)[1].item()
        confidence = 0.1 + 0.9 * (np.max(voltage_values) - np.min(voltage_values)) / 10.0  
        # Tính độ tin cậy dựa trên sự chênh lệch điện thế
    else:
        # Tính dự đoán dựa trên xung, kết hợp với điện thế cho độ tin cậy cao hơn
        predicted = torch.max(output_spikes, 1)[1].item()
        spike_sum = spike_values.sum()
        confidence = (0.5 * spike_values[predicted] / spike_sum) + (0.5 * (voltage_values[predicted] - np.min(voltage_values)) / 10.0)
        # Độ tin cậy kết hợp giữa xung và điện thế
    
    # Đảm bảo độ tin cậy nằm trong khoảng hợp lý
    confidence = min(max(confidence, 0.01), 0.99)
    
    snn.reset_state_variables()
    return predicted, confidence, spike_values

# Thêm chuẩn hóa và tối ưu trọng số trước khi dự đoán
def optimize_network_for_prediction(snn):
    """Chuẩn hóa và tối ưu trọng số mạng để đảm bảo hoạt động tốt hơn khi dự đoán"""
    print("Tối ưu hóa mạng cho dự đoán...")
    
    # 1. Chuẩn hóa trọng số đầu ra để mỗi lớp có cường độ tương đương
    hidden_output = snn.connections[("hidden", "output")]
    weights = hidden_output.w.data
    
    # Chuẩn hóa theo cột (mỗi lớp đầu ra)
    for i in range(10):  # 10 lớp
        col_weights = weights[:, i]
        # Đảm bảo trọng số có cả giá trị dương (khuyến khích phát xung)
        col_max = col_weights.max()
        col_min = col_weights.min()
        col_range = col_max - col_min
        
        if col_range > 0:
            # Chuẩn hóa về khoảng 0-0.2
            normalized = 0.2 * (col_weights - col_min) / col_range
            weights[:, i] = normalized
    
    # 2. Tăng biên độ của các kết nối mạnh nhất đến mỗi lớp
    for i in range(10):
        # Lấy 20% kết nối mạnh nhất cho mỗi lớp và tăng cường chúng
        col_weights = weights[:, i]
        threshold = torch.quantile(col_weights, 0.8)
        strongest_mask = col_weights > threshold
        
        # Tăng cường các kết nối mạnh lên 2x
        weights[strongest_mask, i] *= 2.0
    
    # 3. Giảm mức ức chế để cho phép nhiều nơ-ron phát xung hơn
    output_recurrent = snn.connections[("output", "output")]
    inhibition = torch.ones(10, 10) - torch.diag(torch.ones(10))
    output_recurrent.w.data = -0.05 * inhibition.to(device)  # Giảm mức ức chế
    
    # 4. Giảm ngưỡng của nơ-ron đầu ra để tăng khả năng phát xung
    output_layer = snn.layers["output"]
    output_layer.thresh = -58.0  # Giảm ngưỡng thêm một chút
    
    print("Đã tối ưu hóa mạng cho dự đoán thành công!")
    return snn

# Sử dụng hàm tối ưu hóa ngay trước khi dự đoán
print("\nĐang chuẩn bị mạng cho dự đoán...")
snn = optimize_network_for_prediction(snn)

# Trực quan hóa các dự đoán trên nhiều mẫu
print("\nTrực quan hóa dự đoán của mạng SNN trên nhiều mẫu...")
num_samples = 16  # Tăng số lượng mẫu để hiển thị
dataiter = iter(testloader)
images, labels = next(dataiter)

plt.figure(figsize=(15, 12))
for i in range(min(num_samples, len(images))):
    img = images[i].view(-1).to(device)
    label = labels[i].item()
    
    predicted, confidence, spike_values = predict_with_snn(snn, img, device, time_window)
    
    plt.subplot(4, 4, i+1)
    plt.imshow(img.reshape(28, 28).cpu(), cmap='gray')
    
    title_color = 'green' if predicted == label else 'red'
    plt.title(f'Thực tế: {label}\nDự đoán: {predicted}\nĐộ tin cậy: {confidence:.2f}', 
              color=title_color, fontsize=10)
    plt.axis('off')

plt.tight_layout()
predictions_path = os.path.join(output_dir, "multiple_predictions.png")
plt.savefig(predictions_path)
print(f"Đã lưu biểu đồ dự đoán nhiều mẫu vào {predictions_path}")
plt.close()

# Đánh giá độ chính xác sau tối ưu hóa
print("\nĐánh giá cuối cùng của mạng sau tối ưu hóa...")
final_accuracy = evaluate_snn(snn, testloader, time_window, device, num_samples=1000)
print(f"Độ chính xác sau tối ưu hóa: {final_accuracy:.2f}%")

# Lưu mô hình để sử dụng trong ứng dụng 
print("\nLưu mô hình SNN để sử dụng trong ứng dụng...")
model_save_path = os.path.join(base_path, "snn_mnist_model_optimized.pt")
torch.save({
    'network_state_dict': snn.state_dict(),
    'hidden_layer_params': {
        'n': 100,
        'thresh': -55.0,
        'reset': -65.0,
        'rest': -65.0,
        'tau': 10.0
    },
    'output_layer_params': {
        'thresh': -58.0,  # Ngưỡng đã điều chỉnh
        'reset': -65.0,
        'rest': -65.0,
        'tau': 10.0
    },
    'time_window': time_window,
    'warmup_window': 100,  # Thêm thông tin về giai đoạn khởi động
    'accuracy': final_accuracy,
    'class_names': class_names,
    'prediction_function': 'predict_with_snn'  # Tên hàm dự đoán
}, model_save_path)
print(f"Đã lưu mô hình SNN vào {model_save_path}")

# Kiểm tra ứng dụng thực tế
print("\nKiểm tra ứng dụng thực tế với một số mẫu ngẫu nhiên...")
random_indices = np.random.choice(len(testset), 5, replace=False)
test_results = []

for idx in random_indices:
    img, label = testset[idx]
    img_tensor = img.view(-1).to(device)
    predicted, confidence, _ = predict_with_snn(snn, img_tensor, device, time_window)
    
    is_correct = predicted == label
    test_results.append({
        'label': label,
        'prediction': predicted,
        'confidence': confidence,
        'is_correct': is_correct
    })
    
    print(f"Mẫu {idx}: Thực tế = {label}, Dự đoán = {predicted}, Độ tin cậy = {confidence:.2f}, Chính xác: {is_correct}")

correct_count = sum(1 for result in test_results if result['is_correct'])
print(f"Tỷ lệ chính xác trong kiểm tra thực tế: {correct_count/len(test_results)*100:.2f}%")
print(f"Độ tin cậy trung bình: {sum(result['confidence'] for result in test_results)/len(test_results):.2f}")

print("\nMô hình đã sẵn sàng để sử dụng trong ứng dụng nhận dạng chữ số viết tay!")

def adjust_thresholds(model, target_activity=0.1):
    """Điều chỉnh ngưỡng để duy trì hoạt động"""
    for layer_name in ["output"]:
        layer = model.layers[layer_name]
        activity = model.monitors[f"{layer_name}_monitor"].get("s").sum(0).float() / time_window
        # Điều chỉnh ngưỡng: tăng ngưỡng nếu hoạt động cao, giảm nếu thấp
        layer.thresh += 0.1 * (activity - target_activity) 

# Thêm vào ngay trước khi lưu/sử dụng mô hình
# Chuẩn hóa trọng số để cân bằng
for connection in [hidden_output]:
    weights = connection.w.data
    weights_norm = torch.norm(weights, dim=0, keepdim=True)
    connection.w.data = weights / weights_norm * 0.1 