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

# Create output directory for plots
output_dir = "snn_plots_mnist"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Đã tạo thư mục đầu ra: {output_dir}")

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# Tải dữ liệu MNIST
# Tạo thư mục dữ liệu riêng cho bindsnet
data_dir = "./data/bindsnet_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Đã tạo thư mục dữ liệu: {data_dir}")

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Định nghĩa tên lớp cho MNIST (chữ số từ 0-9)
class_names = [str(i) for i in range(10)]

# Tạo mạng SNN với cấu trúc đơn giản
print("Tạo mạng SNN...")
snn = Network(dt=1.0)

# Lớp đầu vào: 784 nơ-ron (28x28 pixel)
input_layer = Input(n=28 * 28)

# Lớp ẩn: 100 nơ-ron LIF
hidden_layer = LIFNodes(n=100, thresh=-52.0, reset=-65.0, rest=-65.0, tau=10.0)

# Lớp đầu ra: 10 nơ-ron LIF (mỗi nơ-ron tương ứng với một chữ số)
output_layer = LIFNodes(n=10, thresh=-52.0, reset=-65.0, rest=-65.0, tau=10.0)

# Thêm các lớp vào mạng
snn.add_layer(input_layer, name="input")
snn.add_layer(hidden_layer, name="hidden")
snn.add_layer(output_layer, name="output")

# Khởi tạo trọng số kết nối giữa các lớp với cơ chế học STDP
print("Khởi tạo kết nối giữa các lớp...")
input_hidden_weights = 0.05 * torch.rand(784, 100)
input_hidden = Connection(
    source=input_layer,
    target=hidden_layer,
    w=input_hidden_weights,
    update_rule=PostPre,
    nu=(1e-4, 1e-2),
    wmin=0.0,
    wmax=1.0,
)
snn.add_connection(input_hidden, source="input", target="hidden")

hidden_output_weights = 0.05 * torch.rand(100, 10)
hidden_output = Connection(
    source=hidden_layer,
    target=output_layer,
    w=hidden_output_weights,
    update_rule=PostPre,
    nu=(1e-4, 1e-2),
    wmin=0.0,
    wmax=1.0,
)
snn.add_connection(hidden_output, source="hidden", target="output")

# Thêm các monitor để theo dõi hoạt động của mạng
print("Thiết lập các monitor theo dõi hoạt động của mạng...")
snn.add_monitor(Monitor(input_layer, state_vars=["s"]), name="input_monitor")
snn.add_monitor(Monitor(hidden_layer, state_vars=["s", "v"]), name="hidden_monitor")
snn.add_monitor(Monitor(output_layer, state_vars=["s", "v"]), name="output_monitor")

# Bộ mã hóa Poisson
time_window = 50  # Thời gian mô phỏng (ms)
encoder = PoissonEncoder(time=time_window, dt=1.0)

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
        plt.imshow(spikes_input.sum(0).reshape(28, 28).float(), cmap='hot')
        plt.title('Tổng xung (spike) ở lớp đầu vào')
        plt.colorbar()
    
    # Hiển thị xung (spike) theo thời gian ở lớp ẩn
    plt.subplot(3, 3, 4)
    spikes_hidden = spike_record["hidden_monitor"].get("s").squeeze()
    if spikes_hidden.size(0) == time_window:
        plt.imshow(spikes_hidden.T, cmap='binary', aspect='auto')
        plt.title('Xung (spike) theo thời gian ở lớp ẩn')
        plt.xlabel('Thời gian (ms)')
        plt.ylabel('Nơ-ron')
    
    # Hiển thị điện thế theo thời gian ở lớp ẩn (5 nơ-ron đầu tiên)
    plt.subplot(3, 3, 5)
    v_hidden = voltage_record["hidden_monitor"].get("v").squeeze()
    if v_hidden.size(0) == time_window:
        for i in range(5):
            plt.plot(v_hidden[:, i], label=f'Nơ-ron {i}')
        plt.title('Điện thế theo thời gian ở lớp ẩn (5 nơ-ron)')
        plt.xlabel('Thời gian (ms)')
        plt.ylabel('Điện thế (mV)')
        plt.legend()
    
    # Hiển thị xung (spike) theo thời gian ở lớp đầu ra
    plt.subplot(3, 3, 7)
    spikes_output = spike_record["output_monitor"].get("s").squeeze()
    if spikes_output.size(0) == time_window:
        plt.imshow(spikes_output.T, cmap='binary', aspect='auto')
        plt.title('Xung (spike) theo thời gian ở lớp đầu ra')
        plt.xlabel('Thời gian (ms)')
        plt.ylabel('Lớp')
        plt.yticks(range(10), [f'{i}' for i in range(10)])
    
    # Hiển thị tổng xung (spike) theo lớp ở lớp đầu ra
    plt.subplot(3, 3, 8)
    if spikes_output.size(0) == time_window:
        spike_count = spikes_output.sum(0)
        plt.bar(range(10), spike_count)
        plt.title('Tổng xung (spike) theo lớp')
        plt.xlabel('Lớp')
        plt.ylabel('Số lượng xung (spike)')
        plt.xticks(range(10), [f'{i}' for i in range(10)])
        plt.axvline(x=label, color='red', linestyle='--')
        plt.text(label, max(spike_count)/2, 'Lớp thực tế', 
                 rotation=90, verticalalignment='center')
    
    # Hiển thị trọng số kết nối từ lớp ẩn đến lớp đầu ra cho lớp thực tế
    plt.subplot(3, 3, 9)
    plt.hist(hidden_output_weights[:, label].numpy(), bins=30)
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
def evaluate_snn(snn, testloader, encoder, time_window, device, num_samples=100):
    print("Đánh giá mạng SNN...")
    correct = 0
    total = 0
    
    snn.reset_state_variables()
    
    for images, labels in testloader:
        if total >= num_samples:
            break
            
        images = images.view(images.shape[0], -1).to(device)
        labels = labels.to(device)
        
        batch_size = images.shape[0]
        total += batch_size
        
        # Mã hóa Poisson
        rate = images.clone()
        rate[rate <= 0] = 1e-6
        spike_inputs = encoder(rate)
        
        # Chạy mạng SNN
        snn.run(inputs={"input": spike_inputs}, time=time_window)
        
        # Lấy kết quả từ monitor
        output_spikes = snn.monitors["output_monitor"].get("s").sum(0)
        predicted = torch.max(output_spikes, 1)[1]
        
        correct += (predicted == labels).sum().item()
        
        snn.reset_state_variables()
    
    accuracy = 100 * correct / total
    print(f"Độ chính xác: {accuracy:.2f}% trên {total} mẫu kiểm tra")
    return accuracy

# Huấn luyện SNN
print("\nBắt đầu huấn luyện mạng SNN...")
epochs = 5  # Tăng số epochs lên 5 giống như PySNN
training_progress = []

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    # Sử dụng tqdm để hiển thị thanh tiến trình
    pbar = tqdm(trainloader, desc=f"Training Epoch {epoch+1}")
    for images, labels in pbar:
        images = images.view(images.shape[0], -1).to(device)
        
        rate = images.clone()
        rate[rate <= 0] = 1e-6
        spike_inputs = encoder(rate)
        
        snn.run(inputs={"input": spike_inputs}, time=time_window)
        
        snn.reset_state_variables()
        
        # Hiển thị tiến trình trong tqdm
        pbar.set_description(f"Epoch {epoch+1}/{epochs}")
    
    accuracy = evaluate_snn(snn, testloader, encoder, time_window, device, num_samples=500)  # Tăng số mẫu đánh giá
    training_progress.append(accuracy)
    
    print(f"Epoch {epoch+1} hoàn tất! Độ chính xác: {accuracy:.2f}%\n")

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
spike_inputs = encoder(rate.unsqueeze(0))
snn.run(inputs={"input": spike_inputs}, time=time_window)

sample_vis_path = os.path.join(output_dir, "sample_visualization.png")
visualize_sample_and_spikes(img.cpu(), label, snn.monitors, snn.monitors, time_window, save_path=sample_vis_path)

# Trực quan hóa trọng số kết nối
plt.figure(figsize=(15, 6))
for i in range(10):
    weights = input_hidden.w[:, i].reshape(28, 28)
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
    plt.stem(range(100), hidden_output.w[:, i])
    plt.title(f'Trọng số đến lớp {i}')
    plt.xlabel('Nơ-ron ẩn')
    plt.ylabel('Trọng số')
    
plt.tight_layout()
output_weights_path = os.path.join(output_dir, "hidden_output_weights.png")
plt.savefig(output_weights_path)
print(f"Đã lưu biểu đồ trọng số hidden→output vào {output_weights_path}")
plt.close()

print(f"\nHoàn tất! Tất cả các biểu đồ đã được lưu vào thư mục: {output_dir}")