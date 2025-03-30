import nengo
import nengo_dl
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# Tạo thư mục đầu ra cho các biểu đồ
output_dir = "snn_plots_nengo"
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 
    print(f"Đã tạo thư mục đầu ra: {output_dir}")

# Tạo thư mục dữ liệu riêng cho nengo
data_dir = "./data/nengo_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Đã tạo thư mục dữ liệu: {data_dir}")

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# Tải dữ liệu FashionMNIST
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Định nghĩa tên lớp cho FashionMNIST
class_names = ['Áo thun/T-shirt', 'Quần dài/Trouser', 'Áo len/Pullover', 'Váy/Dress', 
               'Áo khoác/Coat', 'Sandal', 'Áo sơ mi/Shirt', 'Giày thể thao/Sneaker', 
               'Túi xách/Bag', 'Bốt cổ cao/Ankle boot']

# Chuyển dữ liệu FashionMNIST thành định dạng cho Nengo
print("Chuẩn bị dữ liệu cho Nengo...")
def prepare_nengo_data(loader, num_samples=100):
    images_list, labels_list = [], []
    for i, (images, labels) in enumerate(loader):
        if i * loader.batch_size >= num_samples:
            break
        images_list.append(images.numpy().reshape(-1, 28*28))  # Chuyển thành vector 784 chiều
        labels_list.append(labels.numpy())
    return np.vstack(images_list), np.hstack(labels_list)

train_images, train_labels = prepare_nengo_data(trainloader, num_samples=640)  # 10 batch
test_images, test_labels = prepare_nengo_data(testloader, num_samples=100)

# Xây dựng mạng SNN với Nengo
print("Tạo mạng SNN với Nengo...")
time_window = 0.05  # Thời gian mô phỏng (giây)

# Định nghĩa hàm tạo mạng để dễ truyền input_signal
def create_network(input_signal):
    with nengo.Network() as model:
        # Lớp đầu vào
        input_node = nengo.Node(lambda t: input_signal[int(t / time_window * len(input_signal)) % len(input_signal)]
                               if t < time_window else 0, size_out=28*28)
        
        # Lớp ẩn: 100 nơ-ron LIF
        hidden = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.LIF(tau_rc=0.01))
        nengo.Connection(input_node, hidden.neurons, transform=0.05 * np.random.rand(100, 28*28))
        
        # Lớp đầu ra: 10 nơ-ron LIF (mỗi nơ-ron ứng với một lớp)
        output = nengo.Ensemble(n_neurons=10, dimensions=1, neuron_type=nengo.LIF(tau_rc=0.01))
        nengo.Connection(hidden.neurons, output.neurons, transform=0.05 * np.random.rand(10, 100))
        
        # Probes để ghi lại hoạt động
        input_probe = nengo.Probe(input_node, synapse=None)
        hidden_spikes_probe = nengo.Probe(hidden.neurons, 'spikes')
        hidden_voltage_probe = nengo.Probe(hidden.neurons, 'voltage')
        output_spikes_probe = nengo.Probe(output.neurons, 'spikes')
        output_voltage_probe = nengo.Probe(output.neurons, 'voltage')
    
    return model

# Hàm trực quan hóa
def visualize_sample_and_spikes(image, label, sim, save_path=None):
    plt.figure(figsize=(18, 12))
    
    # Hiển thị hình ảnh gốc
    plt.subplot(3, 3, 1)
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f'Hình ảnh gốc: {class_names[label]}')
    plt.axis('off')
    
    # Hiển thị xung ở lớp ẩn
    plt.subplot(3, 3, 4)
    hidden_spikes = sim.data[hidden_spikes_probe]
    plt.imshow(hidden_spikes.T, cmap='binary', aspect='auto')
    plt.title('Xung (spike) theo thời gian ở lớp ẩn')
    plt.xlabel('Thời gian (ms)')
    plt.ylabel('Nơ-ron')
    
    # Hiển thị điện thế ở lớp ẩn (5 nơ-ron đầu tiên)
    plt.subplot(3, 3, 5)
    hidden_voltages = sim.data[hidden_voltage_probe]
    for i in range(min(5, hidden_voltages.shape[1])):
        plt.plot(sim.trange() * 1000, hidden_voltages[:, i], label=f'Nơ-ron {i}')
    plt.title('Điện thế theo thời gian ở lớp ẩn (5 nơ-ron)')
    plt.xlabel('Thời gian (ms)')
    plt.ylabel('Điện thế')
    plt.legend()
    
    # Hiển thị xung ở lớp đầu ra
    plt.subplot(3, 3, 7)
    output_spikes = sim.data[output_spikes_probe]
    plt.imshow(output_spikes.T, cmap='binary', aspect='auto')
    plt.title('Xung (spike) theo thời gian ở lớp đầu ra')
    plt.xlabel('Thời gian (ms)')
    plt.ylabel('Lớp')
    plt.yticks(range(10), [f'{i}: {class_names[i]}' for i in range(10)])
    
    # Hiển thị tổng xung ở lớp đầu ra
    plt.subplot(3, 3, 8)
    spike_count = output_spikes.sum(axis=0)
    plt.bar(range(10), spike_count)
    plt.title('Tổng xung (spike) theo lớp')
    plt.xlabel('Lớp')
    plt.ylabel('Số lượng xung (spike)')
    plt.xticks(range(10), [f'{i}' for i in range(10)])
    plt.axvline(x=label, color='red', linestyle='--')
    plt.text(label, max(spike_count)/2, 'Lớp thực tế', rotation=90, verticalalignment='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Đã lưu biểu đồ trực quan hóa vào {save_path}")
    else:
        plt.show()
    plt.close()

# Đánh giá mạng
def evaluate_snn(test_images, test_labels, num_samples=100):
    print("Đánh giá mạng SNN...")
    correct = 0
    total = 0
    
    for i in range(min(num_samples, len(test_images))):
        model = create_network(test_images[i])
        with nengo.Simulator(model, dt=0.001) as sim:
            sim.run(time_window)
            output_spikes = sim.data[output_spikes_probe]
            predicted = np.argmax(output_spikes.sum(axis=0))
            correct += (predicted == test_labels[i])
            total += 1
    
    accuracy = 100 * correct / total
    print(f"Độ chính xác: {accuracy:.2f}% trên {total} mẫu kiểm tra")
    return accuracy

# Huấn luyện và đánh giá
print("\nBắt đầu mô phỏng mạng SNN...")
epochs = 2
training_progress = []

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    batch_count = 0
    
    for i, (images, labels) in enumerate(trainloader):
        if batch_count >= 10:
            break
        print(f"  Đang xử lý batch {batch_count+1}/10...")
        
        input_signal = images.numpy().reshape(-1, 28*28)[0]  # Lấy mẫu đầu tiên trong batch
        model = create_network(input_signal)
        with nengo.Simulator(model, dt=0.001) as sim:
            sim.run(time_window)
        
        batch_count += 1
    
    accuracy = evaluate_snn(test_images, test_labels)
    training_progress.append(accuracy)
    print(f"Epoch {epoch+1} hoàn tất! Độ chính xác: {accuracy:.2f}%\n")

print("Mô phỏng hoàn tất!")

# Trực quan hóa tiến trình huấn luyện
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), training_progress, marker='o')
plt.title('Tiến trình huấn luyện mạng SNN')
plt.xlabel('Epoch')
plt.ylabel('Độ chính xác (%)')
plt.grid(True)
training_plot_path = os.path.join(output_dir, "training_progress.png")
plt.savefig(training_plot_path)
print(f"Đã lưu biểu đồ tiến trình huấn luyện vào {training_plot_path}")
plt.close()

# Trực quan hóa một mẫu kiểm tra
print("\nTrực quan hóa hoạt động của mạng SNN trên một mẫu kiểm tra...")
idx = np.random.randint(0, len(test_images))
img = test_images[idx]
label = test_labels[idx]

model = create_network(img)
with nengo.Simulator(model, dt=0.001) as sim:
    sim.run(time_window)
    sample_vis_path = os.path.join(output_dir, "sample_visualization.png")
    visualize_sample_and_spikes(img, label, sim, save_path=sample_vis_path)

print(f"\nHoàn tất! Tất cả các biểu đồ đã được lưu vào thư mục: {output_dir}")