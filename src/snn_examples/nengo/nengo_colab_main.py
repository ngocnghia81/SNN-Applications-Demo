"""
SNN FashionMNIST Training with Nengo cơ bản trên Google Colab - Phiên bản đơn giản
"""

# Apply numpy monkey patch first to fix the product function
import numpy as np
if not hasattr(np, 'product'):
    print("Applying monkey patch for NumPy product function...")
    np.product = np.prod

# Now import Nengo libraries
import nengo
import nengo_dl
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from google.colab import drive

print(f"NumPy version: {np.__version__}")
print(f"Nengo version: {nengo.__version__}")
print(f"Nengo-DL version: {nengo_dl.__version__}")
print(f"TensorFlow version: {tf.__version__}")

# Kết nối Google Drive để lưu kết quả
drive.mount('/content/drive')
base_path = '/content/drive/MyDrive/SNN_Results'  # Đổi thành đường dẫn trong Drive của bạn
if not os.path.exists(base_path):
    os.makedirs(base_path)
    print(f"Đã tạo thư mục lưu kết quả: {base_path}")

# Tạo thư mục để lưu hình ảnh
output_dir = os.path.join(base_path, "nengo_plots_fashionmnist")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Đã tạo thư mục đầu ra: {output_dir}")

# Thời gian bắt đầu
start_time = time.time()

# Tạo thư mục dữ liệu riêng cho nengo
data_dir = "./data/nengo_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Đã tạo thư mục dữ liệu: {data_dir}")

# Cấu hình thiết bị
device = tf.device("cuda" if tf.config.experimental.list_physical_devices('GPU') else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# 1. Tải dữ liệu FashionMNIST
print("Sử dụng dữ liệu FashionMNIST")

# Tham số tối ưu hóa - giảm số lượng để chạy nhanh hơn
batch_size = 64    # Kích thước batch
time_window = 0.1  # Thời gian mô phỏng (giây)
num_epochs = 1     # Giảm số epochs xuống còn 1
num_train_batches = 20  # Giảm số batch train
num_test_samples = 200  # Giảm số mẫu test

# Load MNIST dataset
print("Loading MNIST dataset...")
(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.mnist.load_data())

# Flatten images
train_images = train_images.reshape((train_images.shape[0], -1))
test_images = test_images.reshape((test_images.shape[0], -1))

# Định nghĩa tên lớp cho FashionMNIST
class_names = ['Áo thun/T-shirt', 'Quần dài/Trouser', 'Áo len/Pullover', 'Váy/Dress', 
               'Áo khoác/Coat', 'Sandal', 'Áo sơ mi/Shirt', 'Giày thể thao/Sneaker', 
               'Túi xách/Bag', 'Bốt cổ cao/Ankle boot']

# 2. Trực quan hóa dữ liệu mẫu
def plot_sample_images(loader, output_dir):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].numpy().squeeze()
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{class_names[labels[i]]}')
            ax.axis('off')
    plt.tight_layout()
    
    sample_path = os.path.join(output_dir, "sample_images.png")
    plt.savefig(sample_path)
    print(f"Đã lưu biểu đồ hình ảnh mẫu vào {sample_path}")
    plt.close()

print("Trực quan hóa hình ảnh mẫu từ FashionMNIST:")
plot_sample_images(trainloader, output_dir)

# 3. Chuẩn bị dữ liệu
print("\nChuẩn bị dữ liệu cho Nengo...")
def prepare_nengo_data(loader, num_samples=1000):
    """Chuẩn bị dữ liệu cho Nengo từ PyTorch DataLoader"""
    images_list, labels_list = [], []
    sample_count = 0
    
    for images, labels in tqdm(loader, desc="Chuẩn bị dữ liệu"):
        batch_size = len(images)
        images_list.append(images.numpy().reshape(-1, 28*28))  # Chuyển thành vector 784 chiều
        labels_list.append(labels.numpy())
        sample_count += batch_size
        if sample_count >= num_samples:
            break
            
    return np.vstack(images_list)[:num_samples], np.hstack(labels_list)[:num_samples]

train_images, train_labels = prepare_nengo_data(trainloader, num_samples=batch_size * num_train_batches)
test_images, test_labels = prepare_nengo_data(testloader, num_samples=num_test_samples)

print(f"Đã chuẩn bị {len(train_images)} mẫu huấn luyện và {len(test_images)} mẫu kiểm tra cho Nengo")

# Khai báo biến toàn cục cho probes
hidden_spikes_probe = None
hidden_voltage_probe = None
output_spikes_probe = None
output_voltage_probe = None

# 4. Xây dựng mạng SNN với Nengo cơ bản
print("\nTạo mạng SNN với Nengo cơ bản...")

# Định nghĩa hàm tạo mạng để dễ truyền input_signal
def create_network(input_signal=None):
    global hidden_spikes_probe, hidden_voltage_probe, output_spikes_probe, output_voltage_probe
    
    with nengo.Network(seed=1) as model:
        # Tham số mạng
        n_hidden = 100  # Giảm số lượng nơ-ron ẩn để chạy nhanh hơn
        n_out = 10      # Số lượng đầu ra (10 lớp)
        
        # Chế độ mô phỏng nếu không có đầu vào
        if input_signal is None:
            input_signal = np.zeros(28*28)
            
        # Lớp đầu vào
        input_node = nengo.Node(lambda t: input_signal[int(t / time_window * len(input_signal)) % len(input_signal)]
                              if t < time_window else np.zeros(28*28), size_out=28*28)
        
        # Lớp ẩn: nơ-ron LIF
        hidden = nengo.Ensemble(
            n_neurons=n_hidden,
            dimensions=1, 
            neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002),
            max_rates=nengo.dists.Uniform(200, 400),  # Tỷ lệ spike tối đa
            intercepts=nengo.dists.Uniform(-0.5, 0.5)  # Ngưỡng kích hoạt
        )
        
        # Kết nối từ đầu vào đến lớp ẩn
        weights_ih = 0.1 * (np.random.rand(n_hidden, 28*28) - 0.5)  # Random weights
        nengo.Connection(
            input_node, 
            hidden.neurons,
            transform=weights_ih,
            synapse=nengo.Alpha(tau=0.005)  # Alpha synapse với hằng số thời gian 5ms
        )
        
        # Lớp đầu ra: nơ-ron LIF
        output = nengo.Ensemble(
            n_neurons=n_out,
            dimensions=1, 
            neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002),
            max_rates=nengo.dists.Uniform(200, 400),
            intercepts=nengo.dists.Uniform(-0.5, 0.5)
        )
        
        # Kết nối từ lớp ẩn đến đầu ra
        weights_ho = 0.1 * (np.random.rand(n_out, n_hidden) - 0.5)  # Random weights
        nengo.Connection(
            hidden.neurons, 
            output.neurons,
            transform=weights_ho,
            synapse=nengo.Alpha(tau=0.005)
        )
        
        # Thêm ức chế bên (lateral inhibition) để cải thiện phân loại
        inhib_matrix = -0.1 * np.ones((n_out, n_out))  # Tạo ma trận ức chế
        np.fill_diagonal(inhib_matrix, 0)  # Không ức chế chính nó
        nengo.Connection(
            output.neurons, 
            output.neurons, 
            transform=inhib_matrix,
            synapse=nengo.Alpha(tau=0.005)
        )
        
        # Probes để ghi lại hoạt động
        input_probe = nengo.Probe(input_node, synapse=0.01)
        hidden_spikes_probe = nengo.Probe(hidden.neurons, 'spikes')
        hidden_voltage_probe = nengo.Probe(hidden.neurons, 'voltage', synapse=0.01)
        output_spikes_probe = nengo.Probe(output.neurons, 'spikes')
        output_voltage_probe = nengo.Probe(output.neurons, 'voltage', synapse=0.01)
    
    return model

# 5. Trực quan hóa hoạt động của mạng
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
    plt.text(label, max(spike_count)/2 if len(spike_count) > 0 and max(spike_count) > 0 else 0.5, 
             'Lớp thực tế', rotation=90, verticalalignment='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Đã lưu biểu đồ trực quan hóa vào {save_path}")
    else:
        plt.show()
    plt.close()

# 6. Đánh giá mạng
def evaluate_snn(test_images, test_labels, num_samples=100):
    """Đánh giá hiệu suất của mạng SNN"""
    print(f"Đánh giá mạng SNN trên {num_samples} mẫu...")
    correct = 0
    total = 0
    
    # Sử dụng dt nhỏ hơn cho độ chính xác tốt hơn
    dt = 0.001  # 1ms
    
    pbar = tqdm(range(min(num_samples, len(test_images))), desc="Đánh giá")
    for i in pbar:
        model = create_network(test_images[i])
        with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
            sim.run(time_window)
            output_spikes = sim.data[output_spikes_probe]
            predicted = np.argmax(output_spikes.sum(axis=0))
            correct += (predicted == test_labels[i])
            total += 1
            
            # Cập nhật thanh tiến trình
            accuracy = 100 * correct / total
            pbar.set_postfix({"accuracy": f"{accuracy:.2f}%"})
    
    accuracy = 100 * correct / total
    print(f"Độ chính xác: {accuracy:.2f}% trên {total} mẫu kiểm tra")
    return accuracy

# 7. Mô phỏng và đánh giá
print("\nBắt đầu mô phỏng mạng SNN...")
training_start_time = time.time()
training_progress = []

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    print(f"Epoch {epoch+1}/{num_epochs}")
    batch_count = 0
    
    # Batch processing
    pbar = tqdm(range(0, min(num_train_batches * batch_size, len(train_images)), batch_size),
               desc=f"Mô phỏng Epoch {epoch+1}")
    
    for i in pbar:
        batch_end = min(i + batch_size, len(train_images))
        batch_imgs = train_images[i:batch_end]
        batch_labels = train_labels[i:batch_end]
        
        # Xử lý và mô phỏng trên mẫu đầu tiên trong batch
        input_signal = batch_imgs[0]
        model = create_network(input_signal)
        with nengo.Simulator(model, dt=0.001, progress_bar=False) as sim:
            sim.run(time_window)
        
        batch_count += 1
        pbar.set_postfix({"batch": f"{batch_count}/{num_train_batches}"})
        
        if batch_count >= num_train_batches:
            break
    
    # Đánh giá sau mỗi epoch
    eval_samples = min(num_test_samples, len(test_images))
    accuracy = evaluate_snn(test_images, test_labels, num_samples=eval_samples)
    training_progress.append(accuracy)
    
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1} hoàn tất trong {epoch_time:.2f} giây! Độ chính xác: {accuracy:.2f}%\n")

training_time = time.time() - training_start_time
print(f"Mô phỏng hoàn tất trong {training_time:.2f} giây ({training_time/60:.2f} phút)!")

# 8. Trực quan hóa tiến trình huấn luyện
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), training_progress, marker='o')
plt.title('Tiến trình mô phỏng mạng SNN với FashionMNIST')
plt.xlabel('Epoch')
plt.ylabel('Độ chính xác (%)')
plt.grid(True)
plt.xticks(range(1, num_epochs+1))
training_plot_path = os.path.join(output_dir, "training_progress.png")
plt.savefig(training_plot_path)
print(f"Đã lưu biểu đồ tiến trình mô phỏng vào {training_plot_path}")
plt.close()

# 9. Trực quan hóa một số mẫu kiểm tra
print("\nTrực quan hóa hoạt động của mạng SNN trên một mẫu kiểm tra...")
idx = np.random.randint(0, len(test_images))
img = test_images[idx]
label = test_labels[idx]

model = create_network(img)
with nengo.Simulator(model, dt=0.001, progress_bar=False) as sim:
    sim.run(time_window)
    sample_vis_path = os.path.join(output_dir, "sample_visualization.png")
    visualize_sample_and_spikes(img, label, sim, save_path=sample_vis_path)

# 10. Trực quan hóa dự đoán trên nhiều mẫu
def plot_predictions(test_images, test_labels, output_dir, num_samples=9):
    """Trực quan hóa dự đoán của mạng trên nhiều mẫu"""
    # Chọn ngẫu nhiên các mẫu
    indices = np.random.choice(len(test_images), min(num_samples, len(test_images)), replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < len(indices):
            idx = indices[i]
            img = test_images[idx]
            label = test_labels[idx]
            
            # Mô phỏng mạng
            model = create_network(img)
            with nengo.Simulator(model, dt=0.001, progress_bar=False) as sim:
                sim.run(time_window)
                output_spikes = sim.data[output_spikes_probe]
                predicted = np.argmax(output_spikes.sum(axis=0))
            
            # Hiển thị hình ảnh và dự đoán
            ax.imshow(img.reshape(28, 28), cmap='gray')
            title_color = 'green' if predicted == label else 'red'
            ax.set_title(f"Dự đoán: {predicted}\nThực tế: {label}", color=title_color)
            ax.axis('off')
    
    plt.tight_layout()
    predictions_path = os.path.join(output_dir, "predictions.png")
    plt.savefig(predictions_path)
    print(f"Đã lưu biểu đồ dự đoán vào {predictions_path}")
    plt.close()

print("Trực quan hóa dự đoán trên nhiều mẫu...")
plot_predictions(test_images, test_labels, output_dir)

# Hiển thị tổng thời gian chạy
total_time = time.time() - start_time
print(f"\nTổng thời gian chạy: {total_time:.2f} giây ({total_time/60:.2f} phút)")
print(f"Hoàn tất! Tất cả các biểu đồ đã được lưu vào thư mục: {output_dir}") 