import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pysnn.network import SNNNetwork
from pysnn.connection import Linear
from pysnn.neuron import LIFNeuron
import os
from tqdm import tqdm

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tạo thư mục để lưu hình ảnh
output_dir = "pysnn_plots_mnist"
os.makedirs(output_dir, exist_ok=True)

# Tạo thư mục dữ liệu riêng cho pysnn
data_dir = "./data/pysnn_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Đã tạo thư mục dữ liệu: {data_dir}")

# 1. Tải và tiền xử lý dữ liệu MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 2. Trực quan hóa dữ liệu mẫu và lưu
def plot_sample_images(loader, output_dir):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        img = images[i].numpy().squeeze()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_images.png"))
    plt.close()

print("Sample images from MNIST dataset:")
plot_sample_images(train_loader, output_dir)

# 3. Triển khai STDP thủ công
class CustomSTDP:
    def __init__(self, connection, learning_rate=(0.05, -0.015), a_plus=1.0, a_minus=1.0):
        self.connection = connection
        self.lr_plus, self.lr_minus = learning_rate
        self.a_plus = a_plus
        self.a_minus = a_minus

    def update(self, pre_spikes, post_spikes):
        # Ensure spikes are 2D tensors (batch_size x neurons)
        if len(pre_spikes.shape) == 1:
            pre_spikes = pre_spikes.unsqueeze(0)
        if len(post_spikes.shape) == 1:
            post_spikes = post_spikes.unsqueeze(0)
            
        # Get the number of pre and post neurons
        pre_neurons = pre_spikes.size(-1)
        post_neurons = post_spikes.size(-1)
        
        delta_w = torch.zeros_like(self.connection.weight)
        
        # Calculate STDP updates
        for i in range(pre_neurons):
            pre_spike_times = pre_spikes[..., i].float()
            for j in range(post_neurons):
                post_spike_times = post_spikes[..., j].float()
                
                if pre_spike_times.sum() > 0 and post_spike_times.sum() > 0:
                    delta_t = post_spike_times.sum() - pre_spike_times.sum()
                    if delta_t > 0:
                        delta_w[j, i] += self.lr_plus * self.a_plus * torch.exp(-delta_t)
                    else:
                        delta_w[j, i] += self.lr_minus * self.a_minus * torch.exp(delta_t)
        
        # Update weights with clamping
        self.connection.weight.data += delta_w.clamp(min=-1.0, max=1.0)

# 4. Định nghĩa mạng SNN
class MNIST_SNN(SNNNetwork):
    def __init__(self, input_size=784, hidden_size=256, output_size=10, batch_size=64, dt=1.0):
        super(MNIST_SNN, self).__init__()
        self.batch_size = batch_size
        self.dt = dt
        self.output_size = output_size  # Store output size
        self.conn1 = Linear(input_size, hidden_size, batch_size, dt, delay=1)
        self.neuron1 = LIFNeuron(
            cells_shape=(batch_size,),  # Chỉ cần batch_size cho shape
            thresh=1.0,
            v_rest=0.0,
            alpha_v=0.9,  # Tốc độ rò rỉ điện áp
            alpha_t=0.9,  # Tốc độ rò rỉ ngưỡng
            dt=dt,        # Bước thời gian
            duration_refrac=5.0,  # Thời gian bất hoạt (ms)
            tau_v=0.9,    # Hằng số thời gian điện áp (0.9 for linear decay)
            tau_t=0.9     # Hằng số thời gian ngưỡng (0.9 for linear decay)
        )
        self.conn2 = Linear(hidden_size, output_size, batch_size, dt, delay=1)
        self.neuron2 = LIFNeuron(
            cells_shape=(batch_size,),  # Chỉ cần batch_size cho shape
            thresh=1.0,
            v_rest=0.0,
            alpha_v=0.9,
            alpha_t=0.9,
            dt=dt,
            duration_refrac=5.0,
            tau_v=0.9,    # Hằng số thời gian điện áp (0.9 for linear decay)
            tau_t=0.9     # Hằng số thời gian ngưỡng (0.9 for linear decay)
        )
        self.stdp1 = CustomSTDP(self.conn1)
        self.stdp2 = CustomSTDP(self.conn2)

    def forward(self, x, time_steps=50):
        batch_size = x.size(0)
        spikes_out = torch.zeros(batch_size, self.output_size).to(device)
        pre_spikes = []
        post_spikes = []

        for t in range(time_steps):
            input_spikes = (x > torch.rand(x.size(), device=device)).float()
            # First layer
            input_trace = input_spikes.view(batch_size, 1, -1)  # Reshape for trace
            input_spikes_prop = input_spikes.view(batch_size, 1, -1)  # Reshape for propagation
            hidden_act, hidden_trace = self.conn1(input_spikes_prop, input_trace)
            hidden_act = hidden_act.view(batch_size, -1)  # Reshape for neuron
            hidden_spikes, hidden_trace = self.neuron1(hidden_act.view(batch_size, -1))  # Ensure correct shape
            # Second layer
            hidden_trace = hidden_spikes.view(batch_size, 1, -1)  # Reshape for trace
            hidden_spikes_prop = hidden_spikes.view(batch_size, 1, -1)  # Reshape for propagation
            output_act, output_trace = self.conn2(hidden_spikes_prop, hidden_trace)
            output_act = output_act.view(batch_size, -1)  # Reshape for neuron
            output_spikes, output_trace = self.neuron2(output_act.view(batch_size, -1))  # Ensure correct shape
            
            # Ensure output_spikes has shape (batch_size, output_size)
            output_spikes = output_spikes.view(batch_size, -1)
            spikes_out += output_spikes
            pre_spikes.append(input_spikes)
            post_spikes.append(hidden_spikes)

        pre_spikes = torch.stack(pre_spikes).sum(0)
        post_spikes = torch.stack(post_spikes).sum(0)
        self.stdp1.update(pre_spikes, post_spikes)
        self.stdp2.update(post_spikes, spikes_out / time_steps)

        return spikes_out / time_steps

# Khởi tạo mạng
model = MNIST_SNN(batch_size=64, dt=1.0).to(device)

# 5. Huấn luyện mạng
num_epochs = 5
time_steps = 20  # Giảm số time steps để tăng tốc độ

# Add training progress tracking
training_progress = []

# Modify the training loop to track progress
for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    for i, (images, labels) in enumerate(pbar):
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)
        
        outputs = model(images, time_steps)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        pbar.set_postfix({'accuracy': f'{accuracy:.2f}%'})
        
        if (i + 1) % 100 == 0:
            print(f"\nBatch [{i+1}/{len(train_loader)}] - Accuracy: {accuracy:.2f}%")
    
    # Record accuracy for this epoch
    training_progress.append(accuracy)

# Plot training progress
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), training_progress, marker='o')
plt.title('Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.savefig(os.path.join(output_dir, "training_progress.png"))
plt.close()

# 6. Trực quan hóa xung và lưu
def plot_spikes(images, model, output_dir, time_steps=50):
    images = images.view(1, -1).to(device)
    spikes_hidden = []
    spikes_output = []
    voltages_hidden = []
    voltages_output = []

    for t in range(time_steps):
        input_spikes = (images > torch.rand(images.size(), device=device)).float()
        hidden_act, _ = model.conn1(input_spikes.view(1, 1, -1), input_spikes.view(1, 1, -1))
        hidden_act = hidden_act.view(1, -1)
        hidden_spikes, _ = model.neuron1(hidden_act)
        hidden_voltages = model.neuron1.v
        
        output_act, _ = model.conn2(hidden_spikes.view(1, 1, -1), hidden_spikes.view(1, 1, -1))
        output_act = output_act.view(1, -1)
        output_spikes, _ = model.neuron2(output_act)
        output_voltages = model.neuron2.v
        
        spikes_hidden.append(hidden_spikes.cpu().detach().numpy())
        spikes_output.append(output_spikes.cpu().detach().numpy())
        voltages_hidden.append(hidden_voltages.cpu().detach().numpy())
        voltages_output.append(output_voltages.cpu().detach().numpy())

    spikes_hidden = np.array(spikes_hidden).squeeze()
    spikes_output = np.array(spikes_output).squeeze()
    voltages_hidden = np.array(voltages_hidden).squeeze()
    voltages_output = np.array(voltages_output).squeeze()

    # Create a comprehensive visualization
    plt.figure(figsize=(18, 12))
    
    # Original image
    plt.subplot(3, 3, 1)
    plt.imshow(images[0].cpu().view(28, 28).numpy(), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    # Input layer spike sum
    plt.subplot(3, 3, 2)
    input_spikes_sum = input_spikes.cpu().view(28, 28).numpy()
    plt.imshow(input_spikes_sum, cmap='hot')
    plt.title("Input Layer Spike Sum")
    plt.colorbar()
    
    # Hidden layer spikes over time
    plt.subplot(3, 3, 4)
    plt.imshow(spikes_hidden.T, cmap='binary', aspect='auto')
    plt.title("Hidden Layer Spikes Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Neuron Index")
    
    # Hidden layer voltages (5 neurons)
    plt.subplot(3, 3, 5)
    for i in range(5):
        plt.plot(voltages_hidden[:, i], label=f'Neuron {i}')
    plt.title("Hidden Layer Voltages (5 Neurons)")
    plt.xlabel("Time Step")
    plt.ylabel("Voltage")
    plt.legend()
    
    # Output layer spikes over time
    plt.subplot(3, 3, 7)
    plt.imshow(spikes_output.T, cmap='binary', aspect='auto')
    plt.title("Output Layer Spikes Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Class (0-9)")
    plt.yticks(range(10), [f'{i}' for i in range(10)])
    
    # Output layer spike counts
    plt.subplot(3, 3, 8)
    spike_counts = spikes_output.sum(0)
    plt.bar(range(10), spike_counts)
    plt.title("Output Layer Spike Counts")
    plt.xlabel("Class")
    plt.ylabel("Number of Spikes")
    plt.xticks(range(10), [f'{i}' for i in range(10)])
    
    # Weight distribution for the target class
    plt.subplot(3, 3, 9)
    target_class = torch.argmax(spike_counts).item()
    weights = model.conn2.weight[:, target_class].cpu().detach().numpy()
    plt.hist(weights, bins=30)
    plt.title(f"Weight Distribution for Class {target_class}")
    plt.xlabel("Weight Value")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comprehensive_spike_visualization.png"))
    plt.close()

def plot_weights(model, output_dir):
    # Plot input-hidden weights
    plt.figure(figsize=(15, 6))
    for i in range(10):
        weights = model.conn1.weight[i].cpu().detach().numpy().reshape(28, 28)
        plt.subplot(2, 5, i+1)
        plt.imshow(weights, cmap='coolwarm')
        plt.title(f'Input→Hidden Weights for Neuron {i}')
        plt.colorbar()
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "input_hidden_weights.png"))
    plt.close()
    
    # Plot hidden-output weights
    plt.figure(figsize=(15, 8))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        weights = model.conn2.weight[:, i].cpu().detach().numpy()
        plt.stem(range(weights.shape[0]), weights)
        plt.title(f'Hidden→Output Weights for Class {i}')
        plt.xlabel('Hidden Neuron')
        plt.ylabel('Weight Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hidden_output_weights.png"))
    plt.close()

# Generate comprehensive visualizations for a sample
sample_image, sample_label = next(iter(test_loader))
print("Generating comprehensive visualizations for a sample test image:")
plot_spikes(sample_image[0], model, output_dir)
plot_weights(model, output_dir)

# 7. Đánh giá mô hình
def evaluate_model(model, loader, time_steps=50):
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)
        outputs = model(images, time_steps)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")

print("Evaluating model...")
evaluate_model(model, test_loader)

# 8. Trực quan hóa dự đoán và lưu
def plot_predictions(loader, model, output_dir, time_steps=50, num_images=9):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images = images.view(images.size(0), -1).to(device)
    outputs = model(images, time_steps)
    predicted = torch.argmax(outputs, dim=1)

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            img = images[i].cpu().view(28, 28).numpy()
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Pred: {predicted[i].item()}, True: {labels[i].item()}")
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "predictions.png"))
    plt.close()

print("Visualizing predictions:")
plot_predictions(test_loader, model, output_dir)