"""
SpikingJelly MNIST Training Script
Huấn luyện SNN sử dụng SpikingJelly, trực quan hóa kết quả và lưu mô hình
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import random
from datetime import datetime

# Import SpikingJelly
try:
    from spikingjelly.activation_based import neuron, functional, surrogate, layer
    from spikingjelly.activation_based import learning, encoding
    from spikingjelly.datasets import play_frame
    print("SpikingJelly đã được import thành công!")
except ImportError:
    print("SpikingJelly chưa được cài đặt. Đang cài đặt...")
    import subprocess
    subprocess.run(["pip", "install", "spikingjelly"])
    from spikingjelly.activation_based import neuron, functional, surrogate, layer
    from spikingjelly.activation_based import learning, encoding
    from spikingjelly.datasets import play_frame
    print("SpikingJelly đã được cài đặt và import thành công!")

# Định nghĩa tham số
class Config:
    def __init__(self):
        # Cài đặt cơ bản
        self.seed = 42
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float
        
        # Cài đặt dữ liệu
        self.batch_size = 64
        self.test_batch_size = 64
        
        # Cài đặt mô hình
        self.model_type = 'mlp'  # 'mlp' hoặc 'cnn'
        self.input_size = 28 * 28  # MNIST image size
        self.hidden_size = 64  # Đã giảm từ 128 xuống 64
        self.output_size = 10    # 10 classes
        
        # Cài đặt neuron
        self.tau = 2.0          # Time constant
        self.v_threshold = 1.0   # Threshold potential
        self.v_reset = 0.0       # Reset potential
        
        # Cài đặt huấn luyện
        self.T = 4              # Số bước thời gian, giảm từ 16 xuống 4 để tránh OOM
        self.epochs = 5
        self.lr = 1e-3
        self.log_interval = 10
        
        # Cài đặt visualization
        self.visual_interval = 100  # Visualization interval
        self.visual_steps = 25      # Số bước hiển thị trong mỗi visualization, giảm từ 50 xuống 25
        
        # Đường dẫn
        self.log_dir = './logs'
        self.model_dir = './models'
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs('./images', exist_ok=True)

config = Config()
os.makedirs(config.log_dir, exist_ok=True)
os.makedirs(config.model_dir, exist_ok=True)
os.makedirs('images', exist_ok=True)

# Định nghĩa mô hình SNN
class SNNMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, tau=2.0, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan()):
        super().__init__()
        # Tạo mạng fully-connected với 3 lớp
        self.fc1 = layer.Linear(input_size, hidden_size)
        self.sn1 = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function)
        
        self.fc2 = layer.Linear(hidden_size, hidden_size)
        self.sn2 = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function)
        
        self.fc3 = layer.Linear(hidden_size, output_size)
        self.sn3 = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function)
        
    def forward(self, x):
        # Kiểm tra kích thước đầu vào và xử lý phù hợp
        # x có thể là [T, N, C, H, W] hoặc [N, C, H, W]
        if x.dim() == 5:  # [T, N, C, H, W]
            # Lưu kích thước ban đầu
            T, N = x.shape[0], x.shape[1]
            # Làm phẳng
            x = x.reshape(T * N, -1)  # [T*N, C*H*W]
            
            # Đảm bảo làm phẳng đúng kích thước cho lớp fc1
            if x.shape[1] != 784:  # 28*28
                x = x.reshape(T * N, 784)
            
            # FC layers
            x = self.fc1(x)
            x = self.sn1(x)
            
            x = self.fc2(x)
            x = self.sn2(x)
            
            x = self.fc3(x)
            x = self.sn3(x)
            
            # Reshape lại [T, N, output_size]
            x = x.reshape(T, N, -1)
        else:
            # x: [N, C, H, W] hoặc đã phẳng [N, features]
            # Làm phẳng nếu chưa
            if x.dim() > 2:
                x = x.flatten(1)  # [N, C*H*W]
            
            # Lớp 1
            x = self.fc1(x)
            x = self.sn1(x)
            
            # Lớp 2
            x = self.fc2(x)
            x = self.sn2(x)
            
            # Lớp 3
            x = self.fc3(x)
            x = self.sn3(x)
        
        return x
    
    def reset(self):
        # Reset các thành phần neuron thay vì gọi functional.reset_net
        if hasattr(self.sn1, 'reset'):
            self.sn1.reset()
        if hasattr(self.sn2, 'reset'):
            self.sn2.reset()
        if hasattr(self.sn3, 'reset'):
            self.sn3.reset()

class SNNCNN(nn.Module):
    def __init__(self, tau=2.0, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan()):
        super().__init__()
        # Tạo mạng CNN kết hợp SNN (giảm số lượng feature maps)
        self.conv1 = layer.Conv2d(1, 16, kernel_size=3, padding=1)  # Giảm 32 -> 16 feature maps
        self.sn1 = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function)
        
        self.pool1 = layer.MaxPool2d(2, 2)
        
        self.conv2 = layer.Conv2d(16, 32, kernel_size=3, padding=1)  # Giảm 64 -> 32 feature maps
        self.sn2 = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function)
        
        self.pool2 = layer.MaxPool2d(2, 2)
        
        # Tính số đầu vào cho lớp fc1: số feature maps * kích thước feature map sau pooling
        # 32 feature maps, kích thước 7x7 sau 2 lần pooling
        self.fc1 = layer.Linear(32 * 7 * 7, config.hidden_size)
        self.sn3 = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function)
        
        self.fc2 = layer.Linear(config.hidden_size, 10)
        self.sn4 = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function)
        
    def forward(self, x):
        # Kiểm tra kích thước đầu vào và xử lý phù hợp
        # x có thể là [T, N, C, H, W] hoặc [N, C, H, W]
        if x.dim() == 5:  # [T, N, C, H, W]
            # Lưu kích thước ban đầu
            T, N = x.shape[0], x.shape[1]
            x = x.flatten(0, 1)  # [T*N, C, H, W]
            
            # Đảm bảo kích thước đúng
            if x.shape[1:] != (1, 28, 28):
                # Reshape nếu cần thiết
                x = x.reshape(-1, 1, 28, 28)
            
            # Conv layers
            x = self.conv1(x)
            x = self.sn1(x)
            x = self.pool1(x)
            
            x = self.conv2(x)
            x = self.sn2(x)
            x = self.pool2(x)
            
            # Flatten
            x = x.flatten(1)
            
            # FC layers
            x = self.fc1(x)
            x = self.sn3(x)
            
            x = self.fc2(x)
            x = self.sn4(x)
            
            # Reshape lại để trả về [T, N, num_classes]
            x = x.reshape(T, N, -1)
            
        else:  # [N, C, H, W]
            # Conv layers
            x = self.conv1(x)
            x = self.sn1(x)
            x = self.pool1(x)
            
            x = self.conv2(x)
            x = self.sn2(x)
            x = self.pool2(x)
            
            # Flatten
            x = x.flatten(1)
            
            # FC layers
            x = self.fc1(x)
            x = self.sn3(x)
            
            x = self.fc2(x)
            x = self.sn4(x)
        
        return x
    
    def reset(self):
        # Reset các thành phần neuron thay vì gọi functional.reset_net
        if hasattr(self.sn1, 'reset'):
            self.sn1.reset()
        if hasattr(self.sn2, 'reset'):
            self.sn2.reset()
        if hasattr(self.sn3, 'reset'):
            self.sn3.reset()
        if hasattr(self.sn4, 'reset'):
            self.sn4.reset()

# Hàm tải dữ liệu MNIST
def load_mnist(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, test_loader

# Hàm training
def train(model, device, train_loader, optimizer, epoch, writer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    model_type = type(model).__name__
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Đảm bảo dữ liệu có kích thước T, N, ...
        if data.dim() == 3:  # [N, H, W]
            # Thêm chiều C (channel)
            data = data.unsqueeze(1)  # [N, C, H, W]
        
        # Đặt lại trạng thái của các neuron trước khi mô phỏng
        if hasattr(model, 'reset'):
            model.reset()
            
        # Repeat T times
        # [T, N, C, H, W]
        T_data = data.unsqueeze(0).repeat(config.T, 1, 1, 1, 1)
                    
        # Tạo biến để lưu các giá trị trung gian cho visualization
        if batch_idx == 0 and epoch % 5 == 0:
            # Lấy mẫu đầu tiên để visualization
            if model_type == "SNNCNN":
                # Với CNN, dữ liệu đã ở dạng [T, 1, C, H, W]
                sample_input = T_data[:, 0:1]  # [T, 1, C, H, W]
            else:  # SNNMLP
                # Với MLP, cần flatten dữ liệu
                sample_input = T_data[:, 0:1].flatten(2)  # [T, 1, 784]
                
            spike_records, voltage_records = {}, {}
            hooks = []
            
            # Đăng ký hooks cho các lớp neuron
            for name, module in model.named_modules():
                if isinstance(module, neuron.LIFNode):
                    def hook_fn(m, x, y, module_name=name):
                        if module_name not in spike_records:
                            spike_records[module_name] = []
                            voltage_records[module_name] = []
                        spike_records[module_name].append(y.detach().cpu().numpy())
                        voltage_records[module_name].append(m.v.detach().cpu().numpy())
                    
                    hooks.append(module.register_forward_hook(hook_fn))
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(T_data)  # [T, N, num_classes]
        loss = F.cross_entropy(output.mean(0), target)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Tính loss và accuracy
        train_loss += loss.item()
        pred = output.mean(0).argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % config.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/accuracy', 100. * correct / total, step)
        
        # Visualization
        if batch_idx == 0 and epoch % 5 == 0:
            # Xóa hooks sau khi đã thu thập dữ liệu
            for hook in hooks:
                hook.remove()
            
            # Vẽ và lưu các hình ảnh visualization
            visualize_snn_activity(sample_input, spike_records, voltage_records, epoch, model_type)
            
            # Thêm visualization vào TensorBoard
            for name, spikes in spike_records.items():
                spikes = np.concatenate(spikes, axis=0)  # [T, ...]
                if model_type == "SNNCNN" and not name.startswith("sn3") and not name.startswith("sn4"):
                    # Đối với các lớp tích chập, tính trung bình theo chiều không gian
                    spikes_mean = spikes.mean(axis=(2, 3))  # [T, C]
                    fig = plt.figure(figsize=(10, 6))
                    plt.imshow(spikes_mean.T, aspect='auto', cmap='hot')
                    plt.colorbar(label='Spike Rate')
                    plt.xlabel('Time steps')
                    plt.ylabel('Channels')
                    plt.title(f'{name} Spike Activity')
                    writer.add_figure(f'spikes/{name}', fig, epoch)
                else:
                    # Đối với các lớp fully connected
                    fig = plt.figure(figsize=(10, 6))
                    plt.imshow(spikes.reshape(config.T, -1).T, aspect='auto', cmap='hot')
                    plt.colorbar(label='Spike Rate')
                    plt.xlabel('Time steps')
                    plt.ylabel('Neurons')
                    plt.title(f'{name} Spike Activity')
                    writer.add_figure(f'spikes/{name}', fig, epoch)
            
    train_loss /= len(train_loader)
    accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return train_loss, accuracy

# Hàm đánh giá (evaluation)
def test(model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            if data.dim() == 3:  # [N, H, W]
                data = data.unsqueeze(1)  # [N, C, H, W]
            
            # Đặt lại trạng thái của các neuron
            if hasattr(model, 'reset'):
                model.reset()
                
            # Repeat T times
            # [T, N, C, H, W]
            T_data = data.unsqueeze(0).repeat(config.T, 1, 1, 1, 1)
            
            # Forward pass
            output = model(T_data)  # [T, N, num_classes]
            loss = F.cross_entropy(output.mean(0), target)
            
            # Tính toán loss và accuracy
            test_loss += loss.item()
            pred = output.mean(0).argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'Test Epoch: {epoch}, Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # Log metrics vào TensorBoard
    writer.add_scalar('test/loss', test_loss, epoch)
    writer.add_scalar('test/accuracy', accuracy, epoch)
    
    return test_loss, accuracy

# Hàm trực quan hóa
def visualize_snn_activity(inputs, spike_records, voltage_records, epoch, model_type):
    """
    Trực quan hóa hoạt động của SNN
    
    Args:
        inputs: Dữ liệu đầu vào [T, 1, ...]
        spike_records: Dict chứa các bản ghi spike của các layer
        voltage_records: Dict chứa các bản ghi voltage của các layer
        epoch: Epoch hiện tại
        model_type: Loại mô hình (SNNMLP hoặc SNNCNN)
    """
    # Tạo thư mục cho hình ảnh nếu chưa tồn tại
    os.makedirs("./images", exist_ok=True)
    
    # Tạo một figure với nhiều subplots
    n_rows = len(spike_records) + 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4 * n_rows), gridspec_kw={'width_ratios': [1, 3]})
    
    # Hiển thị ảnh đầu vào
    if model_type == "SNNCNN":
        # Đối với CNN, lấy ảnh từ frame đầu tiên
        input_img = inputs[0, 0].cpu().numpy()  # [C, H, W]
        axes[0, 0].imshow(input_img.squeeze(), cmap='gray')
    else:
        # Đối với MLP, ảnh đã được flatten
        input_reshaped = inputs[0, 0].cpu().numpy().reshape(28, 28)
        axes[0, 0].imshow(input_reshaped, cmap='gray')
    
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # Hiển thị input spike pattern (temporal pattern)
    if model_type == "SNNCNN":
        # Đối với CNN, lấy trung bình không gian
        try:
            input_over_time = inputs[:, 0, 0].cpu().numpy().squeeze()  # [T]
            if input_over_time.ndim <= 1:  # Đảm bảo input_over_time là vector 1D
                axes[0, 1].plot(input_over_time)
                axes[0, 1].set_title('Input Intensity Over Time')
            else:
                # Nếu vẫn là tensor nhiều chiều, hiển thị dưới dạng heatmap
                axes[0, 1].imshow(np.mean(input_over_time, axis=(1,2) if input_over_time.ndim > 2 else 1), 
                                aspect='auto', cmap='hot')
                axes[0, 1].set_title('Input Intensity Over Time')
        except Exception as e:
            print(f"Lỗi khi visualize CNN input: {e}")
            # Fallback nếu có lỗi
            axes[0, 1].text(0.5, 0.5, f"Không thể hiển thị\n{str(e)}", 
                          horizontalalignment='center', verticalalignment='center')
    else:
        # Đối với MLP, hiển thị spike pattern dưới dạng heatmap
        try:
            input_flat = inputs[:, 0].cpu().numpy()  # [T, 784]
            axes[0, 1].imshow(input_flat, aspect='auto', cmap='hot')
            axes[0, 1].set_title('Input Over Time')
        except Exception as e:
            print(f"Lỗi khi visualize MLP input: {e}")
            # Fallback nếu có lỗi
            axes[0, 1].text(0.5, 0.5, f"Không thể hiển thị\n{str(e)}", 
                          horizontalalignment='center', verticalalignment='center')
    
    axes[0, 1].set_xlabel('Time Steps')
    
    # Hiển thị spike và voltage cho mỗi layer
    for i, (name, spikes) in enumerate(spike_records.items(), 1):
        try:
            # Chuyển đổi danh sách numpy arrays thành một numpy array
            spikes_array = np.concatenate(spikes, axis=0)  # [T, ...]
            voltage_array = np.concatenate(voltage_records[name], axis=0)  # [T, ...]
            
            # Xử lý cho layer CNN
            if model_type == "SNNCNN" and not name.startswith("sn3") and not name.startswith("sn4"):
                # Đối với các lớp tích chập, tính trung bình theo chiều không gian
                spikes_mean = spikes_array.mean(axis=(2, 3)) if spikes_array.ndim > 2 else spikes_array  # [T, C]
                voltage_mean = voltage_array.mean(axis=(2, 3)) if voltage_array.ndim > 2 else voltage_array  # [T, C]
                
                # Hiển thị spike rate trung bình
                im1 = axes[i, 0].imshow(spikes_mean.mean(axis=0).reshape(1, -1), cmap='hot')
                axes[i, 0].set_title(f'{name} Avg Spike Rate')
                axes[i, 0].set_yticks([])
                plt.colorbar(im1, ax=axes[i, 0])
                
                # Hiển thị spike pattern theo thời gian
                im2 = axes[i, 1].imshow(spikes_mean.T, aspect='auto', cmap='hot')
                axes[i, 1].set_title(f'{name} Spike Pattern')
                axes[i, 1].set_xlabel('Time Steps')
                axes[i, 1].set_ylabel('Channels')
                plt.colorbar(im2, ax=axes[i, 1])
            else:
                # Đối với các lớp fully connected, reshape thành 2D
                spikes_flat = spikes_array.reshape(spikes_array.shape[0], -1)  # [T, neurons]
                voltage_flat = voltage_array.reshape(voltage_array.shape[0], -1)  # [T, neurons]
                
                # Hiển thị spike rate trung bình
                im1 = axes[i, 0].imshow(spikes_flat.mean(axis=0).reshape(1, -1), cmap='hot')
                axes[i, 0].set_title(f'{name} Avg Spike Rate')
                axes[i, 0].set_yticks([])
                plt.colorbar(im1, ax=axes[i, 0])
                
                # Hiển thị spike pattern theo thời gian
                im2 = axes[i, 1].imshow(spikes_flat.T, aspect='auto', cmap='hot')
                axes[i, 1].set_title(f'{name} Spike Pattern')
                axes[i, 1].set_xlabel('Time Steps')
                axes[i, 1].set_ylabel('Neurons')
                plt.colorbar(im2, ax=axes[i, 1])
        except Exception as e:
            print(f"Lỗi khi visualize layer {name}: {e}")
            axes[i, 0].text(0.5, 0.5, f"Không thể hiển thị\n{str(e)}", 
                          horizontalalignment='center', verticalalignment='center')
            axes[i, 1].text(0.5, 0.5, f"Không thể hiển thị\n{str(e)}", 
                          horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f'./images/activity_epoch_{epoch}.png', dpi=150)
    plt.close()

# Hàm dự đoán từ mô hình đã huấn luyện
def predict(model, images, config):
    model.eval()
    with torch.no_grad():
        images = images.to(config.device)
        
        output_spikes = []
        model.reset()
        
        # Mô phỏng T bước thời gian
        for t in range(config.T):
            output = model(images)
            output_spikes.append(output)
        
        # Tính tổng số spike qua thời gian
        output_spikes = torch.stack(output_spikes, dim=0)
        total_spikes = output_spikes.sum(0)
        
        # Nhãn dự đoán là lớp có số spike nhiều nhất
        predicted = total_spikes.argmax(dim=1)
        
        # Tính độ tin cậy dựa trên tỉ lệ spike
        confidence = F.softmax(total_spikes, dim=1)
    
    return predicted, confidence, output_spikes

# Hàm trực quan hóa dự đoán
def visualize_prediction(model, device, test_loader, num_samples=5):
    """
    Trực quan hóa kết quả dự đoán từ mô hình
    
    Args:
        model: Mô hình SNN đã huấn luyện
        device: Thiết bị để chạy model
        test_loader: Test data loader
        num_samples: Số lượng mẫu cần hiển thị
    """
    # Chuyển model sang chế độ đánh giá
    model.eval()
    
    # Tạo thư mục cho hình ảnh nếu chưa tồn tại
    os.makedirs("./images", exist_ok=True)
    
    plt.figure(figsize=(15, num_samples * 3))
    
    all_images = []
    all_targets = []
    all_predictions = []
    
    # Lấy một số mẫu từ test_loader
    with torch.no_grad():
        for data, target in test_loader:
            all_images.append(data)
            all_targets.append(target)
            
            # Chỉ lấy một số mẫu
            if len(all_images) * data.shape[0] >= num_samples:
                break
    
    # Ghép các batch để dễ dàng lấy các mẫu
    all_images = torch.cat(all_images)[:num_samples]
    all_targets = torch.cat(all_targets)[:num_samples]
    
    # Dự đoán cho mỗi mẫu
    for i in range(num_samples):
        # Chuẩn bị dữ liệu đầu vào
        data = all_images[i:i+1].to(device)
        target = all_targets[i].item()
        
        if data.dim() == 3:  # [N, H, W]
            data = data.unsqueeze(1)  # [N, C, H, W]
            
        # Đặt lại trạng thái của các neuron
        if hasattr(model, 'reset'):
            model.reset()
        
        # Repeat T times [T, N, C, H, W]
        T_data = data.unsqueeze(0).repeat(config.T, 1, 1, 1, 1)
        
        # Forward pass
        output = model(T_data)  # [T, N, num_classes]
        
        # Tính toán dự đoán
        output_mean = output.mean(0)  # [N, num_classes]
        confidence, pred_class = output_mean.max(1)
        pred_class = pred_class.item()
        confidence = confidence.item()
        
        # Lưu dự đoán
        all_predictions.append(pred_class)
        
        # Hiển thị ảnh và dự đoán
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(all_images[i].squeeze().cpu().numpy(), cmap='gray')
        plt.title(f'Input Image {i+1}')
        plt.axis('off')
        
        # Hiển thị activations
        plt.subplot(num_samples, 3, i * 3 + 2)
        spikes_over_time = output.squeeze().cpu().numpy()
        plt.imshow(spikes_over_time.T, aspect='auto', cmap='hot')
        plt.colorbar(label='Activation')
        plt.xlabel('Time steps')
        plt.ylabel('Classes')
        plt.title('Output Activations')
        
        # Hiển thị confidence
        plt.subplot(num_samples, 3, i * 3 + 3)
        class_confidence = output_mean.squeeze().cpu().numpy()
        plt.bar(range(10), class_confidence)
        plt.xticks(range(10))
        plt.xlabel('Class')
        plt.ylabel('Confidence')
        plt.title(f'Prediction: {pred_class} (Truth: {target})\nConfidence: {confidence:.4f}')
        
        # Highlight the predicted and true classes
        if pred_class == target:
            plt.bar(pred_class, class_confidence[pred_class], color='green')
        else:
            plt.bar(pred_class, class_confidence[pred_class], color='red')
            plt.bar(target, class_confidence[target], color='blue')
    
    plt.tight_layout()
    plt.savefig('./images/predictions.png', dpi=150)
    plt.close()
    
    acc = sum([1 if p == t else 0 for p, t in zip(all_predictions, all_targets)]) / len(all_predictions)
    print(f"Accuracy on {num_samples} visualized samples: {acc*100:.2f}%")

def visualize_training_progress(epoch, train_loss, train_acc, test_loss, test_acc, save_dir='./images/progress'):
    """
    Trực quan hóa tiến trình huấn luyện sau mỗi epoch
    
    Args:
        epoch: Epoch hiện tại
        train_loss: Danh sách loss trong quá trình training
        train_acc: Danh sách accuracy trong quá trình training
        test_loss: Danh sách loss trong quá trình testing
        test_acc: Danh sách accuracy trong quá trình testing
        save_dir: Thư mục lưu hình ảnh
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = list(range(1, epoch + 1))
    
    # Tạo figure với 2 subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Subplot cho Loss
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    ax1.plot(epochs, test_loss, 'r-', label='Test Loss')
    ax1.set_title(f'Training & Test Loss (Epoch {epoch})')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Subplot cho Accuracy
    ax2.plot(epochs, train_acc, 'b-', label='Train Accuracy')
    ax2.plot(epochs, test_acc, 'r-', label='Test Accuracy')
    ax2.set_title(f'Training & Test Accuracy (Epoch {epoch})')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/progress_epoch_{epoch}.png', dpi=150)
    plt.close()

def visualize_spike_timing(model, data_loader, config, num_samples=3, save_dir='./images/spikes'):
    """
    Trực quan hóa hoạt động phát spike theo thời gian của mô hình
    
    Args:
        model: Mô hình SNN đã huấn luyện
        data_loader: Data loader
        config: Cấu hình
        num_samples: Số lượng mẫu cần visualize
        save_dir: Thư mục lưu hình ảnh
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Chuyển model sang chế độ đánh giá
    model.eval()
    model_type = type(model).__name__
    
    # Lấy một số mẫu
    data_iter = iter(data_loader)
    samples, targets = next(data_iter)
    samples = samples[:num_samples]
    targets = targets[:num_samples]
    
    for i, (sample, target) in enumerate(zip(samples, targets)):
        plt.figure(figsize=(15, 10))
        
        # Hiển thị ảnh gốc
        plt.subplot(3, 3, 1)
        plt.imshow(sample.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f'Input Image (Label: {target.item()})')
        plt.axis('off')
        
        # Chuẩn bị dữ liệu
        sample = sample.unsqueeze(0).to(config.device)  # [1, C, H, W]
        if sample.dim() == 3:
            sample = sample.unsqueeze(1)  # [1, 1, H, W]
        
        # Đặt lại trạng thái của mô hình
        model.reset()
        
        # Theo dõi spike và voltage của mỗi lớp theo thời gian
        spike_records = {}
        voltage_records = {}
        
        # Đăng ký hooks cho mỗi neuron layer
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, neuron.LIFNode):
                def hook_fn(m, x, y, module_name=name):
                    if module_name not in spike_records:
                        spike_records[module_name] = []
                        voltage_records[module_name] = []
                    spike_records[module_name].append(y.detach().cpu().numpy())
                    voltage_records[module_name].append(m.v.detach().cpu().numpy())
                
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Mô phỏng từng bước thời gian và lưu kết quả
        outputs = []
        for t in range(config.T):
            if model_type == "SNNCNN":
                # Với CNN, giữ nguyên kích thước ảnh
                x = sample.clone()
            else:
                # Với MLP, flatten ảnh
                x = sample.flatten(1)
            
            # Forward pass
            with torch.no_grad():
                output = model(x)
                outputs.append(output)
        
        # Xóa hooks
        for hook in hooks:
            hook.remove()
        
        # Hiển thị spike pattern cho mỗi lớp
        layer_idx = 2
        for name, spikes in spike_records.items():
            spikes = np.concatenate(spikes)  # [T, ...]
            voltages = np.concatenate(voltage_records[name])  # [T, ...]
            
            if model_type == "SNNCNN" and not name.startswith("sn3") and not name.startswith("sn4"):
                # Đối với các lớp tích chập, tính trung bình theo chiều không gian
                spikes_mean = spikes.mean(axis=(2, 3)) if spikes.ndim > 2 else spikes
                voltages_mean = voltages.mean(axis=(2, 3)) if voltages.ndim > 2 else voltages
                
                plt.subplot(3, 3, layer_idx)
                plt.imshow(spikes_mean.T, aspect='auto', cmap='hot')
                plt.colorbar()
                plt.title(f'{name} Spike Activity')
                plt.xlabel('Time Steps')
                plt.ylabel('Channels')
                
                plt.subplot(3, 3, layer_idx + 3)
                plt.imshow(voltages_mean.T, aspect='auto', cmap='viridis')
                plt.colorbar()
                plt.title(f'{name} Membrane Potential')
                plt.xlabel('Time Steps')
                plt.ylabel('Channels')
            else:
                # Đối với các lớp fully connected, reshape thành 2D
                spikes_flat = spikes.reshape(spikes.shape[0], -1)
                voltages_flat = voltages.reshape(voltages.shape[0], -1)
                
                plt.subplot(3, 3, layer_idx)
                plt.imshow(spikes_flat.T, aspect='auto', cmap='hot')
                plt.colorbar()
                plt.title(f'{name} Spike Activity')
                plt.xlabel('Time Steps')
                plt.ylabel('Neurons')
                
                plt.subplot(3, 3, layer_idx + 3)
                plt.imshow(voltages_flat.T, aspect='auto', cmap='viridis')
                plt.colorbar()
                plt.title(f'{name} Membrane Potential')
                plt.xlabel('Time Steps')
                plt.ylabel('Neurons')
            
            layer_idx += 1
            if layer_idx > 3:  # Limit to displaying only 2 layers
                break
        
        # Hiển thị kết quả dự đoán
        outputs = torch.cat(outputs)  # [T, 1, num_classes]
        outputs_mean = outputs.mean(0).squeeze()  # [num_classes]
        
        plt.subplot(3, 3, 9)
        plt.bar(range(10), outputs_mean.cpu().numpy())
        plt.axvline(x=target.item(), color='r', linestyle='--', label=f'True: {target.item()}')
        plt.title('Output Layer Activity')
        plt.xlabel('Class')
        plt.ylabel('Average Activity')
        plt.xticks(range(10))
        plt.legend()
        
        plt.tight_layout()
        
        # Lưu frame vào file
        frame_path = f'{save_dir}/spike_timing_sample_{i}.png'
        plt.savefig(frame_path, dpi=100)
    
    return

def visualize_feature_maps(model, data_loader, config, save_dir='./images/features'):
    """
    Trực quan hóa các feature maps của mô hình CNN
    
    Args:
        model: Mô hình SNN đã huấn luyện (phải là SNNCNN)
        data_loader: Data loader
        config: Cấu hình
        save_dir: Thư mục lưu hình ảnh
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if not isinstance(model, SNNCNN):
        print("Hàm này chỉ dùng cho mô hình SNNCNN!")
        return
    
    # Chuyển model sang chế độ đánh giá
    model.eval()
    
    # Lấy một mẫu
    data_iter = iter(data_loader)
    sample, target = next(data_iter)
    sample = sample[0].unsqueeze(0)  # Lấy mẫu đầu tiên
    
    # Chuẩn bị dữ liệu
    sample = sample.to(config.device)
    if sample.dim() == 3:
        sample = sample.unsqueeze(1)  # [1, 1, H, W]
    
    # Đặt lại trạng thái của mô hình
    model.reset()
    
    # Lưu trữ feature maps
    feature_maps = {}
    
    # Khai báo hooks để lấy feature maps
    def hook_fn(name):
        def hook(model, input, output):
            feature_maps[name] = output.detach().cpu()
        return hook
    
    # Đăng ký các hooks
    hooks = [
        model.conv1.register_forward_hook(hook_fn('conv1')),
        model.sn1.register_forward_hook(hook_fn('sn1')),
        model.conv2.register_forward_hook(hook_fn('conv2')),
        model.sn2.register_forward_hook(hook_fn('sn2'))
    ]
    
    # Forward pass
    with torch.no_grad():
        model(sample)
    
    # Xóa hooks
    for hook in hooks:
        hook.remove()
    
    # Trực quan hóa feature maps
    for name, feature_map in feature_maps.items():
        if not name.startswith('sn'):  # Chỉ hiển thị feature maps từ các lớp conv
            continue
            
        # Lấy kích thước của feature map
        num_features = feature_map.size(1)
        num_rows = int(np.ceil(np.sqrt(num_features)))
        num_cols = int(np.ceil(num_features / num_rows))
        
        plt.figure(figsize=(num_cols * 2, num_rows * 2))
        
        for i in range(num_features):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(feature_map[0, i].numpy(), cmap='viridis')
            plt.axis('off')
        
        plt.suptitle(f'Feature Maps: {name} (Label: {target[0].item()})')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_maps_{name}.png', dpi=150)
        plt.close()

def visualize_membrane_potential_evolution(model, sample, target, config, save_dir='./images/membrane'):
    """
    Trực quan hóa sự tiến hóa của điện thế màng neuron theo thời gian
    
    Args:
        model: Mô hình SNN đã huấn luyện
        sample: Mẫu dữ liệu đầu vào [C, H, W]
        target: Nhãn của mẫu
        config: Cấu hình
        save_dir: Thư mục lưu hình ảnh
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Chuẩn bị dữ liệu
    sample = sample.unsqueeze(0).to(config.device)  # [1, C, H, W]
    if sample.dim() == 3:
        sample = sample.unsqueeze(1)  # [1, 1, H, W]
    
    # Cho mô hình sang chế độ đánh giá
    model.eval()
    model.reset()
    model_type = type(model).__name__
    
    # Chuẩn bị dữ liệu input
    if model_type == "SNNCNN":
        x = sample.clone()
    else:
        x = sample.flatten(1)
    
    # Lưu trữ điện thế màng theo thời gian
    membrane_potentials = []
    outputs = []
    spikes = []
    
    # Mô phỏng từng bước thời gian
    for t in range(config.T):
        # Forward pass
        with torch.no_grad():
            # Tìm tất cả các lớp neuron
            for name, module in model.named_modules():
                if isinstance(module, neuron.LIFNode):
                    # Reset hook nếu đã đăng ký
                    if hasattr(module, '_forward_hooks'):
                        module._forward_hooks.clear()
                    
                    # Đăng ký hook để lấy điện thế
                    def hook_fn(m, x, y, t=t):
                        # Lưu điện thế và spike
                        membrane_potentials.append((t, name, m.v.detach().cpu().clone()))
                        spikes.append((t, name, y.detach().cpu().clone()))
                    
                    module.register_forward_hook(hook_fn)
            
            # Chạy forward pass
            out = model(x)
            outputs.append(out)
    
    # Tạo animation frame theo thời gian
    frames = []
    for t in range(config.T):
        # Tạo một figure mới cho mỗi frame
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Membrane Potential Evolution (t={t}, Label={target.item()})')
        
        # Hiển thị ảnh đầu vào
        axs[0, 0].imshow(sample.squeeze().cpu(), cmap='gray')
        axs[0, 0].set_title('Input Image')
        axs[0, 0].axis('off')
        
        # Lấy điện thế và spike của từng lớp tại thời điểm t
        t_potentials = [p for p in membrane_potentials if p[0] == t]
        t_spikes = [s for s in spikes if s[0] == t]
        
        # Hiển thị điện thế của lớp đầu
        if t_potentials:
            # Lớp đầu tiên
            first_layer = t_potentials[0]
            potential = first_layer[2]
            
            if model_type == "SNNCNN" and (first_layer[1] == 'sn1' or first_layer[1] == 'sn2'):
                # Đối với CNN, hiển thị heatmap của kênh đầu tiên
                axs[0, 1].imshow(potential[0, 0].numpy(), cmap='viridis')
                axs[0, 1].set_title(f'{first_layer[1]} Membrane Potential')
            else:
                # Đối với MLP, hiển thị dưới dạng bar chart
                axs[0, 1].bar(range(min(50, potential.shape[1])), potential[0, :50].numpy())
                axs[0, 1].set_title(f'{first_layer[1]} Membrane Potential')
                axs[0, 1].set_xlabel('Neuron Index')
                axs[0, 1].set_ylabel('Membrane Potential')
        
        # Hiển thị điện thế của lớp cuối
        if len(t_potentials) > 1:
            # Lớp cuối cùng
            last_layer = t_potentials[-1]
            potential = last_layer[2]
            
            # Hiển thị dưới dạng bar chart
            axs[1, 0].bar(range(potential.shape[1]), potential[0].numpy())
            axs[1, 0].set_title(f'{last_layer[1]} Membrane Potential')
            axs[1, 0].set_xlabel('Output Class')
            axs[1, 0].set_ylabel('Membrane Potential')
            
            # Highlight true class
            axs[1, 0].axvline(x=target.item(), color='r', linestyle='--')
        
        # Hiển thị output tích lũy đến thời điểm hiện tại
        if outputs:
            output_accum = torch.stack([outputs[i] for i in range(t+1)]).mean(0)
            axs[1, 1].bar(range(10), output_accum[0].cpu().numpy())
            axs[1, 1].set_title(f'Accumulated Output (t=0...{t})')
            axs[1, 1].set_xlabel('Output Class')
            axs[1, 1].set_ylabel('Mean Activity')
            
            # Highlight true class
            axs[1, 1].axvline(x=target.item(), color='r', linestyle='--')
        
        plt.tight_layout()
        
        # Lưu frame vào file
        frame_path = f'{save_dir}/frame_{t:03d}.png'
        plt.savefig(frame_path, dpi=100)
        frames.append(frame_path)
        plt.close(fig)
    
    return frames

# Hàm main để chạy quá trình huấn luyện và đánh giá
def main():
    # Đặt seed để tái tạo kết quả
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Khởi tạo TensorBoard writer
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(log_dir=os.path.join(config.log_dir, current_time))
    
    # Tải dữ liệu MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Tạo train/test dataloaders
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True, 
        drop_last=True
    )
    
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True
    )
    
    # Chọn loại mô hình và khởi tạo
    if config.model_type.lower() == 'cnn':
        model = SNNCNN(
            tau=config.tau,
            v_threshold=config.v_threshold,
            v_reset=config.v_reset
        )
    else:
        model = SNNMLP(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            tau=config.tau,
            v_threshold=config.v_threshold,
            v_reset=config.v_reset
        )
    
    model = model.to(config.device)
    print(f"Model: {type(model).__name__}")
    print(f"Device: {config.device}")
    
    # Summary model (có thể bỏ qua vì có thể gây lỗi với CUDA)
    try:
        if config.model_type.lower() == 'cnn':
            sample_input = torch.randn(config.T, 1, 1, 28, 28).to(config.device)
            writer.add_graph(model, sample_input)
        else:
            sample_input = torch.randn(config.T, 1, config.input_size).to(config.device)
            writer.add_graph(model, sample_input)
    except Exception as e:
        print(f"Không thể thêm mô hình vào TensorBoard: {e}")
    
    # Khởi tạo optimizer và loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    # Lưu trữ metrics để trực quan hóa tiến trình
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # Training loop
    best_acc = 0
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train(model, config.device, train_loader, optimizer, epoch, writer)
        test_loss, test_acc = test(model, config.device, test_loader, epoch, writer)
        
        # Lưu metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Trực quan hóa tiến trình huấn luyện
        visualize_training_progress(epoch, train_losses, train_accs, test_losses, test_accs)
        
        # Lưu model có accuracy cao nhất
        if test_acc > best_acc:
            best_acc = test_acc
            model_path = os.path.join(config.model_dir, f'{type(model).__name__}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'test_acc': test_acc,
            }, model_path)
            print(f'Model saved to {model_path}')
    
    # Lưu model cuối cùng
    model_path = os.path.join(config.model_dir, f'{type(model).__name__}_final.pt')
    torch.save({
        'epoch': config.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': train_acc,
        'test_acc': test_acc,
    }, model_path)
    print(f'Final model saved to {model_path}')
    
    # Trực quan hóa dự đoán
    print("Trực quan hóa kết quả dự đoán...")
    visualize_prediction(model, config.device, test_loader)
    
    # Trực quan hóa spike timing
    print("Trực quan hóa spike timing...")
    visualize_spike_timing(model, test_loader, config)
    
    # Nếu là mô hình CNN, trực quan hóa feature maps
    if config.model_type.lower() == 'cnn':
        print("Trực quan hóa feature maps...")
        visualize_feature_maps(model, test_loader, config)
    
    # Trực quan hóa sự tiến hóa của điện thế màng
    print("Trực quan hóa sự tiến hóa của điện thế màng...")
    sample, target = next(iter(test_loader))
    frames = visualize_membrane_potential_evolution(model, sample[0], target[0], config)
    
    # Đóng TensorBoard writer
    writer.close()
    
    print("Quá trình huấn luyện và trực quan hóa đã hoàn thành!")
    print(f"Các ảnh trực quan hóa được lưu trong thư mục 'images/'")
    
    return model

if __name__ == "__main__":
    main() 