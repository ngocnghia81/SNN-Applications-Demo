"""
SpikingJelly MNIST Inference Script
Script dùng để thực hiện dự đoán với mô hình SNN đã huấn luyện
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image

# Import SpikingJelly
try:
    from spikingjelly.activation_based import neuron, functional, surrogate, layer
    print("SpikingJelly đã được import thành công!")
except ImportError:
    print("SpikingJelly chưa được cài đặt. Đang cài đặt...")
    import subprocess
    subprocess.run(["pip", "install", "spikingjelly"])
    from spikingjelly.activation_based import neuron, functional, surrogate, layer
    print("SpikingJelly đã được cài đặt và import thành công!")

# Tạo thư mục kết quả
os.makedirs('inference_results', exist_ok=True)

# Định nghĩa cấu hình từ model đã huấn luyện
class Config:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = './models'
        self.T = 4
        self.tau = 2.0
        self.v_threshold = 1.0
        self.v_reset = 0.0
        self.hidden_size = 64
        self.input_size = 28 * 28
        
config = Config()

# Các cấu trúc model giống với file huấn luyện
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

# Hàm tải model đã huấn luyện
def load_model(model_path=None):
    """
    Tải mô hình đã huấn luyện
    
    Args:
        model_path: Đường dẫn đến file model
    
    Returns:
        model: Mô hình đã tải
        model_type: Loại mô hình ('mlp' hoặc 'cnn')
    """
    if model_path is None:
        # Kiểm tra cả hai loại model có thể có
        mlp_model_path = os.path.join(config.model_dir, 'SNNMLP_best.pt')
        cnn_model_path = os.path.join(config.model_dir, 'SNNCNN_best.pt')
        
        if os.path.exists(mlp_model_path):
            model_path = mlp_model_path
            model_type = 'mlp'
        elif os.path.exists(cnn_model_path):
            model_path = cnn_model_path
            model_type = 'cnn'
        else:
            # Kiểm tra các model final nếu không có best
            mlp_model_path = os.path.join(config.model_dir, 'SNNMLP_final.pt')
            cnn_model_path = os.path.join(config.model_dir, 'SNNCNN_final.pt')
            
            if os.path.exists(mlp_model_path):
                model_path = mlp_model_path
                model_type = 'mlp'
            elif os.path.exists(cnn_model_path):
                model_path = cnn_model_path
                model_type = 'cnn'
            else:
                raise FileNotFoundError(f"Không tìm thấy model tại {config.model_dir}. Hãy huấn luyện model trước!")
    else:
        # Xác định loại model từ tên file
        if 'SNNMLP' in model_path:
            model_type = 'mlp'
        elif 'SNNCNN' in model_path:
            model_type = 'cnn'
        else:
            # Mặc định là MLP nếu không xác định được
            model_type = 'mlp'
    
    print(f"Tải model từ {model_path}...")
    
    # Khởi tạo model theo loại
    if model_type == 'mlp':
        model = SNNMLP(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            output_size=10,
            tau=config.tau,
            v_threshold=config.v_threshold,
            v_reset=config.v_reset
        )
    else:  # cnn
        model = SNNCNN(
            tau=config.tau,
            v_threshold=config.v_threshold,
            v_reset=config.v_reset
        )
    
    # Tải trọng số
    try:
        checkpoint = torch.load(model_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model tải thành công! Độ chính xác trên tập test: {checkpoint.get('test_acc', 'N/A')}%")
        
        model = model.to(config.device)
        model.eval()
        
        return model, model_type
        
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        raise

# Hàm tiền xử lý ảnh đầu vào
def preprocess_image(image_path):
    """
    Tiền xử lý ảnh đầu vào để phù hợp với đầu vào của mô hình
    
    Args:
        image_path: Đường dẫn đến ảnh cần dự đoán
    
    Returns:
        tensor: Tensor đầu vào cho mô hình
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image = Image.open(image_path).convert('L')
    tensor = transform(image).unsqueeze(0)
    
    return tensor

# Hàm dự đoán
def predict(model, image_tensor, model_type):
    """
    Dự đoán nhãn của ảnh
    
    Args:
        model: Mô hình SNN đã được huấn luyện
        image_tensor: Tensor đầu vào
        model_type: Loại mô hình ('mlp' hoặc 'cnn')
    
    Returns:
        predicted: Nhãn dự đoán
        confidence: Độ tin cậy của dự đoán
        spikes: Danh sách các spike theo thời gian
    """
    model.eval()
    
    # Đưa tensor lên device
    image_tensor = image_tensor.to(config.device)
    
    if image_tensor.dim() == 3:  # [N, H, W]
        image_tensor = image_tensor.unsqueeze(1)  # [N, C, H, W]
    
    # Reset trạng thái của mô hình
    model.reset()
    
    with torch.no_grad():
        spikes = []
        
        # Lặp T bước thời gian
        for t in range(config.T):
            if model_type == 'mlp':
                # Đối với MLP, cần flatten ảnh
                input_tensor = image_tensor.flatten(1) if image_tensor.dim() <= 3 else image_tensor
            else:
                # Đối với CNN, giữ nguyên kích thước
                input_tensor = image_tensor
            
            # Forward pass
            output = model(input_tensor)
            spikes.append(output)
        
        # Ghép các spike theo thời gian
        spikes = torch.stack(spikes) if len(spikes) > 0 else None
        
        # Tính trung bình theo thời gian
        output_mean = spikes.mean(0)
        
        # Dự đoán là lớp có giá trị cao nhất
        confidence, predicted = output_mean.max(1)
        
    return predicted.item(), confidence.item(), spikes.cpu()

# Hàm trực quan hóa kết quả
def visualize_prediction(image_tensor, spikes, predicted, confidence, save_path=None):
    """
    Trực quan hóa kết quả dự đoán
    
    Args:
        image_tensor: Tensor ảnh đầu vào
        spikes: Tensor các spike theo thời gian
        predicted: Nhãn được dự đoán
        confidence: Độ tin cậy
        save_path: Đường dẫn để lưu hình ảnh trực quan hóa
    """
    plt.figure(figsize=(12, 8))
    
    # Hiển thị ảnh gốc
    plt.subplot(2, 2, 1)
    plt.imshow(image_tensor.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    
    # Hiển thị pattern của spike theo thời gian
    plt.subplot(2, 2, 2)
    spike_pattern = spikes.squeeze(1).cpu().numpy()
    plt.imshow(spike_pattern.T, aspect='auto', cmap='hot')
    plt.colorbar(label='Spike Rate')
    plt.xlabel('Time Steps')
    plt.ylabel('Output Neurons')
    plt.title('Spike Pattern')
    
    # Hiển thị kết quả dự đoán
    plt.subplot(2, 2, 3)
    output_mean = spikes.mean(0).squeeze(0).cpu().numpy()
    plt.bar(range(10), output_mean)
    plt.axvline(x=predicted, color='r', linestyle='--', label=f'Predicted: {predicted}')
    plt.xticks(range(10))
    plt.xlabel('Class')
    plt.ylabel('Average Activity')
    plt.title(f'Prediction: {predicted}, Confidence: {confidence:.4f}')
    plt.legend()
    
    # Hiển thị activity theo thời gian cho các lớp
    plt.subplot(2, 2, 4)
    plt.plot(spike_pattern[:, predicted], 'r-', linewidth=2, label=f'Class {predicted}')
    for i in range(10):
        if i != predicted:
            plt.plot(spike_pattern[:, i], 'b-', alpha=0.3, label=f'Class {i}' if i == 0 else None)
    plt.xlabel('Time Steps')
    plt.ylabel('Activity')
    plt.title('Neuron Activity Over Time')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Kết quả trực quan hóa được lưu tại {save_path}")
    
    plt.show()

# Hàm đánh giá trên tập test
def evaluate_on_testset(model, model_type, num_samples=100):
    """
    Đánh giá mô hình trên tập test
    
    Args:
        model: Mô hình SNN đã huấn luyện
        model_type: Loại mô hình ('mlp' hoặc 'cnn')
        num_samples: Số lượng mẫu cần đánh giá
    
    Returns:
        accuracy: Độ chính xác trên tập test
    """
    # Tải MNIST test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    correct = 0
    total = 0
    
    for data, target in test_loader:
        if total >= num_samples:
            break
        
        batch_size = min(data.size(0), num_samples - total)
        data = data[:batch_size]
        target = target[:batch_size]
        
        # Dự đoán từng mẫu
        for i in range(batch_size):
            img = data[i:i+1]
            label = target[i].item()
            
            pred, _, _ = predict(model, img, model_type)
            
            if pred == label:
                correct += 1
        
        total += batch_size
        print(f"Đã dự đoán {total}/{num_samples} mẫu, độ chính xác hiện tại: {100*correct/total:.2f}%")
    
    accuracy = 100 * correct / total
    print(f"Độ chính xác trên {total} mẫu test: {accuracy:.2f}%")
    
    return accuracy

# Hàm main
def main():
    try:
        # Tải model
        model, model_type = load_model()
        
        # Menu lựa chọn
        while True:
            print("\n===== SPIKING NEURAL NETWORK INFERENCE =====")
            print("1. Dự đoán ảnh từ tập MNIST")
            print("2. Dự đoán ảnh tùy chỉnh")
            print("3. Đánh giá trên tập test")
            print("4. Thoát")
            choice = input("Nhập lựa chọn của bạn (1-4): ")
            
            if choice == '1':
                # Tải MNIST
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
                
                # Chọn ngẫu nhiên một ảnh
                idx = int(input("Nhập số thứ tự ảnh muốn dự đoán (0-9999): ") or np.random.randint(0, len(test_dataset)))
                image, label = test_dataset[idx]
                image = image.unsqueeze(0)  # Thêm batch dimension
                
                # Dự đoán
                print(f"Mẫu thứ {idx}, nhãn thật: {label}")
                predicted, confidence, spikes = predict(model, image, model_type)
                print(f"Dự đoán: {predicted}, độ tin cậy: {confidence:.4f}")
                
                # Trực quan hóa
                visualize_prediction(image, spikes, predicted, confidence, save_path='images/inference/mnist_prediction.png')
            
            elif choice == '2':
                # Dự đoán ảnh tùy chỉnh
                image_path = input("Nhập đường dẫn tới ảnh: ")
                
                if not os.path.exists(image_path):
                    print(f"Không tìm thấy file {image_path}")
                    continue
                
                # Tiền xử lý ảnh
                image = preprocess_image(image_path)
                
                # Dự đoán
                predicted, confidence, spikes = predict(model, image, model_type)
                print(f"Dự đoán: {predicted}, độ tin cậy: {confidence:.4f}")
                
                # Trực quan hóa
                visualize_prediction(image, spikes, predicted, confidence, save_path='images/inference/custom_prediction.png')
            
            elif choice == '3':
                # Đánh giá trên tập test
                num_samples = int(input("Nhập số lượng mẫu cần đánh giá (mặc định 100): ") or 100)
                evaluate_on_testset(model, model_type, num_samples=num_samples)
            
            elif choice == '4':
                print("Thoát chương trình!")
                break
                
            else:
                print("Lựa chọn không hợp lệ, vui lòng thử lại.")
                
    except FileNotFoundError:
        print("Không tìm thấy model. Hãy huấn luyện model trước!")
        print("Vui lòng chạy script huấn luyện trước (spikingjelly_mnist.py).")
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main() 