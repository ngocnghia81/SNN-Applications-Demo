"""
SNN MNIST Inference with BindsNET - Sử dụng mô hình đã huấn luyện sẵn
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import gdown
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.encoding import PoissonEncoder
from bindsnet.learning import PostPre
from tqdm import tqdm
from google.colab import drive
import time
import matplotlib

# Kết nối Google Drive (nếu chạy trên Colab)
try:
    drive.mount('/content/drive')
    base_path = '/content/drive/MyDrive/SNN_Results'  # Đổi thành đường dẫn trong Drive của bạn
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"Đã tạo thư mục lưu kết quả: {base_path}")
except:
    base_path = './results'
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"Đã tạo thư mục lưu kết quả: {base_path}")

# Tạo thư mục để lưu hình ảnh
output_dir = os.path.join(base_path, "snn_inference_results")
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Đã tạo thư mục đầu ra: {output_dir}")

# Kiểm tra quyền ghi vào thư mục
test_file = os.path.join(output_dir, "test_write.txt")
try:
    with open(test_file, 'w') as f:
        f.write("Test file to verify write permissions")
    print(f"Kiểm tra quyền ghi thành công tại: {output_dir}")
    os.remove(test_file)  # Xóa file test sau khi kiểm tra
except Exception as e:
    print(f"LỖI: Không thể ghi vào thư mục đầu ra: {e}")
    # Thử tạo thư mục con trong thư mục hiện tại
    output_dir = os.path.join(os.getcwd(), "snn_inference_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Đã chuyển thư mục đầu ra sang: {output_dir}")

# In ra đường dẫn đầy đủ để dễ tìm
print(f"Thư mục đầu ra đầy đủ: {os.path.abspath(output_dir)}")

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# Cấu hình matplotlib cho môi trường không có GUI (như server hoặc Google Colab)
matplotlib.use('Agg')  # Sử dụng Agg backend không yêu cầu GUI
print("Đã cấu hình matplotlib cho môi trường không có GUI")

# Tải mô hình từ Google Drive hoặc URL
def download_pretrained_model(output_path, source_type="url", source_id=None):
    """Tải mô hình đã huấn luyện sẵn từ Google Drive hoặc URL"""
    if os.path.exists(output_path):
        print(f"Mô hình đã tồn tại tại: {output_path}")
        return True
    
    try:
        if source_type == "gdrive" and source_id:
            # Tải từ Google Drive
            print(f"Đang tải mô hình từ Google Drive với ID: {source_id}")
            url = f'https://drive.google.com/uc?id={source_id}'
            gdown.download(url, output_path, quiet=False)
        elif source_type == "url" and source_id:
            # Tải từ URL
            print(f"Đang tải mô hình từ URL: {source_id}")
            response = requests.get(source_id, stream=True)
            response.raise_for_status()  # Kiểm tra lỗi
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            # Tạo mô hình giả để demo
            print("Không có nguồn tải mô hình hợp lệ, tạo mô hình giả để demo...")
            return create_dummy_model(output_path)
            
        print(f"Đã tải/tạo mô hình thành công và lưu tại: {output_path}")
        return True
    except Exception as e:
        print(f"Lỗi khi tải mô hình từ nguồn: {e}")
        # Tạo mô hình giả khi có lỗi
        print("Tạo mô hình giả để thay thế...")
        return create_dummy_model(output_path)

# Tạo mô hình giả để demo trong trường hợp không có mô hình thật
def create_dummy_model(output_path):
    """Tạo một mô hình giả với trọng số ngẫu nhiên để demo"""
    # Tạo mạng
    snn = create_network()
    
    # Tạo trọng số ngẫu nhiên cho kết nối
    input_hidden = snn.connections[("input", "hidden")]
    input_hidden.w = 0.1 * torch.rand_like(input_hidden.w)
    
    # Thêm kết nối từ hidden đến output
    hidden_output = Connection(
        source=snn.layers["hidden"],
        target=snn.layers["output"],
        w=0.1 * torch.rand(100, 10)
    )
    snn.add_connection(hidden_output, source="hidden", target="output")
    
    # Lưu dữ liệu mô hình dưới dạng dict thay vì state_dict của mạng
    dummy_data = {
        'time_window': 250,
        'warmup_window': 100,
        'accuracy': 85.5,  # Giá trị giả định
        'class_names': [str(i) for i in range(10)],
        'weights': {
            'input_hidden': input_hidden.w.data.clone().detach().cpu().numpy(),
            'hidden_output': hidden_output.w.data.clone().detach().cpu().numpy()
        },
        'hidden_layer_params': {
            'n': 100,
            'thresh': -55.0,
            'reset': -65.0,
            'rest': -65.0,
            'tau': 10.0
        },
        'output_layer_params': {
            'thresh': -58.0,
            'reset': -65.0,
            'rest': -65.0,
            'tau': 10.0
        }
    }
    
    try:
        # Lưu mô hình
        print(f"Đang lưu mô hình giả vào {output_path}...")
        torch.save(dummy_data, output_path)
        print(f"Đã lưu mô hình giả thành công!")
        return True
    except Exception as e:
        print(f"Lỗi khi lưu mô hình giả: {e}")
        # Tạo thư mục cha nếu không tồn tại
        parent_dir = os.path.dirname(output_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            print(f"Đã tạo thư mục {parent_dir}")
            # Thử lưu lại
            try:
                torch.save(dummy_data, output_path)
                print(f"Đã lưu mô hình giả thành công!")
                return True
            except Exception as e2:
                print(f"Vẫn lỗi khi lưu mô hình giả: {e2}")
                return False
        return False

# Tạo cấu trúc mạng SNN huấn luyện
def create_training_network(n_input=784, n_hidden=100, n_output=10, dt=1.0):
    """Tạo mạng SNN với khả năng học tập sử dụng quy tắc STDP"""
    network = Network(dt=dt)
    
    # Lớp đầu vào
    input_layer = Input(n=n_input, traces=True, trace_tc=20.0)
    
    # Lớp ẩn
    hidden_layer = LIFNodes(
        n=n_hidden, 
        traces=True, 
        trace_tc=20.0,
        thresh=-54.0,
        rest=-65.0,
        reset=-65.0,
        refrac=5,
        tc_decay=100.0
    )
    
    # Lớp đầu ra
    output_layer = LIFNodes(
        n=n_output, 
        traces=True, 
        trace_tc=20.0,
        thresh=-52.0,
        rest=-65.0,
        reset=-65.0,
        refrac=2,
        tc_decay=100.0
    )
    
    # Thêm các lớp vào mạng
    network.add_layer(input_layer, name="input")
    network.add_layer(hidden_layer, name="hidden")
    network.add_layer(output_layer, name="output")
    
    # Kết nối từ đầu vào đến lớp ẩn với khả năng học tập STDP
    input_hidden = Connection(
        source=input_layer,
        target=hidden_layer,
        w=0.01 * torch.randn(n_input, n_hidden),
        update_rule=PostPre,
        nu=(1e-4, 1e-2),  # (nu_pre, nu_post) - tốc độ học
        norm=78.4  # Giá trị chuẩn hóa
    )
    
    # Kết nối từ lớp ẩn đến lớp đầu ra với khả năng học tập STDP
    hidden_output = Connection(
        source=hidden_layer,
        target=output_layer,
        w=0.01 * torch.randn(n_hidden, n_output),
        update_rule=PostPre,
        nu=(1e-4, 1e-2),  # (nu_pre, nu_post) - tốc độ học
        norm=10.0  # Giá trị chuẩn hóa
    )
    
    # Ức chế bên trong lớp ẩn và lớp đầu ra (tránh quá nhiều nơ-ron phát xung cùng lúc)
    hidden_inhibition = Connection(
        source=hidden_layer,
        target=hidden_layer,
        w=-0.1 * (torch.ones(n_hidden, n_hidden) - torch.diag(torch.ones(n_hidden))),
        learning=False  # Không học tập
    )
    
    output_inhibition = Connection(
        source=output_layer,
        target=output_layer,
        w=-0.1 * (torch.ones(n_output, n_output) - torch.diag(torch.ones(n_output))),
        learning=False  # Không học tập
    )
    
    # Thêm các kết nối vào mạng
    network.add_connection(input_hidden, source="input", target="hidden")
    network.add_connection(hidden_output, source="hidden", target="output")
    network.add_connection(hidden_inhibition, source="hidden", target="hidden")
    network.add_connection(output_inhibition, source="output", target="output")
    
    # Thêm các monitor để theo dõi hoạt động của mạng
    network.add_monitor(Monitor(input_layer, state_vars=["s"]), name="input_monitor")
    network.add_monitor(Monitor(hidden_layer, state_vars=["s", "v"]), name="hidden_monitor")
    network.add_monitor(Monitor(output_layer, state_vars=["s", "v"]), name="output_monitor")
    
    return network

# Tạo cấu trúc mạng SNN
def create_network(n_input=784, n_hidden=100, n_output=10):
    """Tạo mạng với cấu trúc tương tự mô hình đã huấn luyện"""
    network = Network(dt=1.0)
    
    # Lớp đầu vào
    input_layer = Input(n=n_input, traces=True)
    
    # Lớp ẩn với ngưỡng thấp hơn để dễ phát xung
    hidden_layer = LIFNodes(n=n_hidden, thresh=-55.0, reset=-65.0, rest=-65.0, tau=10.0, traces=True)
    
    # Lớp đầu ra
    output_layer = LIFNodes(n=n_output, thresh=-58.0, reset=-65.0, rest=-65.0, tau=10.0, traces=True)
    
    # Thêm các lớp vào mạng
    network.add_layer(input_layer, name="input")
    network.add_layer(hidden_layer, name="hidden")
    network.add_layer(output_layer, name="output")
    
    # Kết nối từ đầu vào đến lớp ẩn
    input_hidden = Connection(
        source=input_layer,
        target=hidden_layer,
        w=torch.zeros(n_input, n_hidden)
    )
    network.add_connection(input_hidden, source="input", target="hidden")
    
    # Ức chế bên giữa các nơ-ron đầu ra (mức nhẹ)
    inhibition = torch.ones(n_output, n_output) - torch.diag(torch.ones(n_output))
    output_recurrent = Connection(
        source=output_layer,
        target=output_layer,
        w=-0.05 * inhibition
    )
    network.add_connection(output_recurrent, source="output", target="output")
    
    # Thêm monitor để theo dõi hoạt động
    network.add_monitor(Monitor(input_layer, state_vars=["s"]), name="input_monitor")
    network.add_monitor(Monitor(hidden_layer, state_vars=["s", "v"]), name="hidden_monitor")
    network.add_monitor(Monitor(output_layer, state_vars=["s", "v"]), name="output_monitor")
    
    return network

# Tải trọng số đã huấn luyện
def load_pretrained_model(model_path, device):
    """Tải trọng số đã huấn luyện vào mạng"""
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(model_path):
        print(f"Không tìm thấy file mô hình tại: {model_path}")
        print("Tạo mô hình giả để demo...")
        # Tạo mô hình giả
        create_dummy_model(model_path)
    
    # Tạo mạng
    snn = create_network()
    
    # Tải checkpoint
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            print("Đã tải checkpoint thành công!")
        else:
            print(f"LỖI: Không thể tải file mô hình từ: {model_path}")
            # Tạo checkpoint giả
            checkpoint = {
                'time_window': 250,
                'warmup_window': 100,
                'class_names': [str(i) for i in range(10)],
                'hidden_layer_params': {
                    'n': 100,
                    'thresh': -55.0,
                    'reset': -65.0,
                    'rest': -65.0,
                    'tau': 10.0
                },
                'output_layer_params': {
                    'thresh': -58.0,
                    'reset': -65.0,
                    'rest': -65.0,
                    'tau': 10.0
                }
            }
        
        # Cố gắng tải trọng số từ checkpoint
        if 'network_state_dict' in checkpoint:
            # Cố gắng tải state_dict
            try:
                snn.load_state_dict(checkpoint['network_state_dict'])
                print("Đã tải network_state_dict thành công!")
            except Exception as e:
                print(f"Lỗi khi tải state_dict: {e}")
                print("Sử dụng trọng số ngẫu nhiên...")
        elif 'weights' in checkpoint:
            # Trực tiếp sử dụng trọng số từ dict weights
            print("Sử dụng trọng số từ dict 'weights'...")
            try:
                # Input -> Hidden
                input_hidden = snn.connections[("input", "hidden")]
                input_hidden_weights = checkpoint['weights'].get('input_hidden')
                if input_hidden_weights is not None:
                    input_hidden.w = torch.tensor(input_hidden_weights, device=device)
                else:
                    input_hidden.w = 0.1 * torch.rand_like(input_hidden.w, device=device)
                
                # Hidden -> Output
                if ("hidden", "output") not in snn.connections:
                    print("Thêm kết nối hidden→output...")
                    hidden_output_weights = checkpoint['weights'].get('hidden_output')
                    if hidden_output_weights is not None:
                        w = torch.tensor(hidden_output_weights, device=device)
                    else:
                        w = 0.1 * torch.rand(100, 10, device=device)
                    
                    hidden_output = Connection(
                        source=snn.layers["hidden"],
                        target=snn.layers["output"],
                        w=w
                    )
                    snn.add_connection(hidden_output, source="hidden", target="output")
                print("Đã áp dụng trọng số thành công!")
            except Exception as e:
                print(f"Lỗi khi áp dụng trọng số từ dict: {e}")
                print("Sử dụng trọng số ngẫu nhiên...")
        else:
            print("Không tìm thấy trọng số trong checkpoint, sử dụng trọng số ngẫu nhiên...")
            # Sử dụng trọng số ngẫu nhiên
            input_hidden = snn.connections[("input", "hidden")]
            input_hidden.w = 0.1 * torch.rand_like(input_hidden.w, device=device)
            
            # Gán trực tiếp trọng số cho hidden_output
            if ("hidden", "output") not in snn.connections:
                print("Thêm kết nối hidden→output...")
                hidden_output = Connection(
                    source=snn.layers["hidden"],
                    target=snn.layers["output"],
                    w=0.1 * torch.rand(100, 10, device=device)
                )
                snn.add_connection(hidden_output, source="hidden", target="output")
        
        # Tối ưu mạng
        snn = optimize_network_for_prediction(snn, checkpoint, device)
        print("Đã tối ưu hóa mạng!")
        
        # Chuyển mạng đến thiết bị
        snn = snn.to(device)
        
        return snn, checkpoint
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        # Tạo mạng mặc định nếu có lỗi
        snn = create_network()
        
        # Thêm kết nối hidden→output vì không có trong mạng ban đầu
        hidden_output = Connection(
            source=snn.layers["hidden"],
            target=snn.layers["output"],
            w=0.1 * torch.rand(100, 10, device=device)
        )
        snn.add_connection(hidden_output, source="hidden", target="output")
        
        # Chuyển mạng đến thiết bị
        snn = snn.to(device)
        
        # Giả lập một checkpoint
        checkpoint = {
            'time_window': 250,
            'warmup_window': 100,
            'class_names': [str(i) for i in range(10)]
        }
        
        # Tối ưu mạng
        snn = optimize_network_for_prediction(snn, checkpoint, device)
        print("Đã tối ưu hóa mạng mặc định!")
        
        return snn, checkpoint

# Bộ mã hóa Poisson và hàm mã hóa
def create_encoder(time_window):
    """Tạo bộ mã hóa Poisson"""
    return PoissonEncoder(time=time_window, dt=1.0)

def encode_to_spikes(data, time_window, device, encoder=None):
    """Mã hóa dữ liệu thành xung"""
    if encoder is None:
        encoder = create_encoder(time_window)
    
    with torch.no_grad():
        data_cpu = data.detach().cpu()
        spike_data = encoder(data_cpu)
        return spike_data.to(device)

# Tối ưu mạng cho dự đoán
def optimize_network_for_prediction(snn, checkpoint, device):
    """Tối ưu hóa mạng SNN cho dự đoán chính xác nhất"""
    # Đảm bảo mạng ở chế độ đánh giá (tắt learning)
    for conn_name, connection in snn.connections.items():
        if hasattr(connection, 'update_rule'):
            connection.update_rule = None
        connection.learning = False
    
    # Điều chỉnh ngưỡng phát xung (nếu cần)
    if "hidden" in snn.layers:
        snn.layers["hidden"].thresh = torch.tensor(-55.0)
        snn.layers["hidden"].reset = torch.tensor(-65.0)
        snn.layers["hidden"].rest = torch.tensor(-65.0)
    
    if "output" in snn.layers:
        snn.layers["output"].thresh = torch.tensor(-58.0)
        snn.layers["output"].reset = torch.tensor(-65.0)
        snn.layers["output"].rest = torch.tensor(-65.0)
    
    return snn

# Hàm dự đoán với SNN
def predict_with_snn(network, img, device, checkpoint=None):
    """Dự đoán nhãn của một ảnh sử dụng SNN đã huấn luyện"""
    # Chuẩn bị tham số
    time_window = checkpoint.get("time_window", 250) if checkpoint else 250
    warmup_window = checkpoint.get("warmup_window", 100) if checkpoint else 50
    
    # Chuyển mạng sang chế độ đánh giá
    network.train(False)
    
    # Chuẩn bị dữ liệu
    with torch.no_grad():
        # Tăng độ tương phản cho ảnh đầu vào
        img = torch.clamp(img * 1.5, 0, 1.0)
        
        # Tạo đầu vào xung
        encoder = create_encoder(time_window + warmup_window)
        spike_data = encode_to_spikes(img, time_window + warmup_window, device, encoder)
        
        # Reset trạng thái mạng
        network.reset_state_variables()
        
        # Warm-up phase: Chạy mạng với một khoảng thời gian để ổn định
        if warmup_window > 0:
            warmup_inputs = {"input": spike_data[:warmup_window]}
            network.run(inputs=warmup_inputs, time=warmup_window)
            
            # Reset monitors sau warm-up
            for monitor in network.monitors:
                network.monitors[monitor].reset_state_variables()
        
        # Dự đoán phase: Chạy mạng với dữ liệu đầu vào thực tế
        prediction_inputs = {"input": spike_data[warmup_window:]}
        network.run(inputs=prediction_inputs, time=time_window)
        
        # Lấy dữ liệu từ monitors
        spikes = {
            "input": network.monitors["input_monitor"].get("s"),
            "hidden": network.monitors["hidden_monitor"].get("s"),
            "output": network.monitors["output_monitor"].get("s")
        }
        
        voltage = {
            "hidden": network.monitors["hidden_monitor"].get("v"),
            "output": network.monitors["output_monitor"].get("v")
        }
        
        # Xác định phương pháp dự đoán:
        # 1. Nếu có label_assignments (đã học), sử dụng nó
        if 'label_assignments' in checkpoint:
            label_assignments = checkpoint['label_assignments']
            predicted = predict_from_spikes(spikes, label_assignments)
        else:
            # 2. Nếu không, đơn giản tính tổng số xung ở mỗi nơ-ron đầu ra
            output_spikes = spikes["output"].sum(0)  # Tổng qua thời gian: [time, output] -> [output]
            predicted = torch.argmax(output_spikes).item()
        
        # Tính độ tin cậy
        confidence = calculate_confidence(spikes, voltage, predicted)
        
        # Trả về dự đoán, độ tin cậy, và giá trị spike/voltage để trực quan hóa
        spike_values = spikes["output"].sum(0).cpu().numpy()  # Số spike ở mỗi nơ-ron đầu ra
        voltage_values = voltage["output"].mean(0).cpu().numpy()  # Điện thế trung bình
        
        return predicted, confidence, spike_values, voltage_values

# Tính độ tin cậy từ spike và voltage
def calculate_confidence(spikes, voltage, predicted_class):
    """Tính độ tin cậy dự đoán dựa trên cả spike và voltage"""
    output_spikes = spikes["output"].sum(0)  # [time, output] -> [output]
    output_voltage = voltage["output"].mean(0)  # [time, output] -> [output]
    
    # Chuẩn hóa
    if torch.sum(output_spikes) > 0:
        # Nếu có spike, tính toán dựa trên spike
        spike_probs = output_spikes / (torch.sum(output_spikes) + 1e-10)
        spike_confidence = spike_probs[predicted_class].item()
    else:
        # Nếu không có spike, đặt độ tin cậy spike bằng 0
        spike_confidence = 0.0
    
    # Chuẩn hóa điện thế
    min_v = torch.min(output_voltage)
    norm_voltage = (output_voltage - min_v) / (torch.max(output_voltage) - min_v + 1e-10)
    voltage_probs = norm_voltage / (torch.sum(norm_voltage) + 1e-10)
    voltage_confidence = voltage_probs[predicted_class].item()
    
    # Kết hợp cả hai độ tin cậy với trọng số
    if spike_confidence > 0:
        # Nếu có spike, ưu tiên spike hơn
        combined_confidence = 0.7 * spike_confidence + 0.3 * voltage_confidence
    else:
        # Nếu không có spike, chỉ dựa vào điện thế
        combined_confidence = voltage_confidence
    
    return combined_confidence

# Trực quan hóa hoạt động
def visualize_prediction(snn, img, label, predicted, confidence, spike_values, voltage_values, save_path=None):
    """Trực quan hóa dự đoán và hoạt động nơ-ron"""
    plt.figure(figsize=(12, 10))
    
    # Hình ảnh gốc
    plt.subplot(2, 2, 1)
    plt.imshow(img.reshape(28, 28).cpu().numpy(), cmap='gray')
    title_color = 'green' if predicted == label else 'red'
    plt.title(f'Thực tế: {label}, Dự đoán: {predicted}\nĐộ tin cậy: {confidence:.2f}', 
              color=title_color)
    plt.axis('off')
    
    # Spike ở lớp đầu ra
    plt.subplot(2, 2, 2)
    plt.bar(range(10), spike_values)
    plt.xticks(range(10))
    plt.title('Số xung ở mỗi lớp đầu ra')
    plt.xlabel('Lớp')
    plt.ylabel('Số xung')
    # Đánh dấu lớp thực tế
    plt.axvline(x=label, color='red', linestyle='--')
    plt.text(label, max(spike_values)/2 if max(spike_values) > 0 else 0.5, 
             'Lớp thực tế', rotation=90, verticalalignment='center')
    
    # Điện thế ở lớp đầu ra
    plt.subplot(2, 2, 3)
    plt.bar(range(10), voltage_values)
    plt.xticks(range(10))
    plt.title('Điện thế trung bình ở mỗi lớp đầu ra')
    plt.xlabel('Lớp')
    plt.ylabel('Điện thế')
    # Đánh dấu lớp thực tế
    plt.axvline(x=label, color='red', linestyle='--')
    
    # Dự đoán so với thực tế
    plt.subplot(2, 2, 4)
    plt.scatter(label, predicted, color=title_color, s=100)
    plt.plot([0, 9], [0, 9], 'k--', alpha=0.5)  # Đường chéo lý tưởng
    plt.xlim(-0.5, 9.5)
    plt.ylim(-0.5, 9.5)
    plt.title('Dự đoán vs Thực tế')
    plt.xlabel('Thực tế')
    plt.ylabel('Dự đoán')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        try:
            # Đảm bảo thư mục tồn tại
            save_dir = os.path.dirname(os.path.abspath(save_path))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                print(f"Đã tạo thư mục cho file đầu ra: {save_dir}")
            
            print(f"Đang lưu biểu đồ vào: {os.path.abspath(save_path)}")
            plt.savefig(save_path)
            
            # Kiểm tra xem file đã được tạo chưa
            if os.path.exists(save_path):
                print(f"Đã lưu biểu đồ trực quan hóa thành công vào {save_path}")
            else:
                print(f"LỖI: File không tồn tại sau khi lưu: {save_path}")
        except Exception as e:
            print(f"LỖI khi lưu biểu đồ: {e}")
    else:
        plt.show()
    
    plt.close()

# Tổng hợp hàm đánh giá mô hình
def evaluate_model(snn, dataloader, device, checkpoint=None, num_samples=100, visualize=True):
    """Đánh giá mô hình trên tập dữ liệu và trực quan hóa kết quả"""
    correct = 0
    total = 0
    confidences = []
    
    # Lấy số mẫu dataloader
    actual_num_samples = min(num_samples, len(dataloader.dataset))
    sample_indices = np.random.choice(len(dataloader.dataset), actual_num_samples, replace=False)
    
    # Danh sách lưu kết quả
    results = []
    
    print(f"Đánh giá mô hình trên {actual_num_samples} mẫu...")
    for idx in tqdm(sample_indices):
        img, label = dataloader.dataset[idx]
        img_tensor = img.view(-1).to(device)
        
        # Dự đoán
        predicted, confidence, spike_values, voltage_values = predict_with_snn(
            snn, img_tensor, device, checkpoint
        )
        
        is_correct = (predicted == label)
        total += 1
        correct += is_correct
        confidences.append(confidence)
        
        # Lưu kết quả
        results.append({
            'index': idx,
            'label': label,
            'prediction': predicted,
            'confidence': confidence,
            'is_correct': is_correct
        })
        
        # Trực quan hóa một số mẫu
        if visualize and (total <= 5 or (is_correct is False and total <= 10)):
            save_path = os.path.join(output_dir, f"prediction_sample_{idx}.png")
            print(f"Tạo trực quan hóa cho mẫu {idx} (label={label}, prediction={predicted})")
            visualize_prediction(
                snn, img_tensor, label, predicted, confidence,
                spike_values, voltage_values, save_path
            )
    
    # Tính độ chính xác và độ tin cậy trung bình
    accuracy = 100 * correct / total
    avg_confidence = sum(confidences) / len(confidences)
    
    print(f"Độ chính xác: {accuracy:.2f}% trên {total} mẫu kiểm tra")
    print(f"Độ tin cậy trung bình: {avg_confidence:.2f}")
    
    # Trực quan hóa tổng hợp
    if visualize:
        print("Tạo trực quan hóa tổng hợp kết quả đánh giá...")
        try:
            plt.figure(figsize=(10, 6))
            
            # Biểu đồ độ chính xác và độ tin cậy
            plt.subplot(1, 2, 1)
            plt.bar(['Chính xác', 'Tin cậy'], [accuracy/100, avg_confidence])
            plt.ylim(0, 1)
            plt.title('Độ chính xác và độ tin cậy')
            
            # Biểu đồ phân bố độ tin cậy
            plt.subplot(1, 2, 2)
            correct_conf = [r['confidence'] for r in results if r['is_correct']]
            wrong_conf = [r['confidence'] for r in results if not r['is_correct']]
            
            if correct_conf:
                plt.hist(correct_conf, alpha=0.5, label='Đúng', bins=10, range=(0, 1))
            if wrong_conf:
                plt.hist(wrong_conf, alpha=0.5, label='Sai', bins=10, range=(0, 1))
            
            plt.title('Phân bố độ tin cậy')
            plt.xlabel('Độ tin cậy')
            plt.ylabel('Số lượng mẫu')
            plt.legend()
            
            plt.tight_layout()
            summary_path = os.path.join(output_dir, "evaluation_summary.png")
            print(f"Đang lưu biểu đồ tổng hợp vào: {os.path.abspath(summary_path)}")
            plt.savefig(summary_path)
            
            if os.path.exists(summary_path):
                print(f"Đã lưu biểu đồ tổng hợp đánh giá thành công vào {summary_path}")
            else:
                print(f"LỖI: File tổng hợp không tồn tại sau khi lưu: {summary_path}")
        except Exception as e:
            print(f"LỖI khi tạo biểu đồ tổng hợp: {e}")
        finally:
            plt.close()
    
    return accuracy, avg_confidence, results

# Hàm dùng để xử lý mỗi batch trong quá trình huấn luyện
def train_batch(
    network, 
    dataloader, 
    optimizer=None, 
    time_window=100, 
    device="cpu",
    progress_bar=None,
    label_assignments=None,
    assignments_step=250,
    current_examples=0,
    max_examples=60000
):
    """Huấn luyện SNN trên một batch dữ liệu"""
    # Khởi tạo label_assignments nếu chưa có
    if label_assignments is None:
        n_output = network.layers["output"].n
        label_assignments = torch.zeros(10, n_output, device=device)
        print(f"Khởi tạo label_assignments trong train_batch với shape: {label_assignments.shape}")
    
    # Lưu trữ spikes và voltage cho mỗi mẫu
    sample_spikes = {}
    sample_voltage = {}
    correct = 0
    total = 0
    
    # Tạo bộ mã hóa
    encoder = create_encoder(time_window)
    
    # Lặp qua từng batch dữ liệu
    for batch_idx, (imgs, labels) in enumerate(dataloader):
        # Kiểm tra xem đã quá số mẫu tối đa chưa
        if current_examples >= max_examples:
            break
        
        batch_size = imgs.size(0)
        current_examples += batch_size
        
        # Chuẩn bị dữ liệu
        imgs = imgs.view(batch_size, -1).to(device)  # Flattened: [batch_size, 784]
        labels = labels.to(device)
        
        # Mã hóa thành xung cho toàn bộ batch
        spike_inputs = encode_to_spikes(imgs, time_window, device, encoder)
        
        # Lặp qua từng mẫu trong batch
        for i in range(batch_size):
            # Reset network cho mỗi mẫu
            network.reset_state_variables()
            
            # Tạo đầu vào xung cho mẫu hiện tại
            inputs = {"input": spike_inputs[:, i].unsqueeze(1)}
            
            # Chạy mạng với đầu vào hiện tại
            network.run(inputs=inputs, time=time_window)
            
            # Lấy dữ liệu từ monitors
            sample_spikes = {
                "input": network.monitors["input_monitor"].get("s"),
                "hidden": network.monitors["hidden_monitor"].get("s"),
                "output": network.monitors["output_monitor"].get("s")
            }
            
            sample_voltage = {
                "hidden": network.monitors["hidden_monitor"].get("v"),
                "output": network.monitors["output_monitor"].get("v")
            }
            
            # Cập nhật gán nhãn sau mỗi interval nhất định
            if (current_examples - batch_size + i + 1) % assignments_step == 0:
                print(f"\nCập nhật gán nhãn sau {current_examples} mẫu...")
                # In thông tin về label_assignments và sample_spikes trước khi cập nhật
                output_spikes = sample_spikes["output"].sum(0)
                print(f"Shape của output_spikes: {output_spikes.shape}")
                print(f"Shape của label_assignments: {label_assignments.shape}")
                print(f"Shape của label_assignments[{labels[i].item()}]: {label_assignments[labels[i].item()].shape}")
                
                # Tính toán gán nhãn mới dựa trên hoạt động lớp đầu ra
                label_assignments = update_label_assignments(
                    sample_spikes, labels[i].item(), label_assignments
                )
            
            # Tính độ chính xác
            predicted = predict_from_spikes(sample_spikes, label_assignments)
            correct += int(predicted == labels[i].item())
            total += 1
            
            # Cập nhật progress bar
            if progress_bar is not None and i % 10 == 0:
                progress_bar.set_description(
                    f"Train: {current_examples}/{max_examples} | Acc: {100 * correct / total:.2f}%"
                )
                # Cập nhật progress bar
                try:
                    progress_bar.update(1)  # Chỉ cập nhật 1 đơn vị
                except Exception as e:
                    # Nếu có lỗi (một số tqdm không hỗ trợ update với tham số)
                    pass
                
    # Trả về độ chính xác và assignments
    acc = correct / total if total > 0 else 0
    return acc, label_assignments, current_examples

# Hàm cập nhật label assignments dựa trên spike hiện tại
def update_label_assignments(spikes, label, label_assignments):
    """Cập nhật ma trận gán nhãn dựa trên spike"""
    output_spikes = spikes["output"].sum(0)  # [time, output_size] -> [output_size]
    
    # Cập nhật assignments cho label hiện tại
    # Đảm bảo cả hai tensor có cùng kích thước
    if output_spikes.dim() != label_assignments[label].dim():
        # Nếu output_spikes là vector 1D, chuyển đổi nó thành kích thước phù hợp
        if output_spikes.dim() == 1:
            # Đảm bảo output_spikes có hình dạng phù hợp với label_assignments[label]
            output_spikes = output_spikes.view(*label_assignments[label].shape)
        else:
            # Hoặc đưa label_assignments[label] về cùng hình dạng với output_spikes
            label_assignments[label] = label_assignments[label].reshape(output_spikes.shape)
    
    # Đảm bảo không có vấn đề với thiết bị
    output_spikes = output_spikes.to(label_assignments.device)
    
    # Sau khi điều chỉnh kích thước, thực hiện phép cộng
    try:
        label_assignments[label] += output_spikes
    except RuntimeError as e:
        print(f"Lỗi khi cập nhật label_assignments: {e}")
        print(f"Shape của label_assignments[{label}]: {label_assignments[label].shape}")
        print(f"Shape của output_spikes: {output_spikes.shape}")
        
        # Cách giải quyết thay thế: tạo lại tensor với kích thước phù hợp
        for i in range(output_spikes.numel()):
            if i < label_assignments[label].numel():
                # Cập nhật từng phần tử một
                label_assignments[label].view(-1)[i] += output_spikes.view(-1)[i]
    
    return label_assignments

# Hàm dự đoán nhãn từ spike
def predict_from_spikes(spikes, label_assignments, voting_scheme="activity"):
    """Dự đoán nhãn dựa trên spike và assignments hiện tại"""
    output_spikes = spikes["output"].sum(0)  # [time, output_size] -> [output_size]
    
    # Đảm bảo output_spikes có hình dạng phù hợp
    if output_spikes.dim() > 1 and output_spikes.shape[0] == 1:
        output_spikes = output_spikes.squeeze(0)
    
    # Xếp hạng labels dựa trên voting
    if voting_scheme == "activity":
        # Sử dụng hoạt động trung bình làm cơ sở cho voting
        predictions = torch.zeros(10, device=output_spikes.device)
        for i in range(10):
            # Đảm bảo kích thước phù hợp trước khi tính toán
            if label_assignments[i].dim() > 1 and output_spikes.dim() == 1:
                # Nếu label_assignments có nhiều chiều hơn, sử dụng reshape
                scores = (label_assignments[i].reshape(-1) * output_spikes).sum()
            elif label_assignments[i].dim() == 1 and output_spikes.dim() > 1:
                # Nếu output_spikes có nhiều chiều hơn, cũng sử dụng reshape
                scores = (label_assignments[i] * output_spikes.reshape(-1)).sum()
            else:
                # Nếu cả hai có cùng số chiều, thực hiện phép nhân trực tiếp
                scores = (label_assignments[i] * output_spikes).sum()
            predictions[i] = scores
    else:
        # Sử dụng số lượng neurons active làm cơ sở cho voting
        predictions = torch.zeros(10, device=output_spikes.device)
        for i in range(10):
            # Dùng phiên bản đơn giản hơn để tránh các vấn đề với kích thước
            active_mask = (torch.argmax(label_assignments, dim=0) == i).float()
            if active_mask.shape != output_spikes.shape:
                # Điều chỉnh kích thước nếu cần
                if active_mask.dim() > output_spikes.dim():
                    active_mask = active_mask.view(output_spikes.shape)
                else:
                    output_spikes_reshaped = output_spikes.view(active_mask.shape)
                    predictions[i] = torch.sum(output_spikes_reshaped * active_mask)
                    continue
            predictions[i] = torch.sum(output_spikes * active_mask)
    
    # Lấy predicted label
    return torch.argmax(predictions).item()

# Hàm huấn luyện toàn bộ mô hình
def train_model(
    snn=None, 
    trainloader=None,
    testloader=None,
    device="cpu", 
    epochs=1, 
    batch_size=1, 
    time_window=100,
    max_examples=60000,
    save_path=None
):
    """Huấn luyện mô hình SNN"""
    # Tạo mạng nếu chưa có
    if snn is None:
        snn = create_training_network()
        
    # Chuyển sang device
    snn = snn.to(device)
    
    # Nếu không có dataloader, tạo mới
    if trainloader is None:
        # Tạo thư mục dữ liệu
        data_dir = "./data/mnist_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Tải dữ liệu
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True
        )
        
        testset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False
        )
    
    # Khởi tạo label assignments
    n_output = snn.layers["output"].n
    print(f"Số lượng neuron đầu ra: {n_output}")
    
    # Khởi tạo đúng kích thước cho label_assignments
    # Sử dụng kích thước [10, n_output] phù hợp với cấu trúc neuron đầu ra
    label_assignments = torch.zeros(10, n_output, device=device)
    
    # In thông tin về label_assignments để debug
    print(f"Khởi tạo label_assignments với shape: {label_assignments.shape}")
    
    # Theo dõi tiến trình
    print(f"Bắt đầu huấn luyện trên {epochs} epochs...")
    start_time = time.time()
    
    # Hủy các update_rule trong network.connections mà có giá trị None
    # Đây là workaround cho lỗi AttributeError: 'NoneType' object has no attribute 'update'
    print("Kiểm tra và sửa các update_rule là None trong connections...")
    for conn_name, connection in snn.connections.items():
        if hasattr(connection, 'update_rule') and connection.update_rule is None:
            print(f"Connection {conn_name} có update_rule là None, đặt learning=False")
            connection.learning = False
    
    # Huấn luyện các epochs
    train_acc_history = []
    test_acc_history = []
    current_examples = 0
    
    for epoch in range(epochs):
        # Huấn luyện trên tập train
        with tqdm(total=max_examples, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            # Huấn luyện trên từng batch
            train_acc, label_assignments, current_examples = train_batch(
                network=snn,
                dataloader=trainloader,
                time_window=time_window,
                device=device,
                progress_bar=pbar,
                label_assignments=label_assignments,
                assignments_step=1000,
                current_examples=current_examples,
                max_examples=max_examples
            )
            
        train_acc_history.append(train_acc)
        print(f"Epoch {epoch+1}/{epochs} - Độ chính xác train: {train_acc:.4f}")
        
        # Tính độ chính xác trên tập test
        if testloader is not None:
            print(f"Đánh giá trên tập test...")
            # Tắt chế độ học tập
            for conn_name in snn.connections:
                if hasattr(snn.connections[conn_name], "update_rule"):
                    snn.connections[conn_name].update_rule = None
            
            # Đánh giá trên tập test
            test_acc, _, _ = train_batch(
                network=snn,
                dataloader=testloader,
                time_window=time_window,
                device=device,
                label_assignments=label_assignments,
                max_examples=min(10000, len(testloader.dataset))
            )
            
            # Bật lại chế độ học tập
            for conn_name in snn.connections:
                if conn_name[0] == "input" and conn_name[1] == "hidden":
                    snn.connections[conn_name].update_rule = PostPre
                elif conn_name[0] == "hidden" and conn_name[1] == "output":
                    snn.connections[conn_name].update_rule = PostPre
            
            test_acc_history.append(test_acc)
            print(f"Epoch {epoch+1}/{epochs} - Độ chính xác test: {test_acc:.4f}")
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Tổng thời gian huấn luyện: {training_time:.2f} giây")
    
    # Lưu mô hình
    if save_path is not None:
        print(f"Lưu mô hình vào {save_path}...")
        # Lưu trạng thái của mô hình
        save_model(snn, label_assignments, train_acc_history, test_acc_history, save_path)
    
    return snn, label_assignments, train_acc_history, test_acc_history

# Hàm lưu mô hình
def save_model(snn, label_assignments, train_acc_history, test_acc_history, path):
    """Lưu mô hình SNN đã huấn luyện"""
    try:
        # Tạo dict chứa trạng thái mô hình
        model_state = {
            'network_state_dict': snn.state_dict(),
            'label_assignments': label_assignments,
            'train_acc_history': train_acc_history,
            'test_acc_history': test_acc_history,
            'time_window': 100,  # Mặc định
            'warmup_window': 20,  # Mặc định
            'class_names': [str(i) for i in range(10)],
            'accuracy': test_acc_history[-1] if test_acc_history else train_acc_history[-1] if train_acc_history else 0.0,
            'hidden_layer_params': {
                'n': snn.layers["hidden"].n,
                'thresh': float(snn.layers["hidden"].thresh),
                'reset': float(snn.layers["hidden"].reset),
                'rest': float(snn.layers["hidden"].rest),
                'tau': float(snn.layers["hidden"].tc_decay)
            },
            'output_layer_params': {
                'thresh': float(snn.layers["output"].thresh),
                'reset': float(snn.layers["output"].reset),
                'rest': float(snn.layers["output"].rest),
                'tau': float(snn.layers["output"].tc_decay)
            }
        }
        
        # Lưu mô hình
        torch.save(model_state, path)
        print(f"Đã lưu mô hình thành công vào {path}")
        return True
    except Exception as e:
        print(f"Lỗi khi lưu mô hình: {e}")
        # Thử tạo thư mục cha nếu nó không tồn tại
        try:
            parent_dir = os.path.dirname(path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
                print(f"Đã tạo thư mục {parent_dir}")
                # Thử lưu lại
                torch.save(model_state, path)
                print(f"Đã lưu mô hình thành công vào {path}")
                return True
        except Exception as e2:
            print(f"Vẫn lỗi khi lưu mô hình: {e2}")
            return False

# Hàm chính
def main():
    print("\n=== CHƯƠNG TRÌNH SNN CHO MNIST ===\n")
    
    start_time = time.time()
    
    # Tạo thư mục dữ liệu
    data_dir = "./data/mnist_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Tải dữ liệu MNIST
    print("Tải dữ liệu MNIST...")
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    
    testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True)
    
    # Đường dẫn đến mô hình
    model_path = os.path.join(base_path, "snn_mnist_model.pt")
    
    # Thiết lập giá trị mặc định
    train_mode = True  # Mặc định là huấn luyện
    
    # Kiểm tra tham số dòng lệnh
    import sys
    force_train = False
    force_predict = False
    
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "train":
            train_mode = True
            force_train = True
            print("Chế độ huấn luyện tự động được kích hoạt từ tham số dòng lệnh.")
        elif sys.argv[1].lower() == "predict":
            train_mode = False
            force_predict = True
            print("Chế độ dự đoán tự động được kích hoạt từ tham số dòng lệnh.")
    else:
        # Hỏi người dùng muốn huấn luyện hay dự đoán
        try:
            if not force_train and not force_predict:
                user_choice = input("Bạn muốn huấn luyện mô hình mới (T) hay chỉ sử dụng mô hình có sẵn (F)? [T/F]: ").strip().upper()
                train_mode = user_choice == "T" or user_choice == ""
        except Exception as e:
            print(f"Không thể nhận input: {e}, mặc định sẽ huấn luyện mô hình.")
            train_mode = True
    
    if train_mode:
        print("\n=== CHẾ ĐỘ HUẤN LUYỆN ===")
        # Hỏi số lượng mẫu huấn luyện
        max_examples = 5000  # Mặc định
        epochs = 1  # Mặc định
        
        if not force_train:
            try:
                max_examples = int(input("Số lượng mẫu huấn luyện (tối đa 60000, mặc định 5000): ") or "5000")
                max_examples = min(max(max_examples, 1000), 60000)
            except:
                print("Lỗi khi nhập số mẫu. Sử dụng 5000 mẫu.")
            
            # Hỏi số lượng epochs
            try:
                epochs = int(input("Số lượng epochs (mặc định 1): ") or "1")
                epochs = max(epochs, 1)
            except:
                print("Lỗi khi nhập epochs. Sử dụng 1 epoch.")
        else:
            print(f"Sử dụng cấu hình mặc định: {max_examples} mẫu, {epochs} epoch")
        
        # Huấn luyện mô hình
        print(f"\nBắt đầu huấn luyện với {max_examples} mẫu trong {epochs} epochs...")
        snn, label_assignments, train_acc_history, test_acc_history = train_model(
            snn=None,
            trainloader=trainloader,
            testloader=testloader,
            device=device,
            epochs=epochs,
            batch_size=1,
            time_window=100,
            max_examples=max_examples,
            save_path=model_path
        )
        
        # Visualize training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_acc_history, 'b-', label='Train Accuracy')
        if test_acc_history:
            plt.plot(test_acc_history, 'r-', label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        history_path = os.path.join(output_dir, "training_history.png")
        plt.savefig(history_path)
        print(f"Đã lưu lịch sử huấn luyện vào {history_path}")
        plt.close()
        
        # Đánh giá mô hình dựa trên weights đã học
        print("\nĐánh giá mô hình SNN đã huấn luyện...")
        checkpoint = {
            'time_window': 100,
            'warmup_window': 20,
            'class_names': [str(i) for i in range(10)],
            'label_assignments': label_assignments
        }
    else:
        print("\n=== CHẾ ĐỘ DỰ ĐOÁN ===")
        # Tải mô hình đã huấn luyện sẵn hoặc tạo mô hình giả
        # 1. Đầu tiên tìm mô hình đã lưu
        if not os.path.exists(model_path):
            # Không tìm thấy, thử tải từ nguồn
            source_type = "url"  # gdrive hoặc url
            source_id = None  # ID Drive hoặc URL
            
            # Nếu bạn có URL hoặc ID Google Drive, hãy thay thế các giá trị ở đây:
            # source_type = "gdrive"
            # source_id = "YOUR_GOOGLE_DRIVE_ID"
            # HOẶC:
            # source_type = "url"
            # source_id = "YOUR_DOWNLOAD_URL"
            
            success = download_pretrained_model(model_path, source_type, source_id)
            if not success:
                print("Không thể tải hoặc tạo mô hình. Sử dụng mô hình tạm thời.")
        
        # Tải mô hình
        print("\nTải mô hình SNN...")
        snn, checkpoint = load_pretrained_model(model_path, device)
    
    # Đánh giá mô hình (giảm số mẫu để chạy nhanh hơn)
    print("\nĐánh giá mô hình SNN...")
    accuracy, avg_confidence, results = evaluate_model(
        snn, testloader, device, checkpoint, num_samples=20, visualize=True
    )
    
    # Tạo một grid các dự đoán hình ảnh
    print("\nTạo trực quan hóa grid dự đoán...")
    images, labels = next(iter(testloader))
    
    try:
        plt.figure(figsize=(15, 12))
        for i in range(min(16, len(images))):
            img = images[i].view(-1).to(device)
            label = labels[i].item()
            
            predicted, confidence, _, _ = predict_with_snn(snn, img, device, checkpoint)
            
            plt.subplot(4, 4, i+1)
            plt.imshow(img.reshape(28, 28).cpu(), cmap='gray')
            
            title_color = 'green' if predicted == label else 'red'
            plt.title(f'Thực tế: {label}\nDự đoán: {predicted}\nĐộ tin cậy: {confidence:.2f}', 
                    color=title_color, fontsize=10)
            plt.axis('off')
        
        plt.tight_layout()
        predictions_path = os.path.join(output_dir, "grid_predictions.png")
        print(f"Đang lưu biểu đồ grid dự đoán vào: {os.path.abspath(predictions_path)}")
        plt.savefig(predictions_path)
        
        if os.path.exists(predictions_path):
            print(f"Đã lưu biểu đồ dự đoán tổng hợp thành công vào {predictions_path}")
        else:
            print(f"LỖI: File grid dự đoán không tồn tại sau khi lưu: {predictions_path}")
    except Exception as e:
        print(f"LỖI khi tạo grid dự đoán: {e}")
    finally:
        plt.close()
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTổng thời gian thực thi: {execution_time:.2f} giây")
    print("\nĐường dẫn tới thư mục kết quả: {0}".format(os.path.abspath(output_dir)))
    print("=== HOÀN THÀNH ===")

if __name__ == "__main__":
    main()