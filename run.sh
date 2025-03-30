#!/bin/bash

# Script chạy ứng dụng SNN Traffic Monitoring

# Kiểm tra nếu Python venv đã được kích hoạt
if [[ -z "$VIRTUAL_ENV" ]]; then
    if [[ -d "myenv-new" ]]; then
        source myenv-new/bin/activate
        echo "Đã kích hoạt môi trường ảo myenv-new"
    else
        echo "Cảnh báo: Không tìm thấy môi trường ảo. Sẽ sử dụng Python hệ thống."
    fi
fi

# Chạy ứng dụng với các tham số
python main.py "$@"
