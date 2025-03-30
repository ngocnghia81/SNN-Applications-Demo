#!/usr/bin/env python3
"""
Setup script cho dự án SNN Traffic Monitoring
"""

from setuptools import setup, find_packages

setup(
    name="snn_traffic",
    version="0.1.0",
    description="Ứng dụng giám sát giao thông sử dụng Spiking Neural Networks",
    author="Ngoc Nghia",
    author_email="ngocnghia2004nn@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "numpy==1.23.5",
        "matplotlib>=3.3.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "ultralytics>=8.0.0",
        "tqdm>=4.62.0",
        "scikit-learn<1.3.0",
        "scipy>=1.7.0",
        "seaborn>=0.11.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "snn-traffic=main:main",
        ],
    },
)
