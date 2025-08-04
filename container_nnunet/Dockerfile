# Base image with CUDA
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /opt/ml/code

# Install system dependencies and Python 3.9 via deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Copy and install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Set up nnU-Net environment variables
ENV nnUNet_raw="/opt/ml/input/data/training"
ENV nnUNet_preprocessed="/tmp/nnUNet_preprocessed"
ENV nnUNet_results="/opt/ml/model"
ENV nnUNet_n_proc_DA=8

# Copy training script
COPY train.py ./
RUN chmod +x train.py

# SageMaker config
ENV PYTHONUNBUFFERED=1
ENV SAGEMAKER_PROGRAM=train.py

# Create necessary directories
RUN mkdir -p /opt/ml/input/data/training \
    /opt/ml/model \
    /opt/ml/output \
    /tmp/nnUNet_preprocessed

# No ENTRYPOINT or CMD needed; SageMaker handles execution
