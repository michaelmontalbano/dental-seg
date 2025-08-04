FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /opt/ml/code

# Install system dependencies
RUN apt-get update && apt-get install -y --fix-missing \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy and install requirements
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Set up nnU-Net environment variables
ENV nnUNet_raw="/opt/ml/input/data/training"
ENV nnUNet_preprocessed="/tmp/nnUNet_preprocessed"
ENV nnUNet_results="/opt/ml/model"
ENV nnUNet_n_proc_DA=8

# Copy training script
COPY train.py ./

# Make the training script executable
RUN chmod +x train.py

# Set Python unbuffered for better logging
ENV PYTHONUNBUFFERED=1

# SageMaker expects to run train.py
ENV SAGEMAKER_PROGRAM=train.py

# Create necessary directories
RUN mkdir -p /opt/ml/input/data/training \
    /opt/ml/model \
    /opt/ml/output \
    /tmp/nnUNet_preprocessed

# No ENTRYPOINT or CMD - let SageMaker handle execution
