FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /opt/ml/code

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create python symlink for compatibility
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy training scripts
COPY train.py ./
COPY train /opt/ml/code/train

# Make the training scripts executable
RUN chmod +x train.py train

# Copy train wrapper to /usr/local/bin where SageMaker expects it
RUN cp /opt/ml/code/train /usr/local/bin/train && \
    chmod +x /usr/local/bin/train

# Tell SageMaker which Python program to run and where to find it
ENV SAGEMAKER_PROGRAM=train.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

# Set Python environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
