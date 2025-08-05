FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
WORKDIR /opt/ml/code

# Add Python, pip, and build tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Add Python build dependencies
RUN pip install --upgrade pip setuptools wheel build

# Create python symlink for compatibility
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements and install in one layer with no-cache-dir
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy training scripts
COPY train.py ./
COPY train /opt/ml/code/train
COPY train.py /opt/ml/code/train.py
COPY preprocessing_config.py ./

# Make the training scripts executable
RUN chmod +x train.py train

# Copy train wrapper to /usr/local/bin where SageMaker expects it
RUN cp /opt/ml/code/train /usr/local/bin/train && \
    chmod +x /usr/local/bin/train

# Tell SageMaker which Python program to run and where to find it
ENV SAGEMAKER_PROGRAM=train.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

# Fixed environment variables (removed extra 'N' from NNUNet)
ENV nnUNet_raw_data_base=/opt/ml/input/data/nnUNet_raw
ENV nnUNet_preprocessed=/opt/ml/input/data/nnUNet_preprocessed
ENV nnUNet_results=/opt/ml/model

# ENTRYPOINT removed to let SageMaker handle argument injection