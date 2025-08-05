FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /opt/ml/code

# Install Python and essential packages in one layer to reduce size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create python symlink for compatibility
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements and install in one layer with no-cache-dir
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy training scripts
COPY train.py sagemaker_launcher_nnunet.py ./
COPY train /opt/ml/code/train
COPY preprocessing_config.py dataset_example.json ./

# Make the training scripts executable
RUN chmod +x train.py train

# Copy train wrapper to /usr/local/bin where SageMaker expects it
RUN cp /opt/ml/code/train /usr/local/bin/train && \
    chmod +x /usr/local/bin/train

# Tell SageMaker which Python program to run and where to find it
ENV SAGEMAKER_PROGRAM sagemaker_launcher_nnunet.py
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# Add ENTRYPOINT as fallback
ENTRYPOINT ["python3", "/opt/ml/code/sagemaker_launcher_nnunet.py"]