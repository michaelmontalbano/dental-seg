FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /opt/ml/code

# Install Python and pip (with updated package lists and fix-missing flag)
RUN apt-get update && apt-get install -y --fix-missing python3-pip
RUN pip3 install --upgrade pip

# Copy requirements first and install (for better caching)
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Copy your files including the train wrapper script and fixed MedMamba.py
COPY train.py launcher.py train MedMamba.py ./

# Make the train script executable
RUN chmod +x train

# Ensure train script is in PATH
RUN ln -s /opt/ml/code/train /usr/local/bin/train

# No ENTRYPOINT or CMD - let SageMaker run 'train' directly
