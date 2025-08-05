# Multi-stage build for nnU-Net with SageMaker support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

# Install Python 3.10 and tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Point "python3" and "python" to Python 3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python


# Install conda for better dependency management
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    conda clean -tipsy

# Create conda environment
RUN conda create -n nnunet python=3.9 -y
ENV PATH=$CONDA_DIR/envs/nnunet/bin:$PATH

# Install PyTorch with CUDA support
RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install nnU-Net and dependencies
RUN pip install --no-cache-dir \
    nnunetv2==2.2 \
    SimpleITK==2.2.1 \
    nibabel==4.0.2 \
    pydicom==2.3.1 \
    scikit-image==0.20.0 \
    pandas==2.0.3 \
    matplotlib==3.7.2 \
    tqdm==4.65.0 \
    batchgenerators \
    scipy

# Install SageMaker training toolkit
RUN pip install sagemaker-training

# Production stage
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Copy conda environment from builder
COPY --from=builder /opt/conda /opt/conda
ENV PATH=/opt/conda/envs/nnunet/bin:$PATH
ENV CONDA_DEFAULT_ENV=nnunet

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Set up SageMaker directories
RUN mkdir -p \
    /opt/ml/code \
    /opt/ml/input/data \
    /opt/ml/model \
    /opt/ml/output/data \
    /tmp/nnUNet_preprocessed

# Copy training scripts
COPY train.py /opt/ml/code/train.py
COPY sagemaker_launcher_nnunet.py /opt/ml/code/sagemaker_launcher_nnunet.py
COPY requirements.txt /opt/ml/code/requirements.txt
COPY preprocessing_config.py /opt/ml/code/preprocessing_config.py

# Set environment variables
ENV SAGEMAKER_PROGRAM=sagemaker_launcher_nnunet.py
ENV SM_MODEL_DIR=/opt/ml/model
ENV SM_CHANNEL_TRAINING=/opt/ml/input/data/training
ENV SM_CHANNEL_VALIDATION=/opt/ml/input/data/validation
ENV SM_OUTPUT_DATA_DIR=/opt/ml/output/data

# nnU-Net specific environment variables
ENV nnUNet_raw="/opt/ml/input/data/training"
ENV nnUNet_preprocessed="/tmp/nnUNet_preprocessed"
ENV nnUNet_results="/opt/ml/model"
ENV nnUNet_n_proc_DA=16

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

WORKDIR /opt/ml/code

# Create non-root user for security
ARG USERNAME=sagemaker
ARG USERID=1000
RUN useradd --create-home --shell /bin/bash --uid $USERID $USERNAME && \
    chown -R $USERNAME:$USERNAME /opt/ml && \
    chown -R $USERNAME:$USERNAME /tmp/nnUNet_preprocessed
USER $USERNAME

# Verify GPU availability
RUN python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
