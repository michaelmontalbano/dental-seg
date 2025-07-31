FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /opt/ml/code

# Optional: install wget if not present
RUN apt-get update && apt-get install -y wget

# Download the MedMamba script
RUN wget https://raw.githubusercontent.com/YubiaoYue/MedMamba/main/MedMamba.py -O /opt/ml/code/MedMamba.py

# Copy training scripts
COPY train.py medmamba.py launcher.py requirements.txt ./

# Install Python and pip dependencies
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Set default command
ENTRYPOINT ["python3", "train.py"]
