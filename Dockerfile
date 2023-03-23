ARG CUDA_VERSION=11.4.0

# onnxruntime-gpu requires cudnn
ARG CUDNN_VER=8

# See possible types: https://hub.docker.com/r/nvidia/cuda/tags?page=1&ordering=last_updated
ARG IMAGE_TYPE=runtime
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VER}-${IMAGE_TYPE}-ubuntu20.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive

#install python3.8
RUN apt-get update
RUN apt-get install software-properties-common -y
# RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update
RUN apt-get install python3.8 -y 
RUN apt-get install python3.8-distutils -y
RUN apt-get install python3.8-tk -y
RUN apt-get install curl nano -y
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py
RUN rm get-pip.py

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx && \
    apt-get install libglib2.0-0 && \
    apt-get autoremove -y && \
    apt-get clean -y

# install requirements
RUN apt-get install ffmpeg git -y
WORKDIR /server

COPY requirements.txt .
RUN python3.8 -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 torchtext==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
COPY . /server
RUN alias python=python3.8
RUN echo "alias python=python3.8" >> /root/.bashrc

