FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA 1
ENV TORCH_CUDA_ARCH_LIST "6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0" 
ENV CUDA_HOME /usr/local/cuda-11.1/
ENV CUDA_VISIBLE_DEVICES=0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3 && apt-get install -y python3-pip && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends wget git libgl1 libglib2.0-0 gcc software-properties-common libopenmpi-dev ninja-build
RUN ln -s /usr/bin/python3 /usr/bin/python