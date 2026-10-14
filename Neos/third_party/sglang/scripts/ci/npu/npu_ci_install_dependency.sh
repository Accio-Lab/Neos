#!/bin/bash
set -euo pipefail

PIP_INSTALL="python3 -m pip install --no-cache-dir"
DEVICE_TYPE=$1


                                          
apt update -y && apt install -y \
    unzip \
    build-essential \
    cmake \
    wget \
    curl \
    net-tools \
    zlib1g-dev \
    lld \
    clang \
    locales \
    ccache \
    ca-certificates \
    libgl1 \
    libglib2.0-0
update-ca-certificates
${PIP_INSTALL} --upgrade pip
                                                                    
${PIP_INSTALL} wheel==0.45.1 pybind11 pyyaml decorator scipy attrs psutil


                     
${PIP_INSTALL} memfabric-hybrid==1.0.5


                           
PYTORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
${PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/cpu
PTA_URL="https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.8.0/torch_npu-2.8.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl"
${PIP_INSTALL} ${PTA_URL}


                         
${PIP_INSTALL} triton-ascend


                          
SGLANG_KERNEL_NPU_TAG="2026.02.01.post2"
mkdir sgl-kernel-npu
(cd sgl-kernel-npu && wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.8.0-py311-cann8.5.0-${DEVICE_TYPE}-$(arch).zip \
&& unzip ./sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.8.0-py311-cann8.5.0-${DEVICE_TYPE}-$(arch).zip \
&& ${PIP_INSTALL} ./deep_ep*.whl ./sgl_kernel_npu*.whl \
&& (cd "$(python3 -m pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so))


                  
rm -rf python/pyproject.toml && mv python/pyproject_npu.toml python/pyproject.toml
${PIP_INSTALL} -v -e "python[dev_npu]"
