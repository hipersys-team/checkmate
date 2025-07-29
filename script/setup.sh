#!/bin/bash

SCRIPT_DIR=$(dirname $(readlink -f $0))
CC=gcc-12
CXX=g++-12
CUDACXX=/usr/local/cuda/bin/nvcc

# Install Miniconda
install_miniconda() {
    mkdir -p $HOME/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.9.2-0-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash $HOME/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf $HOME/miniconda3/miniconda.sh

    export PATH="$HOME/miniconda3/bin:$PATH"

    $HOME/miniconda3/bin/conda init bash
    source ~/.bashrc
    eval "$(conda shell.bash hook)"

    # Create and activate the ckpt environment
    conda create -n ckpt -y python=3.11.10
    conda activate ckpt

    # Install required packages in the ckpt environment
    conda install cmake ninja -y
    conda install conda-forge::mkl-static -y
    conda install -c conda-forge libstdcxx-ng -y
    conda install -c conda-forge cudnn -y
}

# Install deps
install_deps() {
    # For RDMA-core
    # Detect Ubuntu version
    UBUNTU_VERSION=$(lsb_release -sr | tr -d '.')

    # Download the CUDA keyring
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cudnn-cuda-12
    rm cuda-keyring_1.1-1_all.deb

    sudo apt-get install build-essential cmake gcc libudev-dev libnl-3-dev libnl-route-3-dev ninja-build pkg-config valgrind python3-dev cython3 python3-docutils pandoc libpcap-dev libnuma-dev libsystemd-dev clang llvm meson gcc-12 g++-12 curl libcurl4-openssl-dev python3-pyelftools python-is-python3 -y

    pip install maturin patchelf pyyaml
}

# Install Rust
install_rust() {
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
    rustup default nightly
    rustup component add rust-src
    rustup update
}

# Setup hugepages
setup_hugepages() {
    pushd ${SCRIPT_DIR}
    mkdir -p /tmp/mnt/huge
    sudo ./dpdk-hugepages.py --mount --directory /tmp/mnt/huge --user `id -u` --group `id -g` --setup 2G
    popd
}

# Set binary capabilities to run dpdk without sudo
set_bin_caps() {
    py=`which python3`.11
    sudo setcap all+ep $py
}

# Not needed directly, but for a uniform evn.
enable_vfio() {
    sudo modprobe vfio enable_unsafe_noiommu_mode=1
}

setup_env() {
    install_deps
    install_rust
    setup_hugepages
    set_bin_caps
    enable_vfio
}
