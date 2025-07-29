#!/bin/bash

SCRIPT_DIR=$(dirname $(readlink -f $0))
LIB_DIR=${SCRIPT_DIR}/../third_party/

NCCL_DIR=${LIB_DIR}/nccl
NCCL_PLUGIN_DIR=${LIB_DIR}/nccl-plugin
TORCH_DIR=${LIB_DIR}/pytorch
NCCL_NET_DIR=${NCCL_PLUGIN_DIR}/cc

export CC=gcc-12
export CXX=g++-12
export CUDACXX=/usr/local/cuda/bin/nvcc
export DPDK_VERSION=v22.11 # Needed for 400G NICs

source ${SCRIPT_DIR}/setup.sh

build_libtpa() {
    pushd ${LIB_DIR}
    cd libtpa
    make -j
    rm /tmp/tpa.mri
    sudo -E make install
    popd
}

build_nccl() {
    pushd ${NCCL_DIR}
    sudo make -j src.install
    popd
}

build_nccl_plugin() {
    pushd ${NCCL_PLUGIN_DIR}
    pushd cc
    make -j
    popd
    popd
}

build_pytorch() {
    pushd ${TORCH_DIR}
    git submodule sync
    git submodule update --init --recursive
    pip install -r requirements.txt
    BUILD_TEST=0 USE_SYSTEM_NCCL=1 python setup.py develop --cmake
    # use `USE_CUDA=0` on storage server; Other flags USE_TBB=1 USE_OPENMP=0
    # USE_CUDA=0 BUILD_TEST=0 python setup.py develop --cmake
    popd
}

build_torchvision() {
    git clone https://github.com/pytorch/vision.git
    pushd vision
    git checkout bf01bab612
    python setup.py install
    popd
    rm -rf vision
}

test_env() {
    pip install omegaconf hydra-core torchsnapshot
    # Test
    python -c "import torch;print(torch.__version__)"
    python -c "import torchvision; print(torchvision.__version__)"
    python -c "import torch;print(torch.version.cuda)"
    python -c "import torch;print(torch.cuda.nccl.version())"
}

if [ -n "$CONDA_PREFIX" ] && [ "$(basename $CONDA_PREFIX)" == "ckpt" ]; then
    echo "Already in the ckpt environment."
    setup_env
else
    echo "Not in ckpt environment. Installing Miniconda and setting up ckpt environment."
    install_miniconda
    setup_env
fi

git submodule update --init --recursive
build_libtpa
build_nccl
build_nccl_plugin
build_pytorch
build_torchvision
test_env
