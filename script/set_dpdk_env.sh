#!/bin/bash

export NCCL_DEBUG_SUBSYS=INIT,GRAPH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_SOCKET_IFNAME=100gp1
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_MAX_NCHANNELS=4
export NCCL_MIN_NCHANNELS=4
export NCCL_DEBUG=INFO
export NCCL_NET_PLUGIN=/data/frankwwy/innet_ckpt/third_party/nccl-plugin/cc/libnccl-net.so
export BAGUA_NET_IMPLEMENT=DPDK

