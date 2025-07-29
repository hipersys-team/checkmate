#!/bin/bash
mkdir -p $SDE/build/p4-build/ckpt
sudo rm -rf $SDE/build/p4-build/ckpt/*
pushd  $SDE/build/p4-build/ckpt
sudo -E cmake $SDE/p4studio -DCMAKE_MODULE_PATH="$SDE/cmake" -DCMAKE_INSTALL_PATH="$SDE_INSTALL$" -DP4_PATH="$HOME/innet_ckpt/switch/p4/ckpt.p4" -DP4_NAME=ckpt -DP4_LANG=p4_16 -DTOFINO=ON -DTOFINO2=OFF
sudo -E make -j
sudo make install
popd
