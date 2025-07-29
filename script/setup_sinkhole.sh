#!/bin/bash

PROJ_ROOT=/data/frankwwy/innet_ckpt

if [ ! -f $PROJ_ROOT/third_party/nullfsvfs/nullfs.ko ]
then
    pushd $PROJ_ROOT/third_party/nullfsvfs/
    make
    popd
fi
sudo insmod $PROJ_ROOT/third_party/nullfsvfs/nullfs.ko
sudo mkdir -p /sinkhole
sudo mount -t nullfs none /sinkhole/
