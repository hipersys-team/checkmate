# Install packages
1. Clone the repository

```
git clone git@github.com:hipersys-team/checkmate.git
cd checkmate
git submodule update --init --recursive
```

2. Set up the environemnt: run `build.sh` WITHTOUT conda environment
```
conda deactivate
chmod a+x script/*.sh
script/build.sh
```

# How to run

Add Hugepages for DPDK

```
cd script/
mkdir -p /tmp/mnt/huge
sudo ./dpdk-hugepages.py --mount --directory /tmp/mnt/huge --user `id -u` --group `id -g` --setup 8G
```

### Run Training

Information can be found in [training](models/README.md).

> [!NOTE]
> To specify the number of storage nodes please set environment variable
`NUM_STORAGE` to the desired number.

### Run Storage Server(s)

Information can be found in [storage](storage/README.md).

> [!NOTE]
> To specify the number of training nodes please set environment variable
`NUM_TRAINING` to the desired number.

# Manual Build

Most steps are automated in [build.sh](script/build.sh) script. However, if you want to build
manually, follow the steps below.

## Compile libtpa
```bash
cd third_party/libtpa
make -j
sudo -E make install
```

> [!IMPORTANT]
> For the newer NICs like ConnectX-7 use DPDK version by setting `export DPDK_VERSION=v22.11`


## Compile NCCL
```bash
cd third_party/nccl
sudo make -j src.install
```

## Compile NCCL Plugin
```bash
cd third_party/nccl-plugin/cc
make clean; make -j
```
