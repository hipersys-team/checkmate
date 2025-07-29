# DPDK Plugin

#### Hugepage without root access

```bash
mkdir -p /tmp/mnt/huge
sudo ./dpdk-hugepages.py --mount --directory /tmp/mnt/huge --user `id -u` --group `id -g` --setup 8G
```

> dpdk { socket-mem = 8192; } is required for large number of workers.

> common_mlx5: Can't create a direct mkey - error 121: Update DPDK version.

#### Binary Capabilities

```bash
sudo setcap all=eip ./build/all_reduce_perf
```

However, this will disable `LD_LIBRARY_PATH`, so plugin path will be disabled automatically.

```bash
echo /home/ankitbwj/innet_ckpt/third_party/nccl-plugin/cc | sudo tee /etc/ld.so.conf.d/ncclnet.conf
sudo ldconfig
```
