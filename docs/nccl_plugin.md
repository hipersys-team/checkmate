## NCCL Plugin

NCCL allows the replacement of the transport protocol for collective communication using transport plugins.

There are mainly two ways to use a custom plugin:
- Replace the `NCCL_NET_PLUGIN` name. For example, `NCCL_NET_PLUGIN=custom` would look for a `libnccl-net-custom.so` file in the system path.
- Add plugin directory to the path where the name of the file is `libnccl-net.so`.

## Run NCCL tests with a custom plugin

```bash
mpirun --allow-run-as-root --bind-to none -H localhost,sr02 --np 2 --mca btl_tcp_if_include 10.2.1.0/24 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BAGUA_NET_LIBRARY_PATH -x NCCL_IB_DISABLE=1 -x NCCL_SOCKET_NTHREADS=1 -x NCCL_BUFFSIZE=16777216 -x NCCL_MIN_NRINGS=8 build/all_reduce_perf -b 1G -e 1G -f 2 -g 1
```

> --bind-to is important as the default bind-to is changed based on a number of processes.

- Make sure that the NCCL versions on both machines are compatible.
- `all_reduce_perf` is running with `mpirun -H localhost`
- `nccl_tests` are compiled with MPI=1
```bash
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda-12.4 NCCL_HOME=/usr/include -j 32
```
- Lastly, localhost is able to ssh without a password.

### NCCL Config

Some of the NCCL configurations can be added to ~/.nccl.conf.

For example, `NCCL_DEBUG=INFO` and `NCCL_SOCKET_IFNAME=100gp1` override the debug message and default interface name.

`NCCL_IB_DISABLE=1` would disable IB and would fall back on socket-based communication across machines.

### Cuda toolkit

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-5
```

### Debugging

CPU and GPU debugging can be done using `gdb` and `cuda-gdb`.
Run `cuda-gdb` and attach the processes using `attach pid`. You might need to allow ptrace from the kernel.

```bash
set cuda break_on_launch application
cuda kernel block thread
b ncclAllReduce
```
