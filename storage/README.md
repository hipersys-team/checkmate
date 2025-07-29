## Stack to handle gradients at Storage

This is a simple TCP stack to handle gradients at storage. The gradients are
sent by the workers and are stored in the storage. The storage is a simple
in-memory storage.

For now, we handle the gradients from two machines.

## Build

```bash
maturin develop -r
```

#### Tests

To Check network performance, run the following command:

```bash
cargo run -r -- -h
```

#### Connection to Switch

To test connection with the switch, run the following command:

```bash
cargo run -r --bin switch
```

#### Chunk Offset information

Run binary to get the chunk offset information. This binary would print the
chunk offset information for the given number of elements and ranks.

cargo run --bin offset nelem nranks

Example:

```bash
cargo run --bin offset 1024 4
```

## Connect to the switch

```bash
cargo run --bin switch
```

## Optimizer Bench

```bash
run_opt_bench.py [-h] [--model {resnet50,resnet152,vgg11,vit_h_14,gpt3_medium,gpt3_large,gpt3_xl,gpt3_6_7B}]
```

**Run**
```bash
cd models
torchrun --nnodes=1 --nproc-per-node=1 --node-rank=0 --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=localhost run_opt_bench.py --model resnet50
```

Use [this script](../script/run_opt_multinode.py) to run `run_opt_bench.py` on multiple nodes.

## Shadow Model 

> [!IMPORTANT]
> torchrun does not respect affinity-related flags; if you need strong core isolation, please specify torchrun-related env variable explicitly.
> ```bash
> NUM_TRAINING=4 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=1234 MKL_NUM_THREADS=16 KMP_AFFINITY=verbose,explicit,proclist=[12-27],granularity=fine python3 run_vision.py --model resnet50
> ```

Storage code doesn't generate tpa.conf file. So, you need to generate it
manually. Find hardware information using the following command:

```bash
sudo lshw -c network -businfo -quiet
```

Also verify the IP address of each of the storage node. While each storage node
is using two NICs, only add IP address of first interface.

## Network Config

### Libtpa

Use libtpa [branch](https://github.com/ankitbhrdwj/libtpa/tree/frank):
```bash
git clone git@github.com:ankitbhrdwj/libtpa.git
cd libtpa
git checkout -b frank
sudo -E make install -j
```

### NIC

Possible NIC features to use on storage
```bash
sudo ethtool --offload 100gp1 lro on
sudo ethtool --offload 100gp2 lro on
sudo ethtool -A 100gp1 rx off tx off
sudo ethtool -A 100gp2 rx off tx off
sudo ethtool --set-priv-flags 100gp1 dropless_rq on
sudo ethtool --set-priv-flags 100gp2 dropless_rq on
```

Also, play with pfc-related changes given in [setup_nic.sh](https://github.com/Flasew/innet_ckpt/blob/main/script/setup_nic.sh)
