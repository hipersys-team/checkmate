### With Explicit TCP Connection

Currently, this version only works with one storage server, as the number of TCP connections and threads needed to connect with multiple storage nodes are detrimental to the performance.

#### Training Node

```bash
cd ~/innet_ckpt/third_party/nccl-plugin/cc
rm libbagua_net.a
make storage
```

While running the training, prepand `STORAGE_SERVER_IP` to the `torchrun` command.

#### Storage Node
Run the storage node without the `switch` feature flag.

```bash
cargo run -r --no-default-features -- -b resnet50
```

>[!TIP]
>Make sure that the server is listening on 5678... ports. The default NCCL tries to connect to those ports for explicit connection. 

### With Switch Connection

Compile NCCL-plugin in normal mode.

```bash
cd ~/innet_ckpt/third_party/nccl-plugin/cc
rm libbagua_net.a
make
```

#### Storage

Set ARP entry for the switch

```bash
sudo arp -s 192.168.10.245 aa:aa:aa:aa:aa:aa -i 100gp1
sudo arp -s 192.168.10.245 aa:aa:aa:aa:aa:aa -i 100gp2
```

```bash
cargo run -r -- -b resnet50
```
