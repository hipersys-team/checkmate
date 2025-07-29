# Restore Benchmark

This benchmark is used to measure the time taken to restore a model from an in-memory copy of the model.

### Install Restore Package

```bash
maturin develop -r
```

### Run Restore Benchmark

#### On Server

```bash
python python/server.py --model <name>
```

#### On Client

```bash
python python/client.py --model <name>
```

### Results

Results are printed on both server and client side which breakdown the time taken for each step in the process.