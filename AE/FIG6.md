# Figure 6 Replication Guide

This document provides detailed instructions for replicating the experiments shown in Figure 6, which compares different checkpointing approaches across various model types and network configurations.

## Overview

Figure 6 demonstrates the performance comparison between different checkpointing strategies:
- **Baseline (no checkpoint)** - Training without checkpointing
- **Torch async checkpoint** - PyTorch's built-in asynchronous checkpointing
- **CheckFreq** - Frequency-based checkpointing optimization
- **Checkmate** - Our proposed checkpointing system with network-aware optimizations
- **Gemini** - Alternative checkpointing approach

The experiments cover both language models (GPT, LLaMA) and vision models across different network stacks (InfiniBand, DPDK).

## Prerequisites

Before running any experiments, ensure you have:
1. Proper cluster setup with training and storage nodes
2. Network configuration (InfiniBand or DPDK stack)
3. All dependencies installed as per the main README

## Experiment Categories

### 1. Non-Checkmate Baseline Runs

These experiments use standard checkpointing approaches without the Checkmate system.

#### Setup Requirements
First, set up the nullfs sinkhole for storing checkpoints:
```bash
cd scripts/
./setup_sinkhole.sh
```

#### Language Models
Scripts are located under `third_party/torch_titan/scripts/` and follow the naming convention:
```
run_<parallelism>_<type>.sh
```

**Parallelism Types:**
- `dp` - Data Parallel (for GPT models)
- `pipeline` - Pipeline Parallel (for LLaMA models)

**Example Commands:**
```bash
# GPT models with data parallelism
./run_dp_ib.sh          # InfiniBand baseline
./run_dp_dpdk.sh        # DPDK baseline
./run_dp_async.sh       # Torch async checkpoint
./run_dp_checkfreq.sh   # CheckFreq approach

# LLaMA models with pipeline parallelism
./run_pipeline_ib.sh          # InfiniBand baseline
./run_pipeline_dpdk.sh        # DPDK baseline
./run_pipeline_async.sh       # Torch async checkpoint
./run_pipeline_checkfreq.sh   # CheckFreq approach
```

#### Vision Models
Scripts are located under `models/vision/` and follow the naming convention:
```
run_<type>.sh
```

Each script runs all three vision models (the specific models are defined within the scripts).

**Example Commands:**
```bash
cd models/vision/
./run_ib.sh          # InfiniBand baseline
./run_dpdk.sh        # DPDK baseline  
./run_async.sh       # Torch async checkpoint
./run_checkfreq.sh   # CheckFreq approach
```

### 2. Checkmate Runs

Checkmate runs require additional infrastructure setup before execution.

#### Infrastructure Setup

1. **Switch Configuration**
   ```bash
   # Follow the detailed instructions in switch/README.md
   cd switch/
   # Set up Tofino switch as described in README.md
   ```

2. **Storage Cluster Setup**
   ```bash
   # Run storage cluster with parameters from the model scripts
   cd storage/
   python run_<model>.py <parameters>
   ```
   
   The specific parameters for each model are provided within the corresponding `run_<model>.sh` scripts.

#### Running Checkmate Experiments

Once the infrastructure is set up, run the Checkmate experiments:

```bash
# Language models
./run_dp_checkmate.sh          # GPT with Checkmate
./run_pipeline_checkmate.sh    # LLaMA with Checkmate

# Vision models  
./run_checkmate.sh            # Vision models with Checkmate
```

### 3. Gemini Runs

Gemini experiments use a different checkpointing system and require separate setup.

#### Setup and Execution
```bash
# Follow instructions in the Gemini-specific README
cd third_party/gemini/
# See README.md for detailed setup and execution instructions
```

## Network Stack Types

The experiments compare performance across different network configurations:

| Type | Description | Use Case |
|------|-------------|----------|
| `ib` | InfiniBand baseline without checkpointing | High-performance networking baseline |
| `dpdk` | DPDK (libtpa) stack without checkpointing | Kernel-bypass networking baseline |
| `async` | PyTorch asynchronous checkpointing | Standard async checkpointing approach |
| `checkfreq` | CheckFreq checkpointing optimization | Frequency-based checkpoint optimization |
| `checkmate` | Checkmate network-aware checkpointing | Our proposed solution |

## Execution Order

For accurate results, follow this execution order:

1. **Setup Phase**
   ```bash
   # Set up nullfs sinkhole
   cd scripts/ && ./setup_sinkhole.sh
   
   # For Checkmate runs: set up switch and storage
   # (Follow switch/README.md and storage setup instructions)
   ```

2. **Baseline Measurements**
   ```bash
   # Run baseline experiments first (ib, dpdk)
   ./run_*_ib.sh
   ./run_*_dpdk.sh
   ```

3. **Checkpointing Approaches**
   ```bash
   # Run checkpointing experiments
   ./run_*_async.sh
   ./run_*_checkfreq.sh
   ./run_*_checkmate.sh  # Only after infrastructure setup
   ```

4. **Alternative Approaches**
   ```bash
   # Run Gemini experiments
   cd third_party/gemini/ && # follow README.md
   ```

## Data Collection

Each script will output performance metrics that correspond to the data points shown in Figure 6. Ensure you:
- Monitor system resources during execution
- Collect timing information for checkpoint operations
- Record network utilization metrics
- Save training throughput measurements

## Troubleshooting

**Common Issues:**
- Ensure all nodes are properly connected and configured
- Verify network stack setup (InfiniBand/DPDK) before running experiments  
- Check that storage cluster is responding before Checkmate runs
- Monitor disk space and network bandwidth during experiments

**For Checkmate-specific issues:**
- Verify switch configuration using switch/README.md
- Check storage cluster logs for connectivity issues
- Ensure proper port configurations in the switch control plane
