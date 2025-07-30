# Switch Control for Checkmate

This directory contains the switch control software for the Checkmate system, which consists of two main components:

- `p4/` - Data plane code written in P4
- `control/` - Control plane code written in Python

## Prerequisites

- Intel Tofino switch properly set up according to Intel's original guidelines
- Intel SDE (Software Development Environment) version 9.12.0
- Environment variables `SDE` and `SDE_INSTALL` configured correctly

## Data Plane Setup (P4)

The data plane code is located in the `p4/` directory and implements the switching logic for checkpoint traffic management.

### Building and Loading

1. **Build the P4 code:**
   ```bash
   cd p4/
   ./build.sh
   ```

2. **Load the switch program:**
   ```bash
   sudo -E ${SDE}/run_switchd.sh -p ckpt
   ```

## Control Plane Setup (Python)

The control plane code is located in the `control/` directory and manages the switch configuration through various Python modules.

### Main Entry Point

The main entry point is `ckpt.py`, which is the only file that needs to be executed directly.

### Cluster Configuration

The system uses `InnetCkptConfig` dataclass to define the cluster configuration for checkpoint operations. Each configuration specifies how checkpoint traffic should be routed between training and storage nodes.

#### InnetCkptConfig Parameters

```python
@dataclass
class InnetCkptConfig:
    rank0_src: str           # Source machine for rank 0 (first training node)
    rank0_dst: str           # Destination machine for rank 0 traffic
    rank0_ports: list[int]   # List of ports used by rank 0 for checkpointing
    rank0_storage: list[str] # List of storage machines for rank 0 checkpoints
    lastrank_src: str        # Source machine for last rank (final training node)
    lastrank_dst: str        # Destination machine for last rank traffic
    lastrank_ports: list[int]# List of ports used by last rank for checkpointing
    lastrank_storage: list[str] # List of storage machines for last rank checkpoints
```

#### Configuration Parameters Explained

- **rank0_src/lastrank_src**: The hostname of the training machine that initiates checkpoint traffic
- **rank0_dst/lastrank_dst**: The hostname of the intermediate node that receives and forwards checkpoint data
- **rank0_ports/lastrank_ports**: TCP port ranges used for checkpoint communication (e.g., `range(41000, 41004)`)
- **rank0_storage/lastrank_storage**: List of storage machine hostnames where checkpoint data will be stored

#### Example Configuration

```python
INNET_CKPT_CFG = [
    InnetCkptConfig(
        rank0_src="uther",           # Training node rank 0
        rank0_dst="venus",           # Intermediate node
        rank0_ports=list(range(41004, 41008)),  # Ports 41004-41007
        rank0_storage=["sr04p1", "sr05p1", "sr01p1", "sr02p1"],  # Storage nodes
        lastrank_src="xana",         # Training node last rank
        lastrank_dst="uther",        # Intermediate node
        lastrank_ports=list(range(41000, 41004)),  # Ports 41000-41003
        lastrank_storage=["sr04p2", "sr05p2", "sr01p2", "sr02p2"],  # Storage nodes
    )
]
```

### Running the Control Plane

```python
# Basic usage
${SDE}/install/bin/python3.10 ckpt.py
```

The main function in `ckpt.py` creates an `InnetCkpt` instance with the following parameters:

```python
ckpt = InnetCkpt(
    PORTS_INFO,                    # Port configuration mapping
    INNET_CKPT_CFG,               # Checkpoint configuration list
    ["sr01p1", "sr02p1"],         # Training machines list
    ["sr04p1", "sr04p2", "sr05p1", "sr05p2"],  # Storage machines list
    "192.168.10.245",             # Switch fake IP address
)
```

### Control Plane Components

The control plane consists of several modules:

- **`l2switch.py`** - Basic L2 switching functionality
- **`ingress.py`** - Ingress flow matching and processing
- **`egress.py`** - Egress port processing and sequence swapping
- **`mcast.py`** - Multicast group management
- **`mirror_cfg.py`** - Mirror session configuration
- **`portmap.py`** - Port mapping utilities

### Network Configuration

The system maintains a `PORTS_INFO` dictionary that maps machine hostnames to their network configuration:

```python
PORTS_INFO = {
    "machine_name": MachinePortInfo(
        hostname,       # Machine hostname
        port_number,    # Switch port number
        lane,          # Port lane (usually 0)
        speed,         # Port speed (BF_SPEED_100G)
        mac_address,   # MAC address
        ip_address     # IPv4 address
    ),
    # ... more machines
}
```

## Usage Notes

1. Ensure all training and storage machines are properly connected to the switch
2. Verify that the port configurations in `PORTS_INFO` match your physical setup
3. Adjust the `InnetCkptConfig` parameters according to your specific cluster topology
4. The switch fake IP (`192.168.10.245` in the example) should be unique and not conflict with any real machine IPs

## Architecture Overview

The system implements a checkpoint traffic management solution where:

1. Training nodes generate checkpoint data
2. The switch intercepts and routes this traffic using multicast
3. Storage nodes receive and persist the checkpoint data
4. The P4 data plane handles high-speed packet processing
5. The Python control plane manages flow rules and multicast groups
