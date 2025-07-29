## Interface Name Update

To update and persist the interface information across reboots, please update `/etc/netplan/50-cloud-init.yaml` or some other file that has already been created for this purpose.

```
    100gp1:
        dhcp4: false
        optional: true
        mtu: 9000
        addresses:
          - 192.168.10.24/24
        match:
          macaddress: 98:03:9b:14:0c:70
        set-name: 100gp1
```

Try and apply changes using:

```bash
sudo netplan generate
sudo netplan apply
```

## Setup NIC Bonding

Update netplan in `/etc/netplan/50-cloud-init.yaml`.

```
  bonds:
    bond0:
      dhcp4: false
      optional: true
      mtu: 9000
      interfaces:
        - enp67s0np0
        - enp199s0np0
      addresses:
        - 192.168.10.24/24  # Primary IP address
        # - 192.168.10.34/24  # Uncomment if you need both IPs
      parameters:
        mode: balance-alb
        lacp-rate: fast
        transmit-hash-policy: layer2+3
        mii-monitor-interval: 100
```

## Setup tpa config

```
net { name = bond0; bonding = slave/0000:43:00.0 slave/0000:c7:00.0; ip = 192.168.10.24; gw = 192.168.10.0; make = 255.255.255.0; }
dpdk { pci = 0000:43:00.0 0000:c7:00.0; socket-mem = 8192; }
tcp { snd_queue_size = 2048; }
trace { enable = 0; }
```
