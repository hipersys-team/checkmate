#!/bin/bash

# List of hostnames, IPs, MAC addresses, and interfaces
HOSTS=(
  "sr01 192.168.10.21/24 98:03:9b:15:b5:f6 100gp1"
  "sr02 192.168.10.22/24 98:03:9b:15:b8:4a 100gp1"
  "sr03 192.168.10.23/24 50:6b:4b:dd:f0:de 100gp1"
  "sr04 192.168.10.24/24 98:03:9b:14:0c:70 100gp1"
  "sr04 192.168.10.34/24 98:03:9b:3f:8f:46 100gp2"
  "sr05 192.168.10.25/24 50:6b:4b:d7:52:98 100gp1"
  "sr05 192.168.10.35/24 50:6b:4b:f5:cc:66 100gp2"
  "uther 192.168.10.9/24 a0:88:c2:96:06:f8 100gp1"
  "venus 192.168.10.10/24 a0:88:c2:96:05:30 100gp1"
  "windows 192.168.10.11/24 a0:88:c2:96:07:5c 100gp1"
  "xana 192.168.10.12/24 a0:88:c2:96:06:0c 100gp1"
  "toffee2 192.168.10.245/24 aa:aa:aa:aa:aa:aa lol"
)

# Get the current hostname of the machine
CURRENT_HOSTNAME=$(hostname)

# Set IP address of the current host (find matching IP by hostname)
for entry in "${HOSTS[@]}"; do
  HOSTNAME=$(echo $entry | awk '{print $1}')
  IP=$(echo $entry | awk '{print $2}')
  INTERFACE=$(echo $entry | awk '{print $4}')
  
  if [[ "$HOSTNAME" == "$CURRENT_HOSTNAME" ]]; then
    echo "Setting IP address $IP on interface $INTERFACE for host $CURRENT_HOSTNAME"
    ip addr add $IP dev $INTERFACE
    ip link set $INTERFACE up
  fi
done

# Add static ARP entries for all hosts except for the current one
for entry in "${HOSTS[@]}"; do
  HOSTNAME=$(echo $entry | awk '{print $1}')
  IP=$(echo $entry | awk '{print $2}' | cut -d'/' -f1) # Extract IP without CIDR
  INTERFACE=$(echo $entry | awk '{print $4}')
  MAC=$(echo $entry | awk '{print $3}')
  
  # Skip the current host
  if [[ "$HOSTNAME" != "$CURRENT_HOSTNAME" ]]; then
    # Loop through all entries for the current host to get the interfaces for it
    for current_entry in "${HOSTS[@]}"; do
      CURRENT_INTERFACE_HOST=$(echo $current_entry | awk '{print $1}')
      CURRENT_INTERFACE=$(echo $current_entry | awk '{print $4}')
      
      # Only add ARP entry for interfaces on the current host
      if [[ "$CURRENT_INTERFACE_HOST" == "$CURRENT_HOSTNAME" ]]; then
        echo "Adding static ARP entry for $HOSTNAME with IP $IP and MAC $MAC on interface $CURRENT_INTERFACE"
        arp -s $IP $MAC -i $CURRENT_INTERFACE
      fi
    done
  fi
done

echo "Static ARP setup completed."

