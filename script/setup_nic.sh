#!/bin/bash

set -e

USAGE="./scripts/setup.sh <net_interface_name>"

if [ "$#" -ne 1 ]
then
  echo "usage: ${USAGE}"
  exit 1
fi

NET_IFACE_NAME=$1

function setup_net_interface {
  sudo ifconfig $NET_IFACE_NAME mtu 9000 up
  ifconfig $NET_IFACE_NAME | grep -q "mtu 9000"
}

function setup_trust_dscp {
    sudo mlnx_qos -i $NET_IFACE_NAME --trust dscp
}

function setup_pfc {
    # Disable RX/TX flow control
    # sudo ethtool -A $NET_IFACE_NAME rx off tx off

    # Disable checksum
    if [ "$(hostname)" = "sr04" ] || [ "$(hostname)" = "sr05" ]; then
        #sudo ethtool -K $NET_IFACE_NAME rx off tx off
        echo "Disable RX/TX checksum offloading"
        exit 0
    fi

    sudo mlnx_qos -i $NET_IFACE_NAME -f 1,1,1,1,1,1,1,1 \
                  -p 0,1,2,3,4,5,6,7 \
                  --prio2buffer 0,0,0,0,0,0,0,0 \
                  -s strict,strict,strict,strict,strict,strict,strict,strict
    sudo mlnx_qos -i $NET_IFACE_NAME --buffer_size=524160,0,0,0,0,0,0,0
}

setup_net_interface || true
setup_trust_dscp || true
setup_pfc || true
