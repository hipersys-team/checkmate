/*******************************************************************************
 *  INTEL CONFIDENTIAL
 *
 *  Copyright (c) 2021 Intel Corporation
 *  All Rights Reserved.
 *
 *  This software and the related documents are Intel copyrighted materials,
 *  and your use of them is governed by the express license under which they
 *  were provided to you ("License"). Unless the License provides otherwise,
 *  you may not use, modify, copy, publish, distribute, disclose or transmit
 *  this software or the related documents without Intel's prior written
 *  permission.
 *
 *  This software and the related documents are provided as is, with no express
 *  or implied warranties, other than those that are expressly stated in the
 *  License.
 ******************************************************************************/


#ifndef _HEADERS_
#define _HEADERS_

typedef bit<48> mac_addr_t;
typedef bit<32> ipv4_addr_t;
typedef bit<128> ipv6_addr_t;
typedef bit<12> vlan_id_t;

typedef bit<16> ether_type_t;
const ether_type_t ETHERTYPE_IPV4 = 16w0x0800;
const ether_type_t ETHERTYPE_ARP = 16w0x0806;
const ether_type_t ETHERTYPE_IPV6 = 16w0x86dd;
const ether_type_t ETHERTYPE_VLAN = 16w0x8100;

typedef bit<8> ip_protocol_t;
const ip_protocol_t IP_PROTOCOLS_ICMP = 1;
const ip_protocol_t IP_PROTOCOLS_TCP = 6;
const ip_protocol_t IP_PROTOCOLS_UDP = 17;

header ethernet_h {
    mac_addr_t dst_addr;
    mac_addr_t src_addr;
    bit<16> ether_type;
}

header vlan_tag_h {
    bit<3> pcp;
    bit<1> cfi;
    vlan_id_t vid;
    bit<16> ether_type;
}

header mpls_h {
    bit<20> label;
    bit<3> exp;
    bit<1> bos;
    bit<8> ttl;
}

header ipv4_h {
    bit<4> version;
    bit<4> ihl;
    bit<1> tag;
    // bit<1> toggle;
    // bit<1> reserved;
    bit<5> storage_session_id;
    bit<2> ecn;
    bit<16> total_len;
    bit<16> identification;
    bit<3> flags;
    bit<13> frag_offset;
    bit<8> ttl;
    bit<8> protocol;
    bit<16> hdr_checksum;
    ipv4_addr_t src_addr;
    ipv4_addr_t dst_addr;
}

header ipv6_h {
    bit<4> version;
    bit<8> traffic_class;
    bit<20> flow_label;
    bit<16> payload_len;
    bit<8> next_hdr;
    bit<8> hop_limit;
    ipv6_addr_t src_addr;
    ipv6_addr_t dst_addr;
}

header tcp_h {
    bit<16> src_port;
    bit<16> dst_port;
    bit<32> seq_no;
    bit<32> ack_no;
    bit<4> data_offset;
    bit<4> res;
    bit<1> cwr;
    bit<1> ece;
    bit<1> urg;
    bit<1> ack;
    bit<1> psh;
    bit<1> rst;
    bit<1> syn;
    bit<1> fin;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgent_ptr;
}

typedef bit<8> tcp_option_kind_t;
typedef bit<8> tcp_option_len_t;

const tcp_option_kind_t TCP_OPT_ENDLIST = 8w0;
const tcp_option_kind_t TCP_OPT_NOP = 8w1;
const tcp_option_kind_t TCP_OPT_MSS = 8w2;
const tcp_option_kind_t TCP_OPT_WSCALE = 8w3;
const tcp_option_kind_t TCP_OPT_SACK_OK = 8w4;
const tcp_option_kind_t TCP_OPT_SACK = 8w5;
const tcp_option_kind_t TCP_OPT_TSTAMP = 8w8;
const tcp_option_kind_t TCP_OPT_CKPTSEQ = 8w7;

#define MAX_NOP 20
#define MAX_SACK_BLOCKS 4

header tcp_option_end_h {
  tcp_option_kind_t kind;
}

header tcp_option_nop_h {
  tcp_option_kind_t kind;
}

header tcp_option_mss_h {
  tcp_option_kind_t kind;
  tcp_option_len_t len;
  bit<16> mss;
}

header tcp_option_ckpt_h {
  bit<8> nop1;
  bit<8> nop2;
  tcp_option_kind_t kind;
  tcp_option_len_t len;
  bit<32> cseq;
}

header tcp_option_wscale_h {
  tcp_option_kind_t kind;
  tcp_option_len_t len;
  bit<8> ws;
}

header tcp_option_sack_ok_h {
  tcp_option_kind_t kind;
  tcp_option_len_t len;
}

header tcp_option_sack_data_h {
  bit<32> left_edge;
  bit<32> right_edge;
}

header tcp_option_sack_h {
  tcp_option_kind_t kind;
  tcp_option_len_t len;
}

header tcp_option_timestamp_h {
  tcp_option_kind_t kind; 
  tcp_option_len_t len;
  bit<32> timestamp;
  bit<32> timestamp_echo;
}

header tcp_options_catchall_h {
  varbit<320> options;
}


header udp_h {
    bit<16> src_port;
    bit<16> dst_port;
    bit<16> hdr_length;
    bit<16> checksum;
}

header icmp_h {
    bit<8> type_;
    bit<8> code;
    bit<16> hdr_checksum;
}

// Address Resolution Protocol -- RFC 6747
header arp_h {
    bit<16> hw_type;
    bit<16> proto_type;
    bit<8> hw_addr_len;
    bit<8> proto_addr_len;
    bit<16> opcode;
    mac_addr_t snd_mac;
    ipv4_addr_t snd_spa;
    mac_addr_t tgt_mac;
    ipv4_addr_t tgt_spa;
    // ...
}

// Segment Routing Extension (SRH) -- IETFv7
header ipv6_srh_h {
    bit<8> next_hdr;
    bit<8> hdr_ext_len;
    bit<8> routing_type;
    bit<8> seg_left;
    bit<8> last_entry;
    bit<8> flags;
    bit<16> tag;
}

// VXLAN -- RFC 7348
header vxlan_h {
    bit<8> flags;
    bit<24> reserved;
    bit<24> vni;
    bit<8> reserved2;
}

// Generic Routing Encapsulation (GRE) -- RFC 1701
header gre_h {
    bit<1> C;
    bit<1> R;
    bit<1> K;
    bit<1> S;
    bit<1> s;
    bit<3> recurse;
    bit<5> flags;
    bit<3> version;
    bit<16> proto;
}

struct header_t {
    ethernet_h ethernet;
    vlan_tag_h vlan_tag;
    arp_h arp;
    ipv4_h ipv4;
    ipv6_h ipv6;
    tcp_h tcp;
    udp_h udp;

    tcp_option_ckpt_h tcp_option_ckpt;
    // tcp_options_catchall_h tcp_option_rest;

    // Add more headers here.
    // tcp_option_mss_h tcp_option_mss;
    // tcp_option_wscale_h tcp_option_wscale;
    // tcp_option_sack_ok_h tcp_option_sack_ok;
    // tcp_option_timestamp_h tcp_option_timestamp;
    // tcp_option_sack_h tcp_option_sack;
    // tcp_option_sack_data4_h tcp_option_sack_data4;
    // tcp_option_sack_data3_h tcp_option_sack_data3;
    // tcp_option_sack_data2_h tcp_option_sack_data2;
    // tcp_option_sack_data_h tcp_option_sack_data1;
    // tcp_option_nop_h[MAX_NOP] tcp_option_nop;
    // tcp_option_end_h tcp_option_end;
}

struct empty_header_t {}

struct empty_metadata_t {}

#endif /* _HEADERS_ */
