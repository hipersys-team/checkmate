#include <core.p4>
#if __TARGET_TOFINO__ == 3
#include <t3na.p4>
#elif __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif

#include "common/headers.p4"
#include "common/util.p4"

//-----------------------------------------------------------------------------
// Metadata Definitions
//-----------------------------------------------------------------------------
struct ckpt_md {
    /* Custom ckpt_md metadata */
    bit<1>   tcp_matched;
    bit<10>  mirror_session;
    bit<8>   src_id;
}

struct metadata_t {
    ckpt_md ckpt;
    bit<16> checksum;
}

struct egress_metadata_t {
    bit<10>  mirror_session;
}


//-----------------------------------------------------------------------------
// IngressParser
//-----------------------------------------------------------------------------
parser IngressParser(
        packet_in packet,
        out header_t hdr,
        out metadata_t ig_md,
        out ingress_intrinsic_metadata_t ig_intr_md) {

    TofinoIngressParser() tofino_parser;

    state start {
        ig_md.ckpt.tcp_matched = 0;
        ig_md.ckpt.mirror_session = 0;
        ig_md.ckpt.src_id = 0;
        ig_md.checksum = 0;
        tofino_parser.apply(packet, ig_intr_md);
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            0x0800: parse_ipv4;
            0x0806: parse_arp;
            default: accept;
        }
    }

    state parse_arp {
        packet.extract(hdr.arp);
        transition accept;
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            6: parse_tcp; // TCP protocol
            default: accept;
        }
    }

    state parse_tcp {
        packet.extract(hdr.tcp);
        transition accept;
    }

}

control Ingress(     
        inout header_t hdr,
        inout metadata_t ig_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    // Action to mark that the packet matches the TCP criteria
    action mark_tcp_matched(MulticastGroupId_t src_id) {
        ig_md.ckpt.tcp_matched = 1;
        invalidate(ig_tm_md.ucast_egress_port);
        // ig_md.ckpt.src_id = src_id;
        ig_dprsr_md.drop_ctl = 0;
        ig_tm_md.mcast_grp_a = src_id;
    }

    // Table to match specific src/dst IP addresses and ports
    table tcp_match {
        key = {
            hdr.ipv4.src_addr: exact;
            hdr.ipv4.dst_addr: exact;
            hdr.ipv4.tag: exact;
            hdr.tcp.dst_port: exact;
            hdr.ipv4.storage_session_id: exact;
        }
        actions = {
            mark_tcp_matched;
            NoAction;
        }
        size = 1024;
    }

    action handle_syn() {
        // shoot the SYN/ACK packet back with swapped IP src and dst
        ipv4_addr_t temp = hdr.ipv4.src_addr;
        hdr.ipv4.src_addr = hdr.ipv4.dst_addr;
        hdr.tcp.src_port = hdr.tcp.dst_port + 100;
        ig_tm_md.ucast_egress_port = ig_intr_md.ingress_port;
        ig_dprsr_md.drop_ctl = 0x0;
        hdr.ipv4.dst_addr = temp;
        hdr.ethernet.dst_addr = hdr.ethernet.src_addr; // bounce back
        hdr.tcp.seq_no = 0;
        hdr.tcp.ack_no = 0;
        // DEBUG
        // ig_dprsr_md.mirror_type = 1;
        // ig_md.ckpt.mirror_session = 1;
    }

    action handle_synack() {
        ipv4_addr_t temp = hdr.ipv4.src_addr;
        hdr.ipv4.src_addr = hdr.ipv4.dst_addr;
        hdr.ipv4.dst_addr = temp;
        hdr.ethernet.dst_addr = hdr.ethernet.src_addr; // bounce back
        bit<16> temp_port = hdr.tcp.src_port;
        hdr.tcp.src_port = hdr.tcp.dst_port;
        hdr.tcp.dst_port = temp_port;
        ig_tm_md.ucast_egress_port = ig_intr_md.ingress_port;
        ig_dprsr_md.drop_ctl = 0x0;
        // SYN-ACK handling 
        hdr.tcp.syn = 0;
        hdr.tcp.ack_no = hdr.tcp.seq_no + 1;
        hdr.tcp.seq_no = 1;
        // DEBUG
        // ig_dprsr_md.mirror_type = 1;
        // ig_md.ckpt.mirror_session = 1;
    }

    action miss() {
        ig_dprsr_md.drop_ctl = 0x1; // Drop packet.
    }

    // Table to match fake tcp connection back to storage
    table tcpsyn_match {
        key = {
            // hdr.ipv4.src_addr: exact;
            hdr.ipv4.dst_addr: exact;
            hdr.tcp.syn: exact;
            hdr.tcp.ack: exact;
        }
        actions = {
            handle_syn;
            handle_synack;
            miss;
            NoAction;
        }
        const default_action = NoAction;
        size = 1024;
    }

    action hit(PortId_t dst_port) {
        ig_tm_md.ucast_egress_port = dst_port;
        ig_dprsr_md.drop_ctl = 0;
        // DEBUG
        // ig_dprsr_md.mirror_type = 1;
        // ig_md.ckpt.mirror_session = 1;
    }

    action broadcast(MulticastGroupId_t mgid) {
        ig_tm_md.mcast_grp_a = mgid;
        ig_dprsr_md.drop_ctl = 0;
    }

    table forward {
        key = {
            hdr.ethernet.dst_addr : exact;
        }

        actions = {
            hit;
            broadcast;
            miss;
        }

        const default_action = miss;
        size = 512;
    }

    action fill_arp_resp(mac_addr_t resp_mac) {
        // ig_tm_md.ucast_egress_port = ig_intr_md.ingress_port;
        hdr.arp.tgt_mac = hdr.arp.snd_mac;
        hdr.arp.snd_mac = resp_mac;
        hdr.arp.opcode = 2;
        ipv4_addr_t tmp = hdr.arp.snd_spa;
        hdr.arp.snd_spa = hdr.arp.tgt_spa;
        hdr.arp.tgt_spa = tmp;
        hdr.ethernet.dst_addr = hdr.ethernet.src_addr;
        hdr.ethernet.src_addr = resp_mac;
        // ig_dprsr_md.drop_ctl = 0;
    }

    table arp {
        key = {
            hdr.arp.opcode: exact;
            hdr.arp.tgt_spa: exact;
        }

        actions = {
            fill_arp_resp;
            NoAction;
        }

        const default_action = NoAction;
        size = 512;
    }

    apply {
        // DEBUG
        // ig_dprsr_md.mirror_type = 1;
        // ig_md.ckpt.mirror_session = 1;

        ig_tm_md.qid = 3;
        ig_tm_md.ingress_cos = 3;
        
        if (hdr.arp.isValid()) {
            arp.apply();
        }

        // else {
            // Apply TCP matching
        forward.apply();
        if (hdr.tcp.isValid()) {
            tcp_match.apply();
            tcpsyn_match.apply();
        }
        // if (ig_md.ckpt.tcp_matched == 1) {
        //     // if () {
        //     ig_md.ckpt.toggle_bit = (bit<8>)hdr.ipv4.toggle;
        //     set_mcast.apply(hdr, ig_md.ckpt, ig_tm_md);
        //     // }
        // }
    }
    // }
}

// ---------------------------------------------------------------------------
// Ingress Deparser
// ---------------------------------------------------------------------------
control IngressDeparser(
        packet_out packet,
        inout header_t hdr,
        in metadata_t ig_md,
        in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {
    Mirror() mirror;
    apply {
        // If mirroring is needed, clone the packet
        if (ig_dprsr_md.mirror_type == 1) {
            mirror.emit(ig_md.ckpt.mirror_session);
        }
        packet.emit(hdr);

    }
}

//-----------------------------------------------------------------------------
// Egress Parser
//-----------------------------------------------------------------------------
parser EgressParser(
        packet_in packet,
        out header_t hdr,
        out egress_metadata_t eg_md,
        out egress_intrinsic_metadata_t eg_intr_md) {

    state start {
        eg_md.mirror_session = 0;
        packet.extract(eg_intr_md);
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            0x0800: parse_ipv4;
            0x0806: parse_arp;
            default: accept;
        }
    }

    state parse_arp {
        packet.extract(hdr.arp);
        transition accept;
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            6: parse_tcp; // TCP protocol
            default: accept;
        }
    }

    state parse_tcp {
        packet.extract(hdr.tcp);
        transition select(hdr.ipv4.tag) {
            0: accept;
            1: parse_cseq;
        }
    }

    state parse_cseq {
        packet.extract(hdr.tcp_option_ckpt);
        transition accept; 
        // select(hdr.tcp.data_offset) {
        //     0..7: accept;
        //     8..15: parse_rest;
        // }
    }



}

//-----------------------------------------------------------------------------
// Egress Control
//-----------------------------------------------------------------------------
control Egress(	
        inout header_t hdr,
        inout egress_metadata_t eg_md,
        in egress_intrinsic_metadata_t eg_intr_md,
        in egress_intrinsic_metadata_from_parser_t eg_intr_md_from_prsr,
        inout egress_intrinsic_metadata_for_deparser_t eg_intr_dprs_md,
        inout egress_intrinsic_metadata_for_output_port_t eg_intr_oport_md) {

    action swap_seq(mac_addr_t dst_mac, ipv4_addr_t src, ipv4_addr_t dst) {
        bit<32> temp = hdr.tcp.seq_no;
        hdr.tcp.seq_no = hdr.tcp_option_ckpt.cseq;
        hdr.tcp.ack_no = 1;
        hdr.tcp.src_port = hdr.tcp.dst_port + 100;
        hdr.ethernet.dst_addr = dst_mac;
        hdr.ipv4.src_addr = src;
        hdr.ipv4.dst_addr = dst;
    }

    table egress_swapseq_match {
        key = {
            eg_intr_md.egress_port: exact;
            hdr.ipv4.tag: exact;
        }
        actions = {
            swap_seq;
            NoAction;
        }
        const default_action = NoAction;
        size = 1024;
    }

    apply {
        // eg_intr_dprs_md.mirror_type = 2;
        // eg_md.mirror_session = 2;
        if (hdr.tcp.isValid()) {
            if (hdr.tcp_option_ckpt.isValid()) {
                // hdr.tcp.data_offset = hdr.tcp.data_offset - 2;
                // hdr.tcp_option_ckpt.setInvalid();
                egress_swapseq_match.apply();
                hdr.tcp_option_ckpt.kind = 0x1;
                hdr.tcp_option_ckpt.len = 0x1;
                hdr.tcp_option_ckpt.cseq = 32w0x01010101;
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Deparser
//-----------------------------------------------------------------------------
control EgressDeparser(	
        packet_out packet,
        inout header_t hdr,
        in egress_metadata_t eg_md,
        in egress_intrinsic_metadata_for_deparser_t eg_intr_dprs_md) {

    Checksum() ipv4_checksum;
    // Mirror() mirror;

    apply {
        if (hdr.ipv4.isValid()) {
            hdr.ipv4.hdr_checksum = ipv4_checksum.update(
            {hdr.ipv4.version,
            hdr.ipv4.ihl,
                    hdr.ipv4.tag,
                    hdr.ipv4.storage_session_id,
                    // hdr.ipv4.toggle,
                    // hdr.ipv4.reserved,
                    // hdr.ipv4.diffserv,
                    hdr.ipv4.ecn,
                    hdr.ipv4.total_len,
                    hdr.ipv4.identification,
                    hdr.ipv4.flags,
                    hdr.ipv4.frag_offset,
                    hdr.ipv4.ttl,
                    hdr.ipv4.protocol,
                    hdr.ipv4.src_addr,
                    hdr.ipv4.dst_addr});
        } 
        // if (eg_intr_dprs_md.mirror_type == 2) {
        //     mirror.emit(eg_md.mirror_session);
        // }
        packet.emit(hdr);
    }
}

//-----------------------------------------------------------------------------
// Pipeline
//-----------------------------------------------------------------------------
Pipeline(IngressParser(),
         Ingress(),
         IngressDeparser(),
         EgressParser(),
         Egress(),
         EgressDeparser()) pipe; 

Switch(pipe) main;
