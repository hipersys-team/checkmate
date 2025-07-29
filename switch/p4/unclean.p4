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

#define NUM_DP_PARTS 1
#define NUM_SRCS 2

//-----------------------------------------------------------------------------
// Metadata Definitions
//-----------------------------------------------------------------------------
struct ckpt_md {
    /* Custom ckpt_md metadata */
    bit<1>   tcp_matched;
    bit<8>   toggle_bit;    // 8 bits b/c tna can't do shit with 1 bit reg
    bit<10>  mirror_session;
    bit<8>   src_id;
}

struct metadata_t {
    ckpt_md ckpt;
    bit<16> checksum;
}

struct egress_metadata_t {
    bit<8> seq_id;
    bit<16> tcp_len;
    bit<16> ihl;
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
    // ParserCounter() parser_counter;

    state start {
        tofino_parser.apply(packet, ig_intr_md);
        ig_md.ckpt.tcp_matched = 0;
        ig_md.ckpt.toggle_bit = 0;
        ig_md.ckpt.mirror_session = 0;
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            0x0800: parse_ipv4;
            default: accept;
        }
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
        // parser_counter.set(hdr.tcp.data_offset, 8w60, 8w6, 3w0, 8w0);
        // parser_counter.decrement(20);
        transition select(hdr.ipv4.diffserv[7:7]) {
            // If there are TCP options, transition to parsing them
            0: accept;
            // Otherwise, accept the packet
            1: parse_cseq;
        }
    }

    state parse_cseq {
        packet.extract(hdr.tcp_option_ckpt);
        transition accept;
    }


    // // State to parse TCP options
    // state parse_tcp_options {
    //     transition select(packet.lookahead<tcp_option_kind_t>()) {
    //         TCP_OPT_ENDLIST: parse_tcp_option_end;
    //         TCP_OPT_NOP: parse_tcp_option_nop;
    //         TCP_OPT_MSS: parse_tcp_option_mss;
    //         TCP_OPT_WSCALE: parse_tcp_option_wscale;
    //         TCP_OPT_SACK_OK: parse_tcp_option_sack_ok;
    //         TCP_OPT_SACK: parse_tcp_option_sack;
    //         // TCP_OPT_TSTAMP: parse_tcp_option_timestamp;
    //         default: accept; // If we encounter an unknown option, stop parsing options
    //     }
    // }
    

    // state parse_tcp_option_end {
    //     packet.extract(hdr.tcp_option_end);
    //     transition accept;
    // }
    

    // state parse_tcp_option_nop {
    //     packet.extract(hdr.tcp_option_nop.next);
    //     parser_counter.decrement(1);
    //     transition select(parser_counter.is_zero()) {
    //         false: parse_tcp_options;
    //         true: accept;
    //     }
    // }

    // state parse_tcp_option_mss {
    //     packet.extract(hdr.tcp_option_mss);
    //     parser_counter.decrement(4);
    //     transition select(parser_counter.is_zero()) {
    //         false: parse_tcp_options;
    //         true: accept;
    //     }
    // }

    // state parse_tcp_option_wscale {
    //     packet.extract(hdr.tcp_option_wscale);
    //     parser_counter.decrement(3);
    //     transition select(parser_counter.is_zero()) {
    //         false: parse_tcp_options;
    //         true: accept;
    //     }
    // }

    // state parse_tcp_option_sack_ok {
    //     packet.extract(hdr.tcp_option_sack_ok);
    //     parser_counter.decrement(2);
    //     transition select(parser_counter.is_zero()) {
    //         false: parse_tcp_options;
    //         true: accept;
    //     }
    // }

    // state parse_tcp_option_sack {
    //     packet.extract(hdr.tcp_option_sack);
    //     parser_counter.decrement(2);
    //     transition select(hdr.tcp_option_sack.len) {
    //         10: parse_tcp_sack_data_1;
    //         18: parse_tcp_sack_data_2;
    //         26: parse_tcp_sack_data_3;
    //         34: parse_tcp_sack_data_4;
    //         default: accept;
    //     }
    // }

    // state parse_tcp_sack_data_4 {
    //     // packet.extract(hdr.tcp_option_sack_data4); 
    //     parser_counter.decrement(32);
    //     transition select(parser_counter.is_zero()) {
    //         false: parse_tcp_options;
    //         true: accept;
    //     }
    // }

    // state parse_tcp_sack_data_3 {
    //     // packet.extract(hdr.tcp_option_sack_data3); 
    //     parser_counter.decrement(24);
    //     transition select(parser_counter.is_zero()) {
    //         false: parse_tcp_options;
    //         true: accept;
    //     }
    // }

    // state parse_tcp_sack_data_2 {
    //     // packet.extract(hdr.tcp_option_sack_data2); 
    //     parser_counter.decrement(16);
    //     transition select(parser_counter.is_zero()) {
    //         false: parse_tcp_options;
    //         true: accept;
    //     }
    // }

    // state parse_tcp_sack_data_1 {
    //     // packet.extract(hdr.tcp_option_sack_data1); // Extract the next SACK block
    //     parser_counter.decrement(8);
    //     transition select(parser_counter.is_zero()) {
    //         false: parse_tcp_options;
    //         true: accept;
    //     }
    // }

    // state parse_tcp_option_timestamp {
    //     packet.extract(hdr.tcp_option_timestamp);
    //     parser_counter.decrement(10);
    //     transition select(parser_counter.is_zero()) {
    //         false: parse_tcp_options;
    //         true: accept;
    //     }
    // }
}

//-----------------------------------------------------------------------------
// Controls
//-----------------------------------------------------------------------------
control SetMirrorPacket (
        inout header_t hdr,
        inout ckpt_md ckpt,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {
    
    const bit<16> LOAD_BAL_VALS = 2;
    
    // Register Definitions
    Register<bit<8>, bit<8>>(255, 0) toggle_bit_reg;
    RegisterAction<bit<8>, _, bit<8>>(toggle_bit_reg) read_and_update_toggle_reg = {
        void apply(inout bit<8> reg_value, out bit<8> curr_value) {
            curr_value = reg_value;
            reg_value = ckpt.toggle_bit;
        }
    };
    // Index of the current egress port in the table
    Register<bit<16>, bit<8>>(255, 0) mirror_session_index_reg;
    RegisterAction<bit<16>, _, bit<16>>(mirror_session_index_reg) 
            read_update_mirror_session_index_reg = {
        void apply(inout bit<16> reg_value, out bit<16> curr_value) {
            if (reg_value == LOAD_BAL_VALS - 1) {
                reg_value = 0;
            }
            else {
                reg_value = reg_value + 1;
            }
            curr_value = reg_value;
        }
    };
    RegisterAction<bit<16>, _, bit<16>>(mirror_session_index_reg) 
            read_mirror_session_index_reg = {
        void apply(inout bit<16> reg_value, out bit<16> curr_value) {
            curr_value = reg_value;
        }
    };

    bit<16> current_session = LOAD_BAL_VALS; // invalid value
    bit<16> egress_mirror_session_id = 0;

    action set_egress_port(bit<16> port) {
        egress_mirror_session_id = port;
    }

    table mirror_port_match {
        key = {
            ckpt.src_id: exact;
            current_session: exact;
        }
        actions = {
            set_egress_port;
            NoAction;
        }
        const default_action = NoAction;
        size = 1024;
    }

    // Logic here:
    // Each source IP address has a toggle bit and a mirror session index.
    // If the toggle bit changes, we need to update the mirror session index.
    // If the toggle bit stays the same, we can use the current mirror session index.
    // We use the mirror_port_match table to match the source IP address and the current session index.
    // The mirror_session specifies the egress port to mirror to.

    apply {
        // Read current toggle bit and egress port from Registers
        bit<8> current_toggle_bit;
        current_toggle_bit = read_and_update_toggle_reg.execute(ckpt.src_id);
        if (ckpt.toggle_bit != current_toggle_bit) {
            // Read current index and compute next index
            current_session = read_update_mirror_session_index_reg.execute(ckpt.src_id);
        } else {
            current_session = read_mirror_session_index_reg.execute(ckpt.src_id);
        }
        // Set mirror port
        mirror_port_match.apply();
        ckpt.mirror_session = (bit<10>)egress_mirror_session_id;
        ig_dprsr_md.mirror_type = 1;
        // borrow this field for some info that the egress can use
        // hdr.ipv4.hdr_checksum[15:8] = current_session;
        // hdr.ipv4.hdr_checksum[7:0] = ckpt.src_id;
        // DEBUG: Put whatever variable you want to show here and TCP dump it
        // hdr.tcp.src_port = (bit<16>)ckpt.src_id;
        // hdr.tcp.dst_port = (bit<16>)current_session; //current_session;
    }
    
}

control Ingress(     
        inout header_t hdr,
        inout metadata_t ig_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    SetMirrorPacket() set_mirror;
    // Action to mark that the packet matches the TCP criteria
    action mark_tcp_matched(bit<8> src_id) {
        ig_md.ckpt.tcp_matched = 1;
        ig_md.ckpt.src_id = src_id;
    }

    // Table to match specific src/dst IP addresses and ports
    table tcp_match {
        key = {
            hdr.ipv4.src_addr: exact;
            hdr.ipv4.dst_addr: exact;
            hdr.tcp.dst_port: exact;
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
        hdr.ipv4.dst_addr = temp;
        hdr.ethernet.dst_addr = hdr.ethernet.src_addr; // bounce back
        hdr.tcp.seq_no = 0;
        hdr.tcp.ack_no = 0;
    }

    action handle_synack() {
        ipv4_addr_t temp = hdr.ipv4.src_addr;
        hdr.ipv4.src_addr = hdr.ipv4.dst_addr;
        hdr.ipv4.dst_addr = temp;
        hdr.ethernet.dst_addr = hdr.ethernet.src_addr; // bounce back
        // SYN-ACK handling 
        hdr.tcp.syn = 0;
        hdr.tcp.ack_no = hdr.tcp.seq_no + 1;
        hdr.tcp.seq_no = 1;
    }

    // Table to match fake tcp connection back to storage
    table tcpsyn_match {
        key = {
            hdr.ipv4.src_addr: exact;
            hdr.ipv4.dst_addr: exact;
            hdr.tcp.syn: exact;
            hdr.tcp.ack: exact;
        }
        actions = {
            handle_syn;
            handle_synack;
            NoAction;
        }
        size = 1024;
    }

    action hit(PortId_t dst_port) {
        ig_tm_md.ucast_egress_port = dst_port;
        ig_dprsr_md.drop_ctl = 0;
    }

    action miss() {
        ig_dprsr_md.drop_ctl = 0x1; // Drop packet.
    }

    table forward {
        key = {
            hdr.ethernet.dst_addr : exact;
        }

        actions = {
            hit;
            miss;
        }

        const default_action = miss;
        size = 1024;
    }

    table drop_ack() {
        key = {
            hdr.ipv4.dst_addr: exact;
        }
        actions = {
            miss;
            NoAction;
        }
        const default_action = NoAction;
        size = 10;
    }

    apply {
        // Apply TCP matching
        if (hdr.tcp.isValid()) {
            tcp_match.apply();
            tcpsyn_match.apply();
        }
        if (ig_md.ckpt.tcp_matched == 1) {
            if (hdr.ipv4.diffserv[7:7] == 1) {
                if (hdr.ipv4.diffserv[6:6] == 0) {
                    ig_md.ckpt.toggle_bit = 0;
                } else {
                    ig_md.ckpt.toggle_bit = 1;
                }
            }
            set_mirror.apply(hdr, ig_md.ckpt, ig_dprsr_md);
        }
        forward.apply();
        drop_ack.apply();
    }
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
        packet.extract(eg_intr_md);
        packet.extract(hdr.ethernet);
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            6: parse_tcp;
            default: accept;
        }
    }
    state parse_tcp {
        packet.extract(hdr.tcp);
        // parser_counter.set(hdr.tcp.data_offset, 8w60, 8w6, 3w0, 8w0);
        // parser_counter.decrement(20);
        transition select(hdr.ipv4.diffserv[7:7]) {
            // If there are TCP options, transition to parsing them
            0: accept;
            // Otherwise, accept the packet
            1: parse_cseq;
        }
    }

    state parse_cseq {
        packet.extract(hdr.tcp_option_ckpt);
        transition accept;
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
        inout egress_intrinsic_metadata_for_deparser_t ig_intr_dprs_md,
        inout egress_intrinsic_metadata_for_output_port_t eg_intr_oport_md) {

    action swap_seq() {
        // eg_md.seq_id = seq_id;
        eg_md.to_process = 1;
    }

    table egress_swapseq_match {
        key = {
            eg_intr_oport_md.egress_port: exact;
            hdr.ipv4.diffserv[7:7]: exact;
        }
        actions = {
            swap_seq;
        }
        const default_action = NoAction;
        size = 1024;
    }

    // bit<32> next_seq;
    // bit<32> seq_diff;
    // bit<32> init_seq;
    // bit<16> tcp_payload_size;

    // Register<bit<32>, bit<8>>(255, 0) seq_num_offset_reg;
    // RegisterAction<bit<32>, _, bit<32>>(seq_num_offset_reg) read_seq_offset = {
    //     void apply(inout bit<32> reg_value, out bit<32> out_value) {
    //         out_value = reg_value;
    //     }
    // };
    // RegisterAction<bit<32>, _, bit<32>>(seq_num_offset_reg) read_set_seq_offset = {
    //     void apply(inout bit<32> reg_value, out bit<32> out_value) {
    //         reg_value = next_seq - init_seq;
    //         out_value = reg_value;
    //     }
    // };

    // Register<bit<32>, bit<8>>(255, 0) next_seq_reg;
    // RegisterAction<bit<32>, _, bit<32>>(next_seq_reg) update_next_seq = {
    //     void apply(inout bit<32> reg_value, out bit<32> out_value) {
    //         out_value = reg_value;
    //         reg_value = next_seq + (bit<32>)tcp_payload_size;
    //     }
    // };

    apply {
        egress_seqid_match.apply();
        if (eg_intr_md.to_process == 1) {
            bit<32> temp = hdr.tcp.seq_no;
            hdr.tcp.seq_no = hdr.tcp_option_ckpt.cseq;
            hdr.tcp_option_ckpt.cseq = hdr.tcp.seq_no;
            hdr.tcp.ack_no = 1;
    //         bit<8> tcp_header_len = (hdr.tcp.data_offset * 4);
    //         bit<8> ip_header_len = (hdr.ipv4.ihl * 4);
    //         bit<16> total_header_size = 14 + ip_header_len + tcp_header_len;
    //         tcp_payload_size = (eg_intr_md.pkt_length - total_header_size);
            
    //         if (hdr.ipv4.diffserv[5:5] == 1) {
    //             // initial packet
    //             init_seq = hdr.tcp.seq_no;
    //             next_seq = update_next_seq.execute(eg_md.seq_id);
    //             seq_diff = read_set_seq_offset(eg_md.seq_id);
    //             hdr.tcp.seq_no = next_seq;
    //         } 
    //         else {
    //             // regular packet to apply mapping
    //             seq_diff = read_seq_offset.execute(eg_md.seq_id);
    //             next_seq = hdr.tcp.seq_no + seq_diff;
    //             hdr.tcp.seq_no = next_seq;
    //             update_next_seq.execute(eg_md.seq_id);
    //         }
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

    apply {
		hdr.ipv4.hdr_checksum = ipv4_checksum.update(
		{hdr.ipv4.version,
		 hdr.ipv4.ihl,
                 hdr.ipv4.diffserv,
                 hdr.ipv4.total_len,
                 hdr.ipv4.identification,
                 hdr.ipv4.flags,
                 hdr.ipv4.frag_offset,
                 hdr.ipv4.ttl,
                 hdr.ipv4.protocol,
                 hdr.ipv4.src_addr,
                 hdr.ipv4.dst_addr});

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
