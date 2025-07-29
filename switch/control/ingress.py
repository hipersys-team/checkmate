from dataclasses import dataclass
import bfrt_grpc.client as gc

from l2switch import MachinePortInfo, SimpleL2Switch
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(encoding="utf-8", level=logging.DEBUG)


@dataclass
class FlowMatchInfo:
    src_ip: str
    dst_ip: str
    dst_port: int
    src_id: int
    storage_session_id: int 


class FlowMatch:

    def __init__(self, switch: SimpleL2Switch, flow_matches_info: list[FlowMatchInfo]):
        self.switch = switch
        self.bfrt_info = switch.bfrt_info
        self.target = switch.target
        self.flow_matches_info: list[FlowMatchInfo] = flow_matches_info
        self.tcp_match_table = self.bfrt_info.table_get("Ingress.tcp_match")
        self.tcp_match_table.info.key_field_annotation_add("hdr.ipv4.src_addr", "ipv4")
        self.tcp_match_table.info.key_field_annotation_add("hdr.ipv4.dst_addr", "ipv4")

    def update_table(self):
        for flow_match in self.flow_matches_info:
            logger.info(f"updating {flow_match}")
            self.tcp_match_table.entry_add(
                self.target,
                [
                    self.tcp_match_table.make_key(
                        [
                            gc.KeyTuple("hdr.ipv4.src_addr", flow_match.src_ip),
                            gc.KeyTuple("hdr.ipv4.dst_addr", flow_match.dst_ip),
                            gc.KeyTuple("hdr.ipv4.tag", 1),
                            gc.KeyTuple("hdr.tcp.dst_port", flow_match.dst_port),
                            gc.KeyTuple("hdr.ipv4.storage_session_id", flow_match.storage_session_id),
                        ]
                    )
                ],
                [
                    self.tcp_match_table.make_data(
                        [gc.DataTuple("src_id", flow_match.src_id)],
                        "Ingress.mark_tcp_matched",
                    )
                ],
            )

    def reset(self):
        logger.info(f"reset tcp_match_table")
        self.tcp_match_table.entry_del(
            self.target,
            [entry_key for _, entry_key in self.tcp_match_table.entry_get(self.target)],
        )


# @dataclass
# class McastPortMatchInfo:
#     src_id: int
#     current_idx: int
#     egress_session: int


# class McastMatch:

#     def __init__(
#         self, switch: SimpleL2Switch, mcast_matches_info: list[McastPortMatchInfo]
#     ):
#         self.switch = switch
#         self.bfrt_info = switch.bfrt_info
#         self.target = switch.target
#         self.mcast_matches_info: list[McastPortMatchInfo] = mcast_matches_info
#         self.mcast_port_match_table = self.bfrt_info.table_get(
#             "Ingress.set_mcast.mcast_port_match"
#         )

#     def update_table(self):
#         for mcast_match in self.mcast_matches_info:
#             logger.info(f"Updating {mcast_match}")
#             self.mcast_port_match_table.entry_add(
#                 self.target,
#                 [
#                     self.mcast_port_match_table.make_key(
#                         [
#                             gc.KeyTuple("ckpt.src_id", mcast_match.src_id),
#                             gc.KeyTuple("current_session", mcast_match.current_idx),
#                         ]
#                     )
#                 ],
#                 [
#                     self.mcast_port_match_table.make_data(
#                         [gc.DataTuple("mcast_group", mcast_match.egress_session)],
#                         "Ingress.set_mcast.set_egress_port",
#                     )
#                 ],
#             )

#     def reset(self):
#         logger.info(f"reset mcast_match")
#         self.mcast_port_match_table.entry_del(
#             self.target,
#             [
#                 entry_key
#                 for _, entry_key in self.mcast_port_match_table.entry_get(self.target)
#             ],
#         )


@dataclass
class SynMatchInfo:
    # src_ip: str
    dst_ip: str


class SynMatch:

    def __init__(self, switch: SimpleL2Switch, syn_match_info: SynMatchInfo):
        self.switch = switch
        self.bfrt_info = switch.bfrt_info
        self.target = switch.target
        self.syn_match_info: SynMatchInfo = syn_match_info
        self.syn_match_table = self.bfrt_info.table_get("Ingress.tcpsyn_match")
        self.syn_match_table.info.key_field_annotation_add("hdr.ipv4.dst_addr", "ipv4")

    def update_table(self):
        # for self.syn_match_info in self.syn_matches_info:
        logger.info(f"updating {self.syn_match_info}")
        self.syn_match_table.entry_add(
            self.target,
            [
                self.syn_match_table.make_key(
                    [
                        # gc.KeyTuple("hdr.ipv4.src_addr", self.syn_match_info.src_ip),
                        gc.KeyTuple("hdr.ipv4.dst_addr", self.syn_match_info.dst_ip),
                        gc.KeyTuple("hdr.tcp.syn", 1),
                        gc.KeyTuple("hdr.tcp.ack", 0),
                    ]
                )
            ],
            [
                self.syn_match_table.make_data(
                    [],
                    "Ingress.handle_syn",
                )
            ],
        )
        self.syn_match_table.entry_add(
            self.target,
            [
                self.syn_match_table.make_key(
                    [
                        # gc.KeyTuple("hdr.ipv4.src_addr", self.syn_match_info.src_ip),
                        gc.KeyTuple("hdr.ipv4.dst_addr", self.syn_match_info.dst_ip),
                        gc.KeyTuple("hdr.tcp.syn", 1),
                        gc.KeyTuple("hdr.tcp.ack", 1),
                    ]
                )
            ],
            [
                self.syn_match_table.make_data(
                    [],
                    "Ingress.handle_synack",
                )
            ],
        )
        self.syn_match_table.entry_add(
            self.target,
            [
                self.syn_match_table.make_key(
                    [
                        # gc.KeyTuple("hdr.ipv4.src_addr", self.syn_match_info.src_ip),
                        gc.KeyTuple("hdr.ipv4.dst_addr", self.syn_match_info.dst_ip),
                        gc.KeyTuple("hdr.tcp.syn", 0),
                        gc.KeyTuple("hdr.tcp.ack", 1),
                    ]
                )
            ],
            [
                self.syn_match_table.make_data(
                    [],
                    "Ingress.miss",
                )
            ],
        )

    def reset(self):
        logger.info(f"reset syn_match_table")
        self.syn_match_table.entry_del(
            self.target,
            [entry_key for _, entry_key in self.syn_match_table.entry_get(self.target)],
        )
