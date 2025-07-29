from dataclasses import dataclass
import bfrt_grpc.client as gc

from l2switch import MachinePortInfo, SimpleL2Switch
import logging

logger = logging.getLogger(__name__)


@dataclass
class EgressPortMatchInfo:
    port_name: str


class EgressPortMatch:

    def __init__(
        self, switch: SimpleL2Switch, egrport_matches_info: list[EgressPortMatchInfo]
    ):
        self.switch = switch
        self.bfrt_info = switch.bfrt_info
        self.target = switch.target
        self.egrport_matches_info: list[EgressPortMatchInfo] = egrport_matches_info
        self.egress_swap_table = self.bfrt_info.table_get("Egress.egress_swapseq_match")
        self.egress_swap_table.info.data_field_annotation_add(
            "dst_mac", "Egress.swap_seq", "mac"
        )
        self.egress_swap_table.info.data_field_annotation_add(
            "src", "Egress.swap_seq", "ipv4"
        )
        self.egress_swap_table.info.data_field_annotation_add(
            "dst", "Egress.swap_seq", "ipv4"
        )

    def update_table(self):
        for egrport in self.egrport_matches_info:
            logger.info(f"updating {egrport}")
            self.egress_swap_table.entry_add(
                self.target,
                [
                    self.egress_swap_table.make_key(
                        [
                            gc.KeyTuple(
                                "eg_intr_md.egress_port",
                                self.switch.get_dev_port_id(egrport.port_name),
                            ),
                            gc.KeyTuple("hdr.ipv4.tag", 1),
                        ]
                    )
                ],
                [
                    self.egress_swap_table.make_data(
                        [
                            gc.DataTuple(
                                "dst_mac", self.switch.ports_info[egrport.port_name].mac
                            ),
                            gc.DataTuple("src", self.switch.fake_ip),
                            gc.DataTuple(
                                "dst", self.switch.ports_info[egrport.port_name].ipv4
                            ),
                        ],
                        "Egress.swap_seq",
                    )
                ],
            )

    def reset(self):
        logger.info(f"reset egress_match")
        self.egress_swap_table.entry_del(
            self.target,
            [
                entry_key
                for _, entry_key in self.egress_swap_table.entry_get(self.target)
            ],
        )
