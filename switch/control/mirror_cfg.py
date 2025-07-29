from dataclasses import dataclass
import bfrt_grpc.client as gc

from l2switch import MachinePortInfo, SimpleL2Switch
import logging

logger = logging.getLogger(__name__)


@dataclass
class MirrorCfgInfo:
    """
    Full info dump of this table:
        In [36]: mirror_cfg_table.info.data_field_name_list_get()
        Out[36]:
        ['$session_enable',
        '$direction',
        '$ucast_egress_port',
        '$ucast_egress_port_valid',
        '$egress_port_queue',
        '$ingress_cos',
        '$packet_color',
        '$level1_mcast_hash',
        '$level2_mcast_hash',
        '$mcast_grp_a',
        '$mcast_grp_a_valid',
        '$mcast_grp_b',
        '$mcast_grp_b_valid',
        '$mcast_l1_xid',
        '$mcast_l2_xid',
        '$mcast_rid',
        '$icos_for_copy_to_cpu',
        '$copy_to_cpu',
        '$max_pkt_len']
    """

    session_id: int
    direction: str
    ucast_egress_machine: str


class MirrorCfg:

    def __init__(
        self, switch: SimpleL2Switch, mirror_cfgs_info: dict[str, MirrorCfgInfo]
    ):
        self.switch = switch
        self.bfrt_info = switch.bfrt_info
        self.target = switch.target
        self.mirror_cfgs_info: list[MirrorCfgInfo] = list(mirror_cfgs_info.values())
        self.mirror_cfg_table = self.bfrt_info.table_get("$mirror.cfg")

        # self.update_table()

    def update_table(self):
        for mirror_cfg in self.mirror_cfgs_info:
            logger.info(f"updateing {mirror_cfg}")
            self.mirror_cfg_table.entry_add(
                self.target,
                [
                    self.mirror_cfg_table.make_key(
                        [gc.KeyTuple("$sid", mirror_cfg.session_id)]
                    )
                ],
                [
                    self.mirror_cfg_table.make_data(
                        [
                            gc.DataTuple("$session_enable", bool_val=True),
                            gc.DataTuple("$direction", str_val=mirror_cfg.direction),
                            gc.DataTuple(
                                "$ucast_egress_port",
                                self.switch.get_dev_port_id(
                                    mirror_cfg.ucast_egress_machine
                                ),
                            ),
                            gc.DataTuple("$ucast_egress_port_valid", bool_val=True),
                            gc.DataTuple("$max_pkt_len", 16384),
                        ],
                        "$normal",
                    )
                ],
            )

    def reset(self):
        logger.info(f"reset mirror_cfg")
        if len(list(self.mirror_cfg_table.entry_get(self.target))) != 0:
            self.mirror_cfg_table.entry_del(
                self.target,
                [
                    entry_key
                    for _, entry_key in self.mirror_cfg_table.entry_get(self.target)
                ],
            )
