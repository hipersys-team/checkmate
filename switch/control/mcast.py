from dataclasses import dataclass
import bfrt_grpc.client as gc
from typing import Optional, List

from l2switch import MachinePortInfo, SimpleL2Switch
import logging

logger = logging.getLogger(__name__)


@dataclass
class MulticastNodeConfigInfo:
    multicast_node: int
    multicast_rid: int
    multicast_group: int
    training_machine: str
    storage_machine: str

@dataclass
class MulticastArbitraryConfigInfo:
    multicast_node: int
    multicast_rid: int
    multicast_group: int
    machines: List[str]


class Multicast:

    def __init__(
        self,
        switch: SimpleL2Switch,
        nodes: Optional[dict[tuple[str, str], MulticastNodeConfigInfo]],
        arbitrary_group: Optional[List[MulticastArbitraryConfigInfo]]
    ):
        self.switch = switch
        self.bfrt_info = switch.bfrt_info
        self.target = switch.target
        self.nodes: dict[tuple[str, str], MulticastNodeConfigInfo] = nodes
        self.arbitrary_group: List[MulticastArbitraryConfigInfo] = arbitrary_group
        self.pre_node_table = self.bfrt_info.table_get("$pre.node")
        self.pre_mgid_table = self.bfrt_info.table_get("$pre.mgid")

        # self.update_table()

    def update_table(self):
        if self.nodes is not None:
            for mcast_cfg in self.nodes.values():
                logger.info(f"updateing {mcast_cfg}")
                self.pre_node_table.entry_add(
                    self.target,
                    [
                        self.pre_node_table.make_key(
                            [gc.KeyTuple("$MULTICAST_NODE_ID", mcast_cfg.multicast_node)]
                        )
                    ],
                    [
                        self.pre_node_table.make_data(
                            [
                                gc.DataTuple("$MULTICAST_RID", mcast_cfg.multicast_rid),
                                gc.DataTuple(
                                    "$DEV_PORT",
                                    int_arr_val=[
                                        self.switch.get_dev_port_id(
                                            mcast_cfg.training_machine
                                        ),
                                        self.switch.get_dev_port_id(
                                            mcast_cfg.storage_machine
                                        ),
                                    ],
                                ),
                            ]
                        )
                    ],
                )
                self.pre_mgid_table.entry_add(
                    self.target,
                    [
                        self.pre_mgid_table.make_key(
                            [gc.KeyTuple("$MGID", mcast_cfg.multicast_group)]
                        )
                    ],
                    [
                        self.pre_mgid_table.make_data(
                            [
                                gc.DataTuple(
                                    "$MULTICAST_NODE_ID",
                                    int_arr_val=[mcast_cfg.multicast_node],
                                ),
                                gc.DataTuple(
                                    "$MULTICAST_NODE_L1_XID_VALID", bool_arr_val=[False]
                                ),
                                gc.DataTuple("$MULTICAST_NODE_L1_XID", int_arr_val=[0]),
                            ]
                        )
                    ],
                )
        if self.arbitrary_group is not None:
            for mcast_cfg in self.arbitrary_group:
                logger.info(f"updateing {mcast_cfg}")
                self.pre_node_table.entry_add(
                    self.target,
                    [
                        self.pre_node_table.make_key(
                            [gc.KeyTuple("$MULTICAST_NODE_ID", mcast_cfg.multicast_node)]
                        )
                    ],
                    [
                        self.pre_node_table.make_data(
                            [
                                gc.DataTuple("$MULTICAST_RID", mcast_cfg.multicast_rid),
                                gc.DataTuple(
                                    "$DEV_PORT",
                                    int_arr_val=[
                                        self.switch.get_dev_port_id(
                                            x
                                        ) for x in mcast_cfg.machines
                                    ],
                                ),
                            ]
                        )
                    ],
                )
                self.pre_mgid_table.entry_add(
                    self.target,
                    [
                        self.pre_mgid_table.make_key(
                            [gc.KeyTuple("$MGID", mcast_cfg.multicast_group)]
                        )
                    ],
                    [
                        self.pre_mgid_table.make_data(
                            [
                                gc.DataTuple(
                                    "$MULTICAST_NODE_ID",
                                    int_arr_val=[mcast_cfg.multicast_node],
                                ),
                                gc.DataTuple(
                                    "$MULTICAST_NODE_L1_XID_VALID", bool_arr_val=[False]
                                ),
                                gc.DataTuple("$MULTICAST_NODE_L1_XID", int_arr_val=[0]),
                            ]
                        )
                    ],
                ) 
        # broadcast entry:
        self.pre_node_table.entry_add(
                self.target,
                [
                    self.pre_node_table.make_key(
                        [gc.KeyTuple("$MULTICAST_NODE_ID", 16383)]
                    )
                ],
                [
                    self.pre_node_table.make_data(
                        [
                            gc.DataTuple("$MULTICAST_RID", 16383),
                            gc.DataTuple(
                                "$DEV_PORT",
                                int_arr_val=[self.switch.get_dev_port_id(i) for i in self.switch.ports_info.keys()],
                            ),
                        ]
                    )
                ],
            )
        self.pre_mgid_table.entry_add(
            self.target,
            [
                self.pre_mgid_table.make_key(
                    [gc.KeyTuple("$MGID", 16383)]
                )
            ],
            [
                self.pre_mgid_table.make_data(
                    [
                        gc.DataTuple(
                            "$MULTICAST_NODE_ID",
                            int_arr_val=[16383],
                        ),
                        gc.DataTuple(
                            "$MULTICAST_NODE_L1_XID_VALID", bool_arr_val=[False]
                        ),
                        gc.DataTuple("$MULTICAST_NODE_L1_XID", int_arr_val=[0]),
                    ]
                )
            ],
        )

    def reset(self):
        # print(list(self.pre_node_table.entry_get(self.target)))
        if len(list(self.pre_node_table.entry_get(self.target))) != 0:
            self.pre_node_table.entry_del(
                self.target,
                [entry_key for _, entry_key in self.pre_node_table.entry_get(self.target)],
            )
        if len(list(self.pre_mgid_table.entry_get(self.target))) != 0:
            self.pre_mgid_table.entry_del(
                self.target,
                [entry_key for _, entry_key in self.pre_mgid_table.entry_get(self.target)],
            )
