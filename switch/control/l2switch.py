from dataclasses import dataclass
import bfrt_grpc.client as gc
from enum import Enum
from portmap import ConnChannelMapper


class PortSpeed(Enum):
    BF_SPEED_100G = "BF_SPEED_100G"
    BF_SPEED_50G = "BF_SPEED_50G"
    BF_SPEED_40G = "BF_SPEED_40G"
    BF_SPEED_25G = "BF_SPEED_25G"
    BF_SPEED_10G = "BF_SPEED_10G"


@dataclass
class MachinePortInfo:
    name: str
    port_id: int
    channel_id: int
    speed: PortSpeed
    mac: str
    ipv4: str


class SimpleL2Switch:

    def __init__(
        self,
        ports_info: list[MachinePortInfo],
        fake_ip: str,
        pipeline_name: str = "ckpt",
        sw_conn: str = "localhost:50052",
        pre_port_devid: int = 32,
    ):
        self.cmp = ConnChannelMapper()
        self.interface = gc.ClientInterface(sw_conn, client_id=0, device_id=0)
        self.interface.bind_pipeline_config(pipeline_name)
        self.bfrt_info = self.interface.bfrt_info_get()
        self.target = gc.Target(device_id=0, pipe_id=0xFFFF)
        self.pipe0 = gc.Target(device_id=0, pipe_id=0)
        self.pipe1 = gc.Target(device_id=0, pipe_id=1)
        self.ports_info: dict[str, MachinePortInfo] = ports_info
        self.fake_ip = fake_ip
        self.pre_port = self.bfrt_info.table_get("$pre.port")
        self.port_table = self.bfrt_info.table_get("$PORT")
        self.port_stats_table = self.bfrt_info.table_get("$PORT_STAT")
        self.port_hdl_info_table = self.bfrt_info.table_get("$PORT_HDL_INFO")
        self.forward_table = self.bfrt_info.table_get("Ingress.forward")
        self.forward_table.info.key_field_annotation_add("hdr.ethernet.dst_addr", "mac")
        self.flow_control_table = self.bfrt_info.table_get("tf1.tm.port.flowcontrol")

        # static arp table on the switch for now.
        self.arp_table = self.bfrt_info.table_get("Ingress.arp")
        self.arp_table.info.key_field_annotation_add("hdr.arp.tgt_spa", "ipv4")
        self.arp_table.info.data_field_annotation_add(
            "resp_mac", "Ingress.fill_arp_resp", "mac"
        )

        self.port_cfg_table = self.bfrt_info.table_get("tf1.tm.port.cfg")
        self.ppg_cfg_table = self.bfrt_info.table_get("tf1.tm.ppg.cfg")
        self.pool_cfg_table = self.bfrt_info.table_get("tf1.tm.pool.cfg")
        self.queue_buffer_table = self.bfrt_info.table_get("tf1.tm.queue.buffer")
        self.curr_ppgid = 1

        # self.setup_pfc()

        self.pre_port_devid = pre_port_devid

        self.reset_pre_port_table()
        self.fill_pre_port_table()

        self.reset_port_table()
        self.fill_port_table()

        self.reset_forward_info()
        self.fill_forawrd_info()

        self.reset_arp_table()
        self.fill_arp_table()

    def setup_pfc(self):
        port_ids = [
            self.cmp.lookup(e.port_id, e.channel_id) for e in self.ports_info.values()
        ]
        for port_id in port_ids:
            self.set_port_queue3_icos3_to_pfc(port_id)

    def get_port_cfg(self, port_id):
        # print(f"{port_id:x}")
        key = self.port_cfg_table.make_key([gc.KeyTuple("dev_port", port_id)])
        (d, k) = next(self.port_cfg_table.entry_get(self.target, [key]))
        return d.to_dict()

    def set_port_queue3_icos3_to_pfc(self, port_id):
        # get the pgid, pgqueue of QID 3 on this port
        port_cfg = self.get_port_cfg(port_id)
        pg_id = port_cfg["pg_id"]
        pg_queue = port_cfg["pg_port_nr"] * 8 + 3
        ig_queue3_map_to_eg_queueid = port_cfg["ingress_qid_map"][3]
        eg_queue_to_eg_pg_queue = port_cfg["egress_qid_queues"][3]
        # setup ppg
        for pipe in [self.pipe0, self.pipe1]:
            key = self.ppg_cfg_table.make_key([gc.KeyTuple("ppg_id", self.curr_ppgid)])
            data = self.ppg_cfg_table.make_data(
                [
                    gc.DataTuple("dev_port", port_id),
                    gc.DataTuple("pfc_skid_max_cells", 8192),
                    gc.DataTuple("icos_3", bool_val=True),
                    gc.DataTuple("hysteresis_cells", 512),
                    gc.DataTuple("guaranteed_cells", 512),
                    gc.DataTuple("pool_id", str_val="IG_APP_POOL_1"),
                    gc.DataTuple("pool_max_cells", 512),
                    gc.DataTuple("pfc_enable", bool_val=True),
                ],
                "dev_port",
            )

            try:
                self.ppg_cfg_table.entry_add(pipe, [key], [data])
                print(
                    f"bfrt.tf1.tm.ppg.cfg.mod_with_dev_port(pipe={pipe.pipe_id_}, ppg_id={self.curr_ppgid}, dev_port={port_id}, pfc_enable=True)"
                )
            except:
                pass
        key = self.flow_control_table.make_key([gc.KeyTuple("dev_port", port_id)])
        data = self.flow_control_table.make_data(
            [
                gc.DataTuple("mode_tx", str_val="PFC"),
                gc.DataTuple("mode_rx", str_val="PFC"),
                gc.DataTuple("cos_to_icos", int_arr_val=[0, 1, 2, 3, 4, 5, 6, 7]),
            ]
        )
        # enable PFC on flow control
        self.flow_control_table.entry_mod(self.target, [key], [data])
        print(
            f'bfrt.tf1.tm.port.flowcontrol.mod(dev_port={port_id}, mode_tx="PFC", mode_rx="PFC", cos_to_icos=[0,1,2,3,4,5,6,7])'
        )
        # egress pool
        key = self.pool_cfg_table.make_key([gc.KeyTuple("pool", "EG_APP_POOL_1")])
        data = self.pool_cfg_table.make_data([gc.DataTuple("size_cells", 65536)])
        self.pool_cfg_table.entry_mod(self.target, [key], [data])

        # egress queue
        for pipe in [self.pipe0, self.pipe1]:
            key = self.queue_buffer_table.make_key(
                [
                    gc.KeyTuple("pg_id", pg_id),
                    gc.KeyTuple("pg_queue", eg_queue_to_eg_pg_queue),
                ]
            )
            data = self.queue_buffer_table.make_data(
                [
                    gc.DataTuple("guaranteed_cells", 4096),
                    gc.DataTuple("pool_id", str_val="EG_APP_POOL_1"),
                    gc.DataTuple("pool_max_cells", 2048),
                    gc.DataTuple("hysteresis_cells", 1024),
                    gc.DataTuple("dynamic_baf", str_val="DISABLE"),
                    gc.DataTuple("tail_drop_enable", bool_val=False),
                ],
                "shared_pool",
            )
            self.queue_buffer_table.entry_mod(pipe, [key], [data])

        self.curr_ppgid += 1

    def reset_arp_table(self):
        self.arp_table.entry_del(
            self.target,
            [entry_key for _, entry_key in self.arp_table.entry_get(self.target)],
        )

    def fill_arp_table(self):
        # for port in self.ports_info.values():
        #     self.arp_table.entry_add(
        #         self.target,
        #         [
        #             self.arp_table.make_key(
        #                 [
        #                     gc.KeyTuple("hdr.arp.opcode", 1),
        #                     gc.KeyTuple("hdr.arp.tgt_spa", port.ipv4),
        #                 ]
        #             )
        #         ],
        #         [
        #             self.arp_table.make_data(
        #                 [
        #                     gc.DataTuple("resp_mac", port.mac),
        #                 ], "fill_arp_resp"
        #             )
        #         ],
        #     )
        self.arp_table.entry_add(
            self.target,
            [
                self.arp_table.make_key(
                    [
                        gc.KeyTuple("hdr.arp.opcode", 1),
                        gc.KeyTuple("hdr.arp.tgt_spa", self.fake_ip),
                    ]
                )
            ],
            [
                self.arp_table.make_data(
                    [
                        gc.DataTuple("resp_mac", "aa:aa:aa:aa:aa:aa"),
                    ],
                    "fill_arp_resp",
                )
            ],
        )

    def get_dev_port_id(self, machine):
        return self.cmp.lookup(
            self.ports_info[machine].port_id, self.ports_info[machine].channel_id
        )

    def fill_pre_port_table(self):
        self.pre_port.entry_add(
            self.target,
            [self.pre_port.make_key([gc.KeyTuple("$DEV_PORT", self.pre_port_devid)])],
            [
                self.pre_port.make_data(
                    [gc.DataTuple("$COPY_TO_CPU_PORT_ENABLE", bool_val=False)]
                )
            ],
        )

    def reset_pre_port_table(self):
        self.pre_port.entry_del(
            self.target,
            [entry_key for _, entry_key in self.pre_port.entry_get(self.target)],
        )

    def fill_port_table(self):
        outport = [
            (self.cmp.lookup(port_info.port_id, port_info.channel_id), port_info.speed)
            for port_info in self.ports_info.values()
        ]
        for port, speed in outport:
            self.port_table.entry_add(
                self.target,
                [self.port_table.make_key([gc.KeyTuple("$DEV_PORT", port)])],
                [
                    self.port_table.make_data(
                        [
                            gc.DataTuple("$SPEED", str_val=speed.value),
                            gc.DataTuple("$FEC", str_val="BF_FEC_TYP_RS"),
                            gc.DataTuple("$PORT_ENABLE", bool_val=True),
                            gc.DataTuple("$TX_PAUSE_FRAME_EN", bool_val=True),
                            gc.DataTuple("$RX_PAUSE_FRAME_EN", bool_val=True),
                            gc.DataTuple("$TX_PFC_EN_MAP", 0xFFFFFFFF),
                            gc.DataTuple("$RX_PFC_EN_MAP", 0xFFFFFFFF),
                            gc.DataTuple(
                                "$AUTO_NEGOTIATION", str_val="PM_AN_FORCE_DISABLE"
                            ),
                        ]
                    )
                ],
            )

    def reset_port_table(self):
        self.port_table.entry_del(
            self.target,
            [entry_key for _, entry_key in self.port_table.entry_get(self.target)],
        )

    def fill_forawrd_info(self):
        for port_info in self.ports_info.values():
            self.forward_table.entry_add(
                self.target,
                [
                    self.forward_table.make_key(
                        [gc.KeyTuple("hdr.ethernet.dst_addr", port_info.mac)]
                    )
                ],
                [
                    self.forward_table.make_data(
                        [
                            gc.DataTuple(
                                "dst_port",
                                self.cmp.lookup(
                                    port_info.port_id, port_info.channel_id
                                ),
                            )
                        ],
                        "Ingress.hit",
                    )
                ],
            )
        self.forward_table.entry_add(
            self.target,
            [
                self.forward_table.make_key(
                    [gc.KeyTuple("hdr.ethernet.dst_addr", "ff:ff:ff:ff:ff:ff")]
                )
            ],
            [
                self.forward_table.make_data(
                    [
                        gc.DataTuple(
                            "mgid",
                            16383,
                        )
                    ],
                    "Ingress.broadcast",
                )
            ],
        )

    def reset_forward_info(self):
        self.forward_table.entry_del(
            self.target,
            [entry_key for _, entry_key in self.forward_table.entry_get(self.target)],
        )
