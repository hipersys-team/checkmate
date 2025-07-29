from l2switch import MachinePortInfo, SimpleL2Switch, PortSpeed
from ingress import (
    FlowMatchInfo,
    FlowMatch,
    # McastPortMatchInfo,
    # McastMatch,
    SynMatchInfo,
    SynMatch,
)
from typing import Optional, List
from mirror_cfg import MirrorCfgInfo, MirrorCfg
from mcast import MulticastNodeConfigInfo, Multicast, MulticastArbitraryConfigInfo
from egress import EgressPortMatch, EgressPortMatchInfo

from dataclasses import dataclass, astuple
import logging

import itertools


logger = logging.getLogger(__name__)
logging.basicConfig(encoding="utf-8", level=logging.DEBUG)


def flatten_list(lst):
    return list(itertools.chain.from_iterable(lst))


@dataclass
class InnetCkptConfig:
    rank0_src: str
    rank0_dst: str
    rank0_ports: list[int]
    rank0_storage: list[str]
    lastrank_src: str
    lastrank_dst: str
    lastrank_ports: list[int]
    lastrank_storage: list[str]

    def __iter__(self):
        return iter(astuple(self))


class InnetCkpt:

    def __init__(
        self,
        ports_info: dict[str, MachinePortInfo],
        ckpt_config: list[InnetCkptConfig],
        training_machines: list[str],
        storage_machines: list[str],
        switch_fakeip: str,
        dryrun=False,
        arbitrary_mcast_group: Optional[List[List[str]]] = []
    ):
        self.port_info = ports_info
        # array, one entry for each DP partition
        self.ckpt_config = ckpt_config
        self.training_machines = training_machines
        self.storage_machines = storage_machines
        self.switch_fakeip = switch_fakeip
        self.arbitrary_mcast_group = arbitrary_mcast_group

        # mirror sessions needs to be generated first so the later
        # code knows which section to refer to for a specific port
        # The other two are independent of the dp rank as well.
        # self.mirror_cfg_info = self.generate_mirror_cfg_info()
        self.mcast_cfg_info = (
            self.generate_mcast_cfg_info()
        )
        self.syn_match_info = self.generate_syn_match_info()
        self.egress_match_info = self.generate_egress_match_info()

        # parallel arraies for each DP rank
        if self.arbitrary_mcast_group:
            self.flow_match_info = self.generate_arbitrary_mcast(self.mcast_cfg_info[0], self.mcast_cfg_info[1])
        else:
            self.flow_match_info, self.src_id_map = self.generate_flow_match_info()
        
        # self.mirror_port_match_info = self.generate_mcast_port_match_info()

        if not dryrun:
            self.switch = self.setup_switch()
            self.setup_match_tables()

        logger.info("Finished setting up.")

    def generate_mirror_cfg_info(self):
        result = {}
        for i, s in enumerate(self.storage_machines):
            result[s] = MirrorCfgInfo(i + 1, "INGRESS", s)
            logger.info(f"mirror cfg adding {s}: {result[s]}")
        return result

    def generate_mcast_cfg_info(self):
        """
        assigns each port a specific mirror session ID starting from 1.
        """
        mgid = 1
        if self.arbitrary_mcast_group:
            result = []
            for group in self.arbitrary_mcast_group:
                result.append(
                    MulticastArbitraryConfigInfo(
                        mgid, 
                        mgid,
                        mgid,
                        group
                    )
                )
                mgid += 1
            return result 
        
        else:
            result = {}
            for s in self.storage_machines:
                for t in self.training_machines:
                    result[(t, s)] = MulticastNodeConfigInfo(mgid, mgid, mgid, t, s)
                    mgid += 1
                    logger.info(f"mcast cfg adding {(t, s)}: {result[(t, s)]}")
            return result

    def get_ipv4_pair(self, src, dst):
        return (self.port_info[src].ipv4, self.port_info[dst].ipv4)

    def generate_arbitrary_mcast(self, r0_reps, rl_reps):
        result = []
        src_id_map = {}
        curr_srcid = 0
        for dp_par, (r0src, r0dst, r0p, r0s, lrsrc, lrdst, lrp, lrs) in enumerate(
            self.ckpt_config
        ):
            # dp_result = []
            # the logic for r0 and last rank are basically the same.
            # separating the code for now in case of future change
            for port in r0p:
                # for sid, storage in enumerate(r0s):
                result.append(
                    FlowMatchInfo(
                        self.port_info[r0src].ipv4,
                        self.port_info[r0dst].ipv4,
                        port,
                        r0_reps.multicast_node,
                        0,
                    )
                )
                logger.info(
                    f"flow match adding {result[-1]} from rank0, DP partition {dp_par}"
                )
            for port in lrp:
                result.append(
                    FlowMatchInfo(
                        self.port_info[lrsrc].ipv4,
                        self.port_info[lrdst].ipv4,
                        port,
                        rl_reps.multicast_node,
                        0,
                    )
                )
                logger.info(
                    f"flow match adding {result[-1]} from last rank, DP partition {dp_par}"
                )
        return result, src_id_map
            

    def generate_flow_match_info(self):
        result = []
        src_id_map = {}
        curr_srcid = 0
        for dp_par, (r0src, r0dst, r0p, r0s, lrsrc, lrdst, lrp, lrs) in enumerate(
            self.ckpt_config
        ):
            dp_result = []
            # the logic for r0 and last rank are basically the same.
            # separating the code for now in case of future change
            for port in r0p:
                for sid, storage in enumerate(r0s):
                    dp_result.append(
                        FlowMatchInfo(
                            self.port_info[r0src].ipv4,
                            self.port_info[r0dst].ipv4,
                            port,
                            self.mcast_cfg_info[r0dst, storage].multicast_group,
                            sid,
                        )
                    )
                    logger.info(
                        f"flow match adding {dp_result[-1]} from rank0, DP partition {dp_par}"
                    )
            src_id_map[self.get_ipv4_pair(r0src, r0dst)] = curr_srcid
            curr_srcid += 1
            for port in lrp:
                for sid, storage in enumerate(lrs):
                    dp_result.append(
                        FlowMatchInfo(
                            self.port_info[lrsrc].ipv4,
                            self.port_info[lrdst].ipv4,
                            port,
                            self.mcast_cfg_info[lrdst, storage].multicast_group,
                            sid,
                        )
                    )
                    logger.info(
                        f"flow match adding {dp_result[-1]} from last rank, DP partition {dp_par}"
                    )
            src_id_map[self.get_ipv4_pair(lrsrc, lrdst)] = curr_srcid
            curr_srcid += 1
            result.append(dp_result)
        return result, src_id_map

    # def generate_mcast_port_match_info(self):
    #     result = []
    #     added = {}
    #     for dp_par, (r0src, r0dst, _, r0s, lrsrc, lrdst, _, lrs) in enumerate(
    #         self.ckpt_config
    #     ):
    #         dp_result = []
    #         r0_src_id = self.src_id_map[self.get_ipv4_pair(r0src, r0dst)]
    #         for idx, storage in enumerate(r0s):
    #             dp_result.append(
    #                 McastPortMatchInfo(
    #                     r0_src_id, idx, self.mcast_cfg_info[(r0dst, storage)].multicast_group
    #                 )
    #             )
    #             logger.info(
    #                 f"mcast port match adding {dp_result[-1]} from rank0 ({r0src} -> {r0dst}, {storage}), DP partition {dp_par}"
    #             )
    #         lr_src_id = self.src_id_map[self.get_ipv4_pair(lrsrc, lrdst)]
    #         for idx, storage in enumerate(lrs):
    #             dp_result.append(
    #                 McastPortMatchInfo(
    #                     # lr_src_id, idx, self.mirror_cfg_info[storage].session_id
    #                     lr_src_id, idx, self.mcast_cfg_info[(lrdst, storage)].multicast_group
    #                 )
    #             )
    #             logger.info(
    #                 f"mcast port match adding {dp_result[-1]} from last rank ({lrsrc} -> {lrdst}, {storage}), DP partition {dp_par}"
    #             )
    #         result.append(dp_result)
    #     return result

    def generate_syn_match_info(self):
        # for now there seems to only be one switch fake IP needed.
        result = SynMatchInfo(self.switch_fakeip)
        logger.info(f"syn match adding result")
        return result

    def generate_egress_match_info(self):
        """
        Generate the port list that will need sequence swapping.
        For now all these ports with DSCP TAGGED bit set will be swapped
        """
        result = []
        for s in self.storage_machines:
            result.append(EgressPortMatchInfo(s))
            logger.info(f"egress seq swap adding {result[-1]}")
        return result

    def setup_switch(self):
        return SimpleL2Switch(self.port_info, self.switch_fakeip)

    def setup_match_tables(self):
        flow_match = FlowMatch(self.switch, flatten_list(self.flow_match_info))
        flow_match.reset()
        flow_match.update_table()
        # mirror_match = McastMatch(self.switch, flatten_list(self.mirror_port_match_info))
        # mirror_match.reset()
        # mirror_match.update_table()
        if self.arbitrary_mcast_group:
            mcast_cfg = Multicast(self.switch, nodes=None, arbitrary_group=self.mcast_cfg_info)
        else:
            mcast_cfg = Multicast(self.switch, self.mcast_cfg_info, arbitrary_group=None)
        mcast_cfg.reset()
        mcast_cfg.update_table()
        syn_match = SynMatch(self.switch, self.syn_match_info)
        syn_match.reset()
        syn_match.update_table()
        egress_match = EgressPortMatch(self.switch, self.egress_match_info)
        egress_match.reset()
        egress_match.update_table()

        # debug_mirror_cfg = MirrorCfgInfo(1, 'INGRESS', "xana")
        # # eg_debug_mirror_cfg = MirrorCfgInfo(2, 'EGRESS', "windows")
        # mirror_cfg_dbg = MirrorCfg(self.switch, {'windows': debug_mirror_cfg})
        # mirror_cfg_dbg.reset()
        # mirror_cfg_dbg.update_table()


PORTS_INFO = {
    "uther": MachinePortInfo(
        "uther", 12, 0, PortSpeed.BF_SPEED_100G, "98:03:9b:14:0c:70", "192.168.10.9"
    ),
    "venus": MachinePortInfo(
        "venus", 13, 0, PortSpeed.BF_SPEED_100G, "98:03:9b:15:b8:4a", "192.168.10.10"
    ),
    "windows": MachinePortInfo(
        "windows", 3, 0, PortSpeed.BF_SPEED_100G, "98:03:9b:15:b5:f6", "192.168.10.11"
    ),
    "xana": MachinePortInfo(
        "xana", 4, 0, PortSpeed.BF_SPEED_100G, "50:6b:4b:d7:52:98", "192.168.10.12"
    ),
    "midna": MachinePortInfo(
        "midna", 1, 0, PortSpeed.BF_SPEED_100G, "04:3f:72:d8:3d:48", "192.168.10.1"
    ),
    "navi": MachinePortInfo(
        "navi", 2, 0, PortSpeed.BF_SPEED_100G, "0c:42:a1:b4:38:74", "192.168.10.2"
    ),
    "oshun": MachinePortInfo(
        "oshun", 14, 0, PortSpeed.BF_SPEED_100G, "04:3f:72:d8:46:dc", "192.168.10.3"
    ),
    "parvati": MachinePortInfo(
        "parvati", 15, 0, PortSpeed.BF_SPEED_100G, "04:3f:72:d8:3d:40", "192.168.10.4"
    ),
    "quiritis": MachinePortInfo(
        "quiritis", 16, 0, PortSpeed.BF_SPEED_100G, "0c:42:a1:b4:38:c4", "192.168.10.5"
    ),
    "rhiannon": MachinePortInfo(
        "rhiannon", 17, 0, PortSpeed.BF_SPEED_100G, "04:3f:72:d8:3d:44", "192.168.10.6"
    ),
    "saria": MachinePortInfo(
        "saria", 5, 0, PortSpeed.BF_SPEED_100G, "0c:42:a1:ad:77:32", "192.168.10.7"
    ),
    "tara": MachinePortInfo(
        "tara", 6, 0, PortSpeed.BF_SPEED_100G, "0c:42:a1:b4:38:04", "192.168.10.8"
    ),
    "sr01p1": MachinePortInfo(
        "sr01p1", 18, 0, PortSpeed.BF_SPEED_100G, "04:3f:72:de:b9:64", "192.168.10.21"
    ),
    "sr02p1": MachinePortInfo(
        "sr02p1", 19, 0, PortSpeed.BF_SPEED_100G, "04:3f:72:de:b9:44", "192.168.10.22"
    ),
    "sr03": MachinePortInfo(
        "sr03", 7, 0, PortSpeed.BF_SPEED_100G, "50:6b:4b:dd:f0:de", "192.168.10.23"
    ),
    "sr04p1": MachinePortInfo(
        "sr04p1", 8, 0, PortSpeed.BF_SPEED_100G, "a0:88:c2:96:05:30", "192.168.10.24"
    ),
    "sr05p1": MachinePortInfo(
        "sr05p1", 9, 0, PortSpeed.BF_SPEED_100G, "a0:88:c2:96:07:5c", "192.168.10.25"
    ),
    "sr01p2": MachinePortInfo(
        "sr01p2", 20, 0, PortSpeed.BF_SPEED_100G, "04:3f:72:de:b3:08", "192.168.10.21"
    ),
    "sr02p2": MachinePortInfo(
        "sr02p2", 21, 0, PortSpeed.BF_SPEED_100G, "04:3f:72:de:b6:24", "192.168.10.22"
    ),
    "sr04p2": MachinePortInfo(
        "sr04p2", 10, 0, PortSpeed.BF_SPEED_100G, "a0:88:c2:96:06:0c", "192.168.10.24"
    ),
    "sr05p2": MachinePortInfo(
        "sr05p2", 11, 0, PortSpeed.BF_SPEED_100G, "a0:88:c2:96:06:f8", "192.168.10.25"
    ),
}

RANK0 = "uther"
RANK1 = "venus"
RANKN = "xana"

INNET_CKPT_CFG = [
    InnetCkptConfig(
        RANK0,
        RANK1,
        list(range(41004, 41008)),
        ["sr04p1", "sr05p1", "sr01p1", "sr02p1"],
        RANKN,
        RANK0,
        list(range(41000, 41004)),
        ["sr04p2", "sr05p2", "sr01p2", "sr02p2"],
    )
]

ALL_TRAINING = [
    "midna",
    "navi",
    "oshun",
    "parvati",
    "quiritis",
    "rhiannon",
    "saria",
    "tara",
    "uther",
    "venus",
    "windows",
    "xana",
]
ALL_STORAGE = [
    "sr04p1",
    "sr04p2",
    "sr05p1",
    "sr05p2",
    "sr01p1",
    "sr01p2",
    "sr02p1",
    "sr02p2",
]

PIPE0_RANK0 = "midna"
PIPE0_RANK1 = "navi"
PIPE0_RANKN = "rhiannon"

PIPE1_RANK0 = "saria"
PIPE1_RANK1 = "tara"
PIPE1_RANKN = "xana"

PIPE0_CFG = InnetCkptConfig(
    PIPE0_RANK0,
    PIPE0_RANK1,
    list(range(41004, 41008)),
    ["sr04p1", "sr05p1"],
    PIPE0_RANKN,
    PIPE0_RANK0,
    list(range(41000, 41004)),
    ["sr04p2", "sr05p2"],

)

PIPE1_CFG = InnetCkptConfig(
    PIPE1_RANK0,
    PIPE1_RANK1,
    list(range(41028, 41032)),
    ["sr01p1", "sr02p1"],
    PIPE1_RANKN,
    PIPE1_RANK0,
    list(range(41024, 41028)),
    ["sr01p2", "sr02p2"],

)

INNET_CKPT_CFG_PP = [
    PIPE0_CFG,
    PIPE1_CFG
]


INNET_CKPT_CFG = [
    InnetCkptConfig(
        "sr01p1",
        "sr02p1",
        list(range(41004, 41008)),
        ["sr04p1", "sr05p1"],
        "sr02p1",
        "sr01p1",
        list(range(41000, 41004)),
        ["sr04p2", "sr05p2"],
    )
]

INNET_CKPT_CFG_TEST_MCAST = [
    InnetCkptConfig(
        "uther",
        "venus",
        list(range(41004, 41008)),
        ["sr04p1", "sr05p1"],
        "xana",
        "uther",
        list(range(41000, 41004)),
        ["sr04p2", "sr05p2"],
    )
]
# STORAGE_REP = [
#     ["midna", "navi", "oshun", "parvati", "quiritis", "rhiannon", "saria", "tara", "venus"],
#     ["sr01p1", "sr01p2", "sr02p1", "sr02p2", "sr04p1", "sr04p2", "sr05p1", "sr05p2", "uther"]
# ]



def main():
    ckpt = InnetCkpt(
        PORTS_INFO,
        INNET_CKPT_CFG,
        ["sr01p1", "sr02p1"],
        ["sr04p1", "sr04p2", "sr05p1", "sr05p2"],
        "192.168.10.245",
    )
    # counter test config
    # ckpt = InnetCkpt(
    #     PORTS_INFO,
    #     INNET_CKPT_CFG_TEST_MCAST,
    #     ["uther", "venus", "windows", "xana"],
    #     ["midna", "navi", "oshun", "parvati", "quiritis", "rhiannon", "saria", "tara", "sr01p1", "sr01p2", "sr02p1", "sr02p2", "sr04p1", "sr04p2", "sr05p1", "sr05p2"],
    #     "192.168.10.245",
    #     arbitrary_mcast_group=STORAGE_REP,
    #     # dryrun=True 
    # )
    
    # pipeine config
    # ckpt = InnetCkpt(
    #     PORTS_INFO,
    #     INNET_CKPT_CFG_PP,
    #     ALL_TRAINING,
    #     ALL_STORAGE,
    #     "192.168.10.245",
    #     # dryrun=True
    # )

    # null config
    # ckpt = InnetCkpt(PORTS_INFO, [], [], [], "192.168.10.245")
    return ckpt


if __name__ == "__main__":
    ckpt = main()
