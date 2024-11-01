# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict, Counter
from typing import Tuple, Dict, List, Optional
from numpy.typing import NDArray

from gpu_cluster import GPUCluster
from search_space.plan import InterStagePlan

class ClusterBandwidth(ABC):
    def __init__(self, gpu_cluster):
        self.gpu_cluster = gpu_cluster
        self.total_devices = gpu_cluster.get_total_num_devices()
        self.rank_map, self.rank_node_map = self._get_device_placement(num_nodes=gpu_cluster.get_num_nodes(),
                                                                       num_devices=gpu_cluster.get_num_devices_per_node())

    @abstractmethod
    def _get_dp_groups(self, *args):
        pass

    @abstractmethod
    def _get_pp_groups(self, *args):
        pass

    @abstractmethod
    def get_slowest_pp_bandwidth(self, *args):
        pass

    @abstractmethod
    def get_slowest_dp_bandwidth(self, *args):
        pass

    def _get_device_placement(self, num_nodes: int, num_devices: int) -> Tuple[Dict, Dict]:
        rank_map = defaultdict(list)
        rank_node_map = dict()

        nodes = [num_devices] * num_nodes

        counter = 0
        for node_num, device_count in zip(range(len(nodes)), nodes):
            for inner_loop in range(device_count):
                rank_map[node_num].append(counter)
                rank_node_map[counter] = node_num
                counter += 1

        return rank_map, rank_node_map

    def _get_intra_bandwidth(self, device_type: Optional[str] = None) -> int:
        if device_type is None:
            return self.gpu_cluster.get_intra_bandwidth(0)
        for node_id in self.gpu_cluster.nodes.keys():
            if self.gpu_cluster.nodes[node_id].device_type.name == device_type:
                return self.gpu_cluster.get_intra_bandwidth(node_id)

    def _get_inter_bandwidth(self, device_types: Optional[List] = None) -> int:
        if device_types is None:
            return self.gpu_cluster.get_inter_bandwidth(0)

        slowest_bandwidth = float('inf')
        for node_id in self.gpu_cluster.nodes.keys():
            for device_type in device_types:
                cur_device_type = self.gpu_cluster.nodes[node_id].device_type.name
                cur_bandwidth = self.gpu_cluster.get_inter_bandwidth(node_id)
                if cur_device_type == device_type and cur_bandwidth < slowest_bandwidth:
                    slowest_bandwidth = cur_bandwidth

        return slowest_bandwidth


class HomoClusterBandwidth(ClusterBandwidth):
    def __init__(self, gpu_cluster):
        super().__init__(gpu_cluster)

        self.inter_bandwidth = super()._get_inter_bandwidth()
        self.intra_bandwidth = super()._get_intra_bandwidth()

    def _check_devices_within_node(self, device_group):
        nodes = [self.rank_node_map[device_rank] for device_rank in device_group]
        if len(set(nodes)) == 1:
            return True

    def _get_model_groups(self, strategy: Tuple[int, int, int]) -> NDArray:
        pp_deg, tp_deg, dp_deg = strategy
        assert tp_deg * dp_deg * pp_deg == self.total_devices, \
            "There is an issue with the strategy for uniform partitioning."

        devices = np.array(range(self.total_devices)).reshape(pp_deg, -1, tp_deg)
        model_groups = np.concatenate(devices, axis=1)
        return model_groups

    def _get_pp_groups(self, model_groups: NDArray, tp_deg: int, stage_id: int) -> List:
        pp_groups = []
        for model_group in model_groups:
            for cur_tp in range(tp_deg):
                device_rank = model_group[stage_id * tp_deg + cur_tp]
                next_device_rank = model_group[(stage_id + 1) * tp_deg + cur_tp]
                pp_groups.append([device_rank, next_device_rank])

        return pp_groups

    def _get_dp_groups(self, strategy: Tuple[int, int, int]) -> List:
        (pp_deg, tp_deg, dp_deg) = strategy
        assert tp_deg * dp_deg * pp_deg == self.total_devices, \
            "There is an issue with the strategy for uniform partitioning."

        devices = np.array(range(self.total_devices)).reshape(pp_deg, -1, tp_deg)
        dp_groups = [devices[pp].flatten().tolist() for pp in range(devices.shape[0])]
        return dp_groups

    def get_slowest_pp_bandwidth(self, strategy: Tuple[int, int, int], stage_id: int) -> int:
        model_groups = self._get_model_groups(strategy)
        (pp_deg, tp_deg, dp_deg) = strategy

        assert stage_id < pp_deg, "stage_id cannot be greater than pp_deg."

        pp_groups = self._get_pp_groups(model_groups, tp_deg, stage_id)
        slowest_bandwidth = self.intra_bandwidth

        for pp_group in pp_groups:
            if not self._check_devices_within_node(pp_group):
                slowest_bandwidth = self.inter_bandwidth
        return slowest_bandwidth

    def get_slowest_dp_bandwidth(self, strategy: Tuple[int, int, int]) -> int:
        dp_groups = self._get_dp_groups(strategy)
        slowest_bandwidth = self.intra_bandwidth
        for dp_group in dp_groups:
            if not self._check_devices_within_node(dp_group):
                slowest_bandwidth = self.inter_bandwidth

        return slowest_bandwidth


class HetClusterBandwidth(ClusterBandwidth):
    def __init__(self, gpu_cluster: GPUCluster, plan: InterStagePlan):
        super().__init__(gpu_cluster)
        self.plan = plan

        self.node_sequence = plan.node_sequence
        self.device_groups = plan.device_groups

    def _get_pp_groups(self, stage_id: int) -> Tuple[List, List]:
        pp_group_ranks = [i for i in range(sum(self.device_groups[:stage_id]), sum(self.device_groups[:stage_id + 2]))]
        pp_group_nodes = [self.rank_node_map[device_rank] for device_rank in pp_group_ranks]
        return pp_group_ranks, pp_group_nodes

    def _get_dp_groups(self, stage_id: int, strategy: Tuple[int, int]) -> List:
        device_ranks = [i for i in range(sum(self.device_groups[:stage_id]), sum(self.device_groups[:stage_id + 1]))]
        dp_deg, tp_deg = strategy

        dp_groups = [[] for _ in  range(dp_deg)]
        for tp_idx in range(tp_deg):
            for dp_idx in range(dp_deg):
                dp_groups[dp_idx].append(device_ranks.pop(0))
        return dp_groups

    def _sorted_device_types_by_node_sequence(self) -> List[str]:
        device_names = [self.gpu_cluster.nodes[node_id].device_type.name for node_id in self.gpu_cluster.nodes.keys()]
        node_sequence = [device_type.name for device_type in self.plan.node_sequence]

        device_type_count = Counter(device_names)
        sorted_device_types = []
        for key in node_sequence:
            sorted_device_types.extend([key] * device_type_count[key])

        return sorted_device_types

    def get_slowest_pp_bandwidth(self, stage_id: int) -> int:
        sorted_device_types = self._sorted_device_types_by_node_sequence()
        pp_group_ranks, pp_group_nodes = self._get_pp_groups(stage_id)
        device_types = [sorted_device_types[node_id] for node_id in list(set(pp_group_nodes))]

        if len(device_types) == 1:
            return self._get_intra_bandwidth(device_types[0])
        else:
            return self._get_inter_bandwidth(device_types)

    def get_slowest_dp_bandwidth(self, strategy: Tuple[int, int], stage_id: int) -> int:
        slowest_bandwidth = float('inf')
        sorted_device_types = self._sorted_device_types_by_node_sequence()
        dp_groups = self._get_dp_groups(stage_id, strategy)
        for dp_group in dp_groups:
            group_nodes = [self.rank_node_map[device_rank] for device_rank in dp_group]
            device_types = [sorted_device_types[node_id] for node_id in list(set(group_nodes))]

            if len(device_types) == 1:
                bandwidth = self._get_intra_bandwidth(device_types[0])
            else:
                bandwidth = self._get_inter_bandwidth(device_types)

            if bandwidth < slowest_bandwidth:
                slowest_bandwidth = bandwidth

        return slowest_bandwidth
