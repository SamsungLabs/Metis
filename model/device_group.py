# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
import math
from collections import Counter
from typing import List, Dict, Tuple, TYPE_CHECKING

from utils import DeviceType
from model.load_balancer import DataLoadBalancer

if TYPE_CHECKING:
    from search_space.plan import InterStagePlan


class StagePerformance:
    def __init__(self, model_config, profile_data: Dict, gpu_cluster, plan: 'InterStagePlan'):
        self.model_config = model_config
        self.profile_data = profile_data
        self.gpu_cluster = gpu_cluster
        self.plan = plan
        self.rank_device_map = self._get_device_placement(plan.node_sequence)
        self.total_devices = gpu_cluster.get_total_num_devices()

    def _get_device_placement(self, node_sequence: List[DeviceType]) -> Dict[int, str]:
        rank_device_map = dict()

        device_types = []
        for device_type in node_sequence:
            num_dtype_devices = self.gpu_cluster.get_num_nodes_by_device_type(device_type.name)
            device_types += ([device_type.name] * num_dtype_devices)

        for device_rank in range(self.gpu_cluster.get_total_num_devices()):
            rank_device_map[device_rank] = device_types[device_rank]
        return rank_device_map

    def get_device_placement(self) -> Dict[int, str]:
        return self.rank_device_map

    def _get_execution_time(self, device_type: str, key: str) -> float:
        return sum(self.profile_data[f'DeviceType.{device_type}'][key]['time']['layer-computes'])

    def _get_hetero_device_group_execution_time(self, device_types: List[str], intra_strategy: Tuple[int, int],
                                                hetero_bs: List[int]) -> List[float]:
        dp_deg, tp_deg = intra_strategy
        execution_costs = []
        for dp_id, h_mbs in enumerate(hetero_bs):
            device_type = device_types[(len(device_types) // dp_deg) * dp_id]
            comb_h_mbs = [2 ** i for i in range(int(math.log2(h_mbs)) if h_mbs != 0 else 0, -1, -1) if h_mbs & 2 ** i]
            inner_dp_cost = 0.
            for h_mbs_slice in comb_h_mbs:
                inner_dp_cost += self._get_execution_time(device_type, key=f'tp{tp_deg}_bs{h_mbs_slice}')
            execution_costs.append(inner_dp_cost)

        return execution_costs

    def get_intra_stage_compute_performance(self, strategies: List[Tuple[int, int]], gbs: int, batches: int) -> List[float]:
        stage_performance = []
        for stage_id, intra_strategy in zip(range(len(self.plan.device_groups)), strategies):
            dp_deg, tp_deg = intra_strategy
            bs = gbs // batches // dp_deg

            start_rank = sum(self.plan.device_groups[:stage_id])
            end_rank = sum(self.plan.device_groups[:stage_id + 1])

            hetero_device_group = False
            device_types = [self.rank_device_map[rank] for rank in range(start_rank, end_rank)]
            if len(set(device_types)) > 1:
                hetero_device_group = True

            if hetero_device_group:
                data_load_balancer = DataLoadBalancer(self.profile_data, self.model_config)
                hetero_bs = data_load_balancer.partition_data(device_types, intra_strategy, gbs // batches)

                execution_costs = self._get_hetero_device_group_execution_time(device_types, intra_strategy, hetero_bs)
                cur_performance = 0
                if max(execution_costs) != 0:
                    cur_performance = 1. / max(execution_costs)
                stage_performance.append(cur_performance)
            else:
                device_type = device_types[0]
                profile_cost = self._get_execution_time(device_type, key=f'tp{tp_deg}_bs{bs}')
                stage_performance.append(1. / profile_cost)

        total_performance = sum(stage_performance)
        stage_compute_performance = [s_performance / total_performance for s_performance in stage_performance]

        return stage_compute_performance

    def get_device_group_memory_capacity(self) -> List[int]:
        stage_memory_capacity: List[int] = []
        for stage_id in range(len(self.plan.device_groups)):
            start_rank = sum(self.plan.device_groups[:stage_id])
            end_rank = sum(self.plan.device_groups[:stage_id + 1])

            device_types = [self.rank_device_map[rank] for rank in list(range(start_rank, end_rank))]
            device_type_dict = dict(Counter(device_types))

            inner_stage_memory_capacity = []
            for device_type in device_type_dict.keys():
                inner_stage_memory_capacity.append(
                    self.gpu_cluster.get_device_memory_for_device_type(device_type) * device_type_dict[device_type])
            stage_memory_capacity.append(sum(inner_stage_memory_capacity))
        return stage_memory_capacity
