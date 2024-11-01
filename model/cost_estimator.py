# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
from abc import ABC, abstractmethod
import math
from functools import reduce
from typing import List, Dict, Tuple

from arguments import parse_args
from gpu_cluster import GPUCluster
from utils import ModelConfig
from model.utils import partition_layers_by_stage
from model.cluster_bandwidth import HomoClusterBandwidth, HetClusterBandwidth
from model.load_balancer import DataLoadBalancer
from search_space.plan import UniformPlan, InterStagePlan


class CostEstimator(ABC):
    def __init__(self, profile_data: Dict, model_config: ModelConfig, model_volume, gpu_cluster: GPUCluster):
        self.profile_data = profile_data
        self.model_config = model_config
        self.model_volume = model_volume
        self.gpu_cluster = gpu_cluster

    @abstractmethod
    def _get_execution_cost(self, *args):
        pass

    @abstractmethod
    def get_cost(self, *args):
        pass

    def _detect_oom_occurrence(self, stage_memory: List[int]) -> bool:
        return True if self.gpu_cluster.get_device_memory(0) < max(stage_memory) else False

    def _get_batch_generate_cost(self, batches: int) -> float:
        return self.profile_data["model"]["batch_generator"] * batches

    def _get_dp_cost(self, stage_parameters: List[int], bandwidth:int, dp_deg: int) -> float:
        max_parameter_size = max(stage_parameters)

        bandwidth *= 1024 * 1024
        dp_const = 2 * (dp_deg - 1) / (dp_deg * bandwidth)
        dp_cost = dp_const * max_parameter_size
        return dp_cost

    def _get_pp_cost(self, activation_size: int, bandwidth: int) -> float:
        bandwidth *= 1024 * 1024
        return activation_size / bandwidth

    def _get_parameter_update_cost(self, *args) -> float:
        profile_time = self.profile_data["model"]["optimizer_time"]
        return self._get_specific_parameter_update_cost(profile_time, *args)

    @abstractmethod
    def _get_specific_parameter_update_cost(self, profile_time, *args):
        pass

    def _get_fb_sync_cost(self, device_types: List[str], tp_deg: int, batch_size: int) -> float:
        if device_types is None:
            device_types = [next(iter(self.profile_data))]

        def _get_nested_value(d, keys):
            return reduce(lambda d, key: d.get(key) if d else None, keys, d)

        fb_sync_costs = []
        for device_type in device_types:
            nested_keys = [f'DeviceType.{device_type}', f'tp{tp_deg}_bs{batch_size}', 'time', 'fb_sync']
            fb_sync_cost = _get_nested_value(self.profile_data, nested_keys)
            if not fb_sync_cost:
                raise KeyError(f"key(fb_sync) not found in profile_data")
            fb_sync_costs.append(fb_sync_cost)

        return max(fb_sync_costs)

    def _get_demand_device_memory(self, device_type: str, start_layer_id: int, end_layer_id: int, tp_deg: int, bs: int)\
            -> int:
        key = f'tp{tp_deg}_bs{bs}'
        if key not in self.profile_data[f'DeviceType.{device_type}'].keys():
            raise KeyError(f"key({key}) not found in profile_data")

        return sum(self.profile_data[f'DeviceType.{device_type}'][key]['memory'][start_layer_id: end_layer_id])


class HomoCostEstimator(CostEstimator):
    def __init__(self, profile_data: Dict, model_config: ModelConfig, model_volume, gpu_cluster: GPUCluster):
        super().__init__(profile_data, model_config, model_volume, gpu_cluster)
        self.cluster_bandwidth = HomoClusterBandwidth(gpu_cluster)

    def _get_specific_parameter_update_cost(self, optimizer_time: float, pp_deg: int, tp_deg: int) -> float:
        return optimizer_time / pp_deg / tp_deg

    def _get_execution_cost(self, device_type: str, start_layer_id: int, end_layer_id: int, tp_deg: int, batch_size: int):
        key = f'tp{tp_deg}_bs{batch_size}'
        if key not in self.profile_data[f'DeviceType.{device_type}'].keys():
            raise KeyError(f"key({key}) not found in profile_data")

        return sum(self.profile_data[f'DeviceType.{device_type}'][key]['time']['layer-computes'][start_layer_id:end_layer_id])

    def get_cost(self, plan: UniformPlan, device_type: str) -> Tuple[float, List, bool]:
        tp_deg, pp_deg, dp_deg = plan.tp, plan.pp, plan.dp

        stage_parameters = []
        model_parameters = self.model_volume.get_parameter_size(tp_deg)

        stage_layer_counts = partition_layers_by_stage(self.model_volume.get_num_layers(), pp_deg)
        bs = plan.mbs
        num_mbs = plan.gbs // plan.mbs // plan.dp

        lens, stage_memory = [], []
        pp_cost, fb_sync_cost = 0., 0.
        for stage_id in range(len(stage_layer_counts)):
            start_layer_id, end_layer_id = sum(stage_layer_counts[:stage_id]), sum(stage_layer_counts[:stage_id + 1])

            lens.append(self._get_execution_cost(device_type, start_layer_id, end_layer_id, tp_deg, bs))
            stage_parameters.append(sum(model_parameters[start_layer_id:end_layer_id]))

            demand_memory = self._get_demand_device_memory(device_type, start_layer_id, end_layer_id, tp_deg, bs)
            stage_memory.append(demand_memory)

            if stage_id == (len(stage_layer_counts) - 1):
                fb_sync_cost = self._get_fb_sync_cost([device_type], tp_deg, bs) * num_mbs
            else:
                activation_size = self.model_volume.get_activation_size(end_layer_id, bs, tp_deg)

                pp_bandwidth = self.cluster_bandwidth.get_slowest_pp_bandwidth((pp_deg, tp_deg, dp_deg), stage_id)
                pp_cost += self._get_pp_cost(activation_size, pp_bandwidth)

        oom_detected = self._detect_oom_occurrence(stage_memory)
        max_l = max(lens)
        execution_cost = ((num_mbs - 1) * max_l) + sum(lens)
        parameter_update_cost = self._get_parameter_update_cost(pp_deg, tp_deg)

        dp_bandwidth = self.cluster_bandwidth.get_slowest_dp_bandwidth((pp_deg, tp_deg, dp_deg))
        dp_cost = self._get_dp_cost(stage_parameters, dp_bandwidth, dp_deg)

        batch_generate_cost = self._get_batch_generate_cost(num_mbs)
        time_cost = execution_cost + fb_sync_cost + parameter_update_cost + dp_cost + pp_cost + batch_generate_cost
        stage_memory = [f'{round(cur_memory/1024/1024/1024, 2)}GB' for cur_memory in stage_memory]
        return time_cost, stage_memory, oom_detected


class HeteroCostEstimator(CostEstimator):
    def __init__(self, profile_data: Dict, model_config: ModelConfig, model_volume, gpu_cluster: GPUCluster):
        super().__init__(profile_data, model_config, model_volume, gpu_cluster)

    def _get_specific_parameter_update_cost(self, optimizer_time: float, tp_deg: int, num_layers: int) -> float:
        ratio = num_layers / self.model_config.num_layers
        return optimizer_time / tp_deg * ratio

    def _get_execution_time(self, device_type: str, key: str, start_layer_id: int, end_layer_id: int) -> float:
        return sum(self.profile_data[f'DeviceType.{device_type}'][key]['time']['layer-computes'][start_layer_id:end_layer_id])

    def _get_hetero_device_group_execution_time(self, device_types: List[str], intra_strategy: Tuple[int, int],
                                                hetero_bs: List[int], start_layer_id: int, end_layer_id: int) -> List[float]:
        args = parse_args()
        dp_deg, tp_deg = intra_strategy
        execution_costs = []
        for dp_id, h_mbs in enumerate(hetero_bs):
            if h_mbs == 0:
                continue

            device_type = device_types[(len(device_types) // dp_deg) * dp_id]
            comb_h_mbs = [2 ** i for i in range(int(math.log2(h_mbs)), -1, -1) if h_mbs & 2 ** i]
            inner_dp_cost = 0.

            for h_mbs_slice in comb_h_mbs:
                if h_mbs_slice > args.max_profiled_batch_size:
                    raise KeyError(f"batch_size({h_mbs_slice}) not found in profile_data")

                inner_dp_cost += self._get_execution_time(device_type, f'tp{tp_deg}_bs{h_mbs_slice}',
                                                          start_layer_id, end_layer_id)
            execution_costs.append(inner_dp_cost)

        return execution_costs

    def _get_execution_cost(self, device_types: List[str], start_layer_id: int, end_layer_id:int,
                            intra_strategy: Tuple[int, int], gbs: int, batches: int) -> float:
        dp_deg, tp_deg = intra_strategy

        # homogeneous device group
        if len(set(device_types)) == 1:
            device_type = device_types[0]
            key = f'tp{tp_deg}_bs{gbs // dp_deg // batches}'
            if key not in self.profile_data[f'DeviceType.{device_type}'].keys():
                raise KeyError(f"key({key}) not found in profile_data")

            profile_time = self.profile_data[f'DeviceType.{device_type}'][key]['time']['layer-computes']
            execution_cost = sum(profile_time[start_layer_id:end_layer_id])
            return execution_cost
        # heterogeneous device group
        else:
            data_load_balancer = DataLoadBalancer(self.profile_data, self.model_config)
            hetero_bs = data_load_balancer.partition_data(device_types, intra_strategy, gbs // batches)
            print(f'data loadbalancer: {hetero_bs}')

            execution_costs = self._get_hetero_device_group_execution_time(device_types, intra_strategy, hetero_bs,
                                                                           start_layer_id, end_layer_id)
            return max(execution_costs)

    def get_cost(self, plan: InterStagePlan, strategies: List[Tuple[int, int]], layer_partition: List[int],
                 rank_device_map: Dict[int, str]) -> float:
        print(f'node_sequence: {plan.node_sequence}, device_group: {plan.device_groups}, num_stage: {plan.num_stage}, '
              f'batches: {plan.batches}, gbs: {plan.gbs}, strategies: {strategies}, '
              f'layer_partition: {layer_partition}')

        cluster_bandwidth = HetClusterBandwidth(self.gpu_cluster, plan)

        lens = []
        pp_cost, dp_costs, fb_sync_cost, parameter_update_costs = 0., [], 0., []
        for stage_id, intra_strategy in zip(range(plan.num_stage), strategies):
            start_layer_id, end_layer_id = layer_partition[stage_id], layer_partition[stage_id + 1]

            start_rank = sum(plan.device_groups[:stage_id])
            end_rank = sum(plan.device_groups[:stage_id + 1])
            device_types = [rank_device_map[rank] for rank in range(start_rank, end_rank)]

            execution_cost = self._get_execution_cost(device_types, start_layer_id, end_layer_id, intra_strategy, plan.gbs, plan.batches)
            lens.append(execution_cost)

            dp_deg, tp_deg = intra_strategy
            mbs = plan.gbs // dp_deg // plan.batches
            if stage_id == (plan.num_stage - 1):
                fb_sync_cost = self._get_fb_sync_cost(device_types, tp_deg, mbs) * plan.batches
            else:
                #pp communication cost
                activation_size = self.model_volume.get_activation_size(end_layer_id, mbs, tp_deg)
                pp_bandwidth = cluster_bandwidth.get_slowest_pp_bandwidth(stage_id)
                pp_cost += self._get_pp_cost(activation_size, pp_bandwidth)

            #dp communication cost
            stage_parameters = self.model_volume.get_parameter_size_by_stage(tp_deg, start_layer_id, end_layer_id)
            dp_bandwidth = cluster_bandwidth.get_slowest_dp_bandwidth(intra_strategy, stage_id)
            dp_costs.append(self._get_dp_cost([stage_parameters], dp_bandwidth, dp_deg))
            parameter_update_costs.append(self._get_parameter_update_cost(tp_deg, (end_layer_id - start_layer_id)))

        max_l = max(lens)
        execution_cost = ((plan.batches - 1) * max_l) + sum(lens)
        batch_generate_cost = self._get_batch_generate_cost(plan.batches)

        print(f'execution_cost: {execution_cost}, fb_sync_cost: {fb_sync_cost}, '
              f'parameter_upate_costs: {max(parameter_update_costs)}, dp_cost: {max(dp_costs)}, pp_cost: {pp_cost}')
        time_cost = (execution_cost + fb_sync_cost + max(parameter_update_costs) + max(dp_costs) + pp_cost
                     + batch_generate_cost)

        return time_cost
