# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
import copy
import math
from collections import Counter
from typing import List, Tuple, Dict, Union, TYPE_CHECKING

from gpu_cluster import GPUCluster
from utils import DeviceType

if TYPE_CHECKING:
    from search_space.plan import InterStagePlan


class LayerLoadBalancer:
    def __init__(self, gpu_cluster: GPUCluster, profile_data: Dict, model_config, gbs: int):
        self.gpu_cluster = gpu_cluster
        self.profile_data = profile_data
        self.model_config = model_config
        self.gbs = gbs
        self.norm_layer_duration = self._get_nomalize_layer_duration()

    def _get_nomalize_layer_duration(self) -> List[float]:
        device_type = next(iter(self.profile_data))
        layers_duration = self.profile_data[device_type]['tp1_bs1']['time']['layer-computes']

        total_layer_duration = sum(layers_duration)
        return [layer_duration / total_layer_duration for layer_duration in layers_duration]

    def _get_stage_memory_demand(self, layer_partition: List[int], strategies: List[Tuple[int, int]],
                                 device_group: List[int], device_types: List[str], gbs: int, batches: int,
                                 mem_coef: float = 5.0) -> List[float]:
        stage_memory = []
        for stage_id, strategy in enumerate(strategies):
            dp_deg, tp_deg = strategy
            start_rank = sum(device_group[:stage_id])
            end_rank = sum(device_group[:stage_id + 1])
            cur_device_types = [device_types[rank] for rank in range(start_rank, end_rank)]

            start_layer_id, end_layer_id = layer_partition[stage_id], layer_partition[stage_id + 1]
            cur_stage_memory_demand = 0.001
            if len(set(cur_device_types)) == 1:
                bs = gbs // batches // dp_deg
                profile_memory = self.profile_data[f'DeviceType.{device_types[0]}'][f'tp{tp_deg}_bs{bs}']['memory']
                cur_stage_memory_demand += sum(profile_memory[start_layer_id:end_layer_id]) * mem_coef
            else:
                data_load_balancer = DataLoadBalancer(self.profile_data, self.model_config)
                hetero_bs = data_load_balancer.partition_data(device_types, strategy, gbs // batches)
                for h_mbs in hetero_bs:
                    comb_h_mbs = [2 ** i for i in range(int(math.log2(h_mbs)) if h_mbs != 0 else 0, -1, -1) if h_mbs & 2 ** i]
                    for slice_h_mbs in comb_h_mbs:
                        profile_memory = self.profile_data[f'DeviceType.{device_types[0]}'][f'tp{tp_deg}_bs{slice_h_mbs}']['memory']
                        cur_stage_memory_demand += sum(profile_memory[start_layer_id:end_layer_id]) * mem_coef
            stage_memory.append(cur_stage_memory_demand)

        return stage_memory

    def _detect_out_of_memory(self, stage_memory_demand: List[float], stage_memory_capacity: List[float])\
            -> Tuple[bool, List[float]]:
        memory_usage = [m_capa - m_demand for m_capa, m_demand in zip(stage_memory_capacity, stage_memory_demand)]
        if min(memory_usage) < 0:
            return True, memory_usage
        else:
            return False, memory_usage

    def _partition_layers_by_compute_performance(self, stage_compute_performance: List[float]) -> Tuple[List[int], List[float]]:
        compute_balancer = LayerComputeBalancer(len(stage_compute_performance), self.model_config.num_layers,
                                          stage_compute_performance.copy(), self.norm_layer_duration)
        layer_partition, sc_demand = compute_balancer.run()
        return layer_partition, sc_demand

    def _adj_compute_performance(self, sc_capa: List[float], sm_capa: List[float], sm_demand: List[float]) \
            -> Union[List[float], None]:
        # Examine the computing performance difference among stages with ample memory. Lower the computing capacity
        # of stages with insufficient memory to a level that satisfies the available memory. Increase computing
        # capacity to allow stages with available memory to handle tasks that couldn't be processed due to memory
        # shortage. Distribute tasks considering the performance and remaining meomry of each stage.
        adj_sc_capa = []
        available_compute_capacity = []
        extra_required_capacity = 0.
        for c_capa, m_capa, m_demand in zip(sc_capa, sm_capa, sm_demand):
            if m_capa > m_demand:
                adj_sc_capa.append(c_capa)
                extra_c_capa = (c_capa * m_capa / m_demand) - c_capa
                available_compute_capacity.append(extra_c_capa)
            else:
                available_compute_capacity.append(0)
                adj_c_capa = c_capa * (m_capa / m_demand) * 0.9
                adj_sc_capa.append(adj_c_capa)
                extra_required_capacity += (c_capa - adj_c_capa)

        if sum(available_compute_capacity) < extra_required_capacity:
            print('Even with the reallocation of layers, memory issues persist.')
            return None

        additional_alloc_sc_capa = [0. for _ in range(len(sc_capa))]
        while extra_required_capacity > 0.01:
            tmp_total = sum([c_capa if a_capa > 0.001 else 0 for a_capa, c_capa in zip(available_compute_capacity, sc_capa)])
            c_capa_ratio = [c_capa / tmp_total if a_capa > 0.001 else 0 for a_capa, c_capa in zip(available_compute_capacity, sc_capa)]

            for (stage_id, c_ratio), a_capa in zip(enumerate(c_capa_ratio), available_compute_capacity):
                alloc_capa = a_capa if extra_required_capacity * c_ratio > a_capa else extra_required_capacity * c_ratio
                additional_alloc_sc_capa[stage_id] += alloc_capa
                available_compute_capacity[stage_id] -= alloc_capa
                extra_required_capacity -= alloc_capa

        adj_sc_capa = [alloc_capa + adj_capa for alloc_capa, adj_capa in zip(additional_alloc_sc_capa,  adj_sc_capa)]
        return adj_sc_capa

    def _device_types_by_node_sequence(self, node_sequence: List[DeviceType]) -> List[str]:
        device_names = [self.gpu_cluster.nodes[node_id].device_type.name for node_id in self.gpu_cluster.nodes.keys()]
        node_sequence = [device_type.name for device_type in node_sequence]
        num_devices = self.gpu_cluster.nodes[0].num_devices

        device_type_count = Counter(device_names)
        sorted_device_types = []
        for key in node_sequence:
            sorted_device_types.extend([key] * device_type_count[key] * num_devices)

        return sorted_device_types

    def partition_layer(self, plan: 'InterStagePlan', strategies: List[Tuple[int, int]],
                        stage_compute_performance: List[float], stage_memory_capacity: List[int],
                        max_partition_attempts: int = 3) -> Tuple[Union[List, None], int, Union[List, None]]:
        device_types = self._device_types_by_node_sequence(plan.node_sequence)

        cur_partition_attempt = 1
        while cur_partition_attempt <= max_partition_attempts:
            layer_partition, stage_compute_demand = self._partition_layers_by_compute_performance(stage_compute_performance)
            stage_memory_demand = self._get_stage_memory_demand(layer_partition, strategies, plan.device_groups,
                                                                device_types, plan.gbs, plan.batches)
            memory_exceeded, memory_state = self._detect_out_of_memory(stage_memory_demand, stage_memory_capacity)
            print(f'layer_partition: {layer_partition}')
            print(f'stage_memory_demand: {stage_memory_demand}, memory_state: {memory_state}')
            if not memory_exceeded:
                return layer_partition, cur_partition_attempt, memory_state

            stage_compute_performance = self._adj_compute_performance(stage_compute_performance, stage_memory_capacity,
                                                                      stage_memory_demand)
            if not stage_compute_performance:
                return None, -1, None

            cur_partition_attempt += 1
            print(f'adj_stage_compute_performance({cur_partition_attempt}): {stage_compute_performance}')
        return None, -1, None


class DataLoadBalancer:
    def __init__(self, profile_data: Dict, model_config):
        self.profile_data = profile_data
        self.model_config = model_config

    def _get_execution_time(self, device_type: str, key: str) -> float:
        return sum(self.profile_data[f'DeviceType.{device_type}'][key]['time']['layer-computes'])

    def partition_data(self, device_types: List[str], intra_strategy: Tuple[int, int], bs: int) -> List[int]:
        dp_deg, tp_deg = intra_strategy

        inner_stage_performance = []
        group_size = len(device_types) // dp_deg
        for i in range(dp_deg):
            dp_group = device_types[i * group_size: (i + 1) * group_size]
            profile_cost = self._get_execution_time(dp_group[0], f'tp{tp_deg}_bs1')
            inner_stage_performance.append(1. / profile_cost)

        inner_total_performance = sum(inner_stage_performance)
        inner_stage_compute_performance = [s_performance / inner_total_performance
                                           for s_performance in inner_stage_performance]

        hetero_bs = [int(bs * c_performance) for c_performance in inner_stage_compute_performance]
        remainder = bs - sum(hetero_bs)

        remainder_ratio = [(bs * c_performance) - int(bs * c_performance)
                           for c_performance in inner_stage_compute_performance]

        sorted_indices = sorted(range(len(remainder_ratio)), key=lambda i: remainder_ratio[i], reverse=True)
        for i in range(remainder):
            hetero_bs[sorted_indices[i]] += 1

        return hetero_bs


class LayerComputeBalancer:
    def __init__(self, num_stage: int, num_layer: int, sc_capa: List[float], lc_demand: List[float], hallucination: int = 7):
        self.num_stage = num_stage
        self.num_layer = num_layer * hallucination
        self.sc_capa_bak = sc_capa.copy()
        self.sc_capa = sc_capa
        self.lc_demand = lc_demand
        self.expand_lc_demand = []
        for c_demand in lc_demand:
            tmp_demand = c_demand / hallucination
            for i in range(hallucination):
                self.expand_lc_demand.append(tmp_demand)

        self.hallucination = hallucination

    def run(self):
        self._init_allocation()
        self._alloc_first_pass_forward()
        self._alloc_first_pass_backward()
        self._alloc_unassigned_first_pass()
        self._alloc_real_value()
        self._alloc_first_pass_adjust()

        partition = self._get_partition()
        sc_demand = self._get_stage_compute_demand(partition)
        return partition, sc_demand

    def _init_allocation(self):
        self.lid_alloc_stage = dict()
        for i in range(self.num_stage):
            self.lid_alloc_stage[i] = []

        self.un_assigned_layer = []

    def _alloc_first_pass_forward(self, k: int = 0):
        for stage_id in range(self.num_stage - 1):
            for layer_id in range(k, self.num_layer - 1 - (1 * self.hallucination)):
                if self.sc_capa[stage_id] > self.expand_lc_demand[layer_id]:
                    self.sc_capa[stage_id] -= self.expand_lc_demand[layer_id]
                    self.lid_alloc_stage[stage_id].append(layer_id)
                    k = layer_id + 1
                else:
                    k = layer_id
                    self.un_assigned_layer.append(layer_id)
                    k = layer_id + 1
                    break
        for layer_id in range(k, self.num_layer):
            self.un_assigned_layer.append(layer_id)

        self.un_assigned_layer = list(set(sorted(self.un_assigned_layer)))

    def _alloc_first_pass_backward(self):
        last_stage = self.num_stage - 1
        _un_assigned_layer = sorted(self.un_assigned_layer.copy(), reverse=True).copy()
        for layer_id in _un_assigned_layer:
            if len(self.lid_alloc_stage[last_stage]) < self.hallucination:
                self.sc_capa[last_stage] -= self.expand_lc_demand[layer_id]
                self.lid_alloc_stage[last_stage].append(layer_id)
                self.un_assigned_layer.remove(layer_id)
                continue

            if (layer_id + 1) != min(self.lid_alloc_stage[last_stage]):
                continue

            if self.sc_capa[last_stage] > self.expand_lc_demand[layer_id]:
                self.sc_capa[last_stage] -= self.expand_lc_demand[layer_id]
                self.lid_alloc_stage[last_stage].append(layer_id)
                self.un_assigned_layer.remove(layer_id)

    def _alloc_unassigned_first_pass(self):
        def get_proper_stage(d_layer_group: Dict, q: int) -> int:
            min_stage_id, max_stage_id = min(list(d_layer_group.keys())), max(list(d_layer_group.keys()))
            min_value, max_value = float('inf'), float('-inf')

            for key in d_layer_group.keys():
                inner_group = d_layer_group[key]
                if len(inner_group) == 0:
                    continue

                cur_min_value, cur_max_value = min(list(inner_group)), max(list(inner_group))
                if q > cur_max_value and cur_max_value > max_value:
                    min_stage_id = key
                    max_value = cur_max_value
                if q < cur_min_value and cur_min_value < min_value:
                    max_stage_id = key
                    min_value = cur_min_value

            stage_id, max_s_capa = None, float('-inf')
            for s_id in range(min_stage_id, max_stage_id + 1):
                if self.sc_capa[s_id] > max_s_capa:
                    max_s_capa = self.sc_capa[s_id]
                    stage_id = s_id

            return stage_id

        _un_assigned_layer = sorted(self.un_assigned_layer.copy())
        for layer_id in _un_assigned_layer:
            c_layer = self.expand_lc_demand[layer_id]
            stage_id = get_proper_stage(self.lid_alloc_stage, layer_id)

            self.sc_capa[stage_id] -= c_layer
            self.lid_alloc_stage[stage_id].append(layer_id)
            self.un_assigned_layer.remove(layer_id)

        for key in self.lid_alloc_stage.keys():
            self.lid_alloc_stage[key] = sorted(self.lid_alloc_stage[key])


    def _alloc_real_value(self):
        lid_alloc_stage = dict()
        for stage_id in range(self.num_stage):
            real_value_group = [int(layer_id / self.hallucination) for layer_id in self.lid_alloc_stage[stage_id]]
            filtered_real_value = [layer_id for layer_id in real_value_group if
                                   real_value_group.count(layer_id) > (self.hallucination / 2)]
            lid_alloc_stage[stage_id] = sorted(list(set(filtered_real_value)))
        self.lid_alloc_stage = lid_alloc_stage
        self.num_layer /= self.hallucination

        sc_capa = []
        for stage_id in range(len(lid_alloc_stage.keys())):
            if len(lid_alloc_stage[stage_id]):
                s_layer_id, e_layer_id = lid_alloc_stage[stage_id][0], lid_alloc_stage[stage_id][-1]
                sc_capa.append(self.sc_capa_bak[stage_id] - sum(self.lc_demand[s_layer_id:e_layer_id+1]))
            else:
                sc_capa.append(self.sc_capa_bak[stage_id])

        self.sc_capa = sc_capa

    def _alloc_first_pass_adjust(self):
        def get_near_max(idx: int, sc_capa: List) -> int:
            max_idx, min_value = None, float('inf')
            if (idx - 1) >= 0 and sc_capa[idx - 1] < min_value:
                max_idx = idx - 1
                min_value = sc_capa[idx - 1]
            if (idx + 1) < len(sc_capa) and sc_capa[idx + 1] < min_value:
                max_idx = idx + 1
                min_value = sc_capa[idx + 1]
            if max_idx is None or len(self.lid_alloc_stage[max_idx]) == 1:
                max_idx = None
            return max_idx

        _opt_sc_capa = self.sc_capa.copy()
        _opt_alloc_stage = copy.deepcopy(self.lid_alloc_stage)

        num_search = 0
        while True:
            num_search += 1
            _sc_capa_d = [(i, _opt_sc_capa[i]) for i in range(len(_opt_sc_capa))]
            sorted_sc_capa_d = sorted(_sc_capa_d, key=lambda kv: kv[1], reverse=True)
            stage_id, c_capa = sorted_sc_capa_d[0]

            near_idx = get_near_max(stage_id, _opt_sc_capa)
            if (near_idx is not None) and len(_opt_alloc_stage[near_idx]):
                if stage_id > near_idx:
                    _layer = _opt_alloc_stage[near_idx].pop(-1)
                    _opt_alloc_stage[stage_id].append(_layer)
                    _opt_alloc_stage[stage_id] = sorted(_opt_alloc_stage[stage_id])

                    c_demand = self.lc_demand[_layer]
                    _opt_sc_capa[stage_id] -= c_demand
                    _opt_sc_capa[near_idx] += c_demand
                else:
                    _layer = _opt_alloc_stage[near_idx].pop(0)
                    _opt_alloc_stage[stage_id].append(_layer)
                    _opt_alloc_stage[stage_id] = sorted(_opt_alloc_stage[stage_id])

                    c_demand = self.lc_demand[_layer]
                    _opt_sc_capa[stage_id] -= c_demand
                    _opt_sc_capa[near_idx] += c_demand

            if max(_opt_sc_capa) > max(self.sc_capa) or num_search > 3:
                break

            self.lid_alloc_stage = copy.deepcopy(_opt_alloc_stage)
            self.sc_capa = _opt_sc_capa.copy()

    def _get_partition(self) -> List[int]:
        partition = [0]
        for key in self.lid_alloc_stage.keys():
            stage_layers = self.lid_alloc_stage[key]
            partition.append(partition[key] + len(stage_layers))

        return partition

    def _get_stage_compute_demand(self, partition: List[int]) -> List[float]:
        stage_compute = []
        for i in range(len(partition) - 1):
            s_pos, e_pos = partition[i], partition[i + 1]
            stage_compute.append(sum(self.lc_demand[s_pos: e_pos]))

        return stage_compute

