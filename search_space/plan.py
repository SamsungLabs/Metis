# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
import copy
from dataclasses import dataclass
from itertools import permutations
from typing import List, Tuple, Union

from search_space.device_group import gen_device_group_shapes, gen_dgroups_for_stages_with_variance
from model.load_balancer import LayerLoadBalancer
from model.device_group import StagePerformance
from utils import DeviceType

@dataclass
class UniformPlan:
    dp: int
    pp: int
    tp: int
    mbs: int
    gbs: int


@dataclass
class InterStagePlan:
    ns_idx:int
    node_sequence: List[DeviceType]
    dg_idx: int
    device_groups: List[int]
    num_stage: int
    batches: int
    gbs: int


@dataclass
class IntraStagePlan:
    strategies: List[Tuple[int, int]]
    memory_state: List[float]
    layer_partition: List[int]
    num_repartition: int


class UniformPlanGenerator:
    def __init__(self, num_devices: int, max_tp: int, max_gbs: int):
        self.num_devices = num_devices
        self.max_tp = max_tp
        self.max_gbs = max_gbs
        self.curr = UniformPlan(dp=num_devices, pp=1, tp=1, gbs=num_devices, mbs=0)

    def _find_next_mbs(self) -> int:
        mbs = self.curr.mbs + 1
        while self.curr.gbs % mbs > 0 and mbs <= self.curr.gbs:
            mbs += 1
        return mbs

    def _find_next_gbs(self) -> int:
        gbs = self.curr.gbs + 1
        while self.max_gbs % gbs > 0 and gbs <= self.max_gbs:
            gbs += 1
        return gbs

    def _find_next_dp_pp_tp(self) -> Union[UniformPlan, None]:
        plan = self.curr
        while True:
            if plan.tp == self.max_tp and plan.pp == self.num_devices:
                # Invalid
                return None
            elif plan.tp == self.max_tp:
                plan.pp += 1
                plan.dp = self.num_devices // plan.pp
                plan.tp = self.num_devices // plan.dp // plan.pp
            else:
                plan.tp += 1
                plan.dp = self.num_devices // plan.tp // plan.pp

            # valid plan for megatron-lm
            if plan.dp * plan.pp * plan.tp == self.num_devices:
                break
        return plan

    def __iter__(self):
        return self

    def __next__(self) -> UniformPlan:
        self.curr.mbs = self._find_next_mbs()

        if self.curr.mbs * self.curr.dp > self.curr.gbs:
            self.curr.mbs = 1
            self.curr.gbs = self._find_next_gbs()

        if self.curr.gbs > self.max_gbs:
            self.curr.mbs = 1

            self.curr = self._find_next_dp_pp_tp()
            if self.curr is None:
                raise StopIteration

            self.curr.gbs = self.curr.dp

        return self.curr


class InterStagePlanGenerator:
    def __init__(self, device_types: set, num_devices: int, gbs: int, num_layers: int, variance: float = 0.5,
                 max_permute_len: int = 4):

        self.node_sequences = list(permutations(device_types))
        self.num_devices = num_devices
        self.gbs = gbs
        self.num_layers = num_layers
        self.variance = variance
        self.max_permute_len = max_permute_len
        self.group_shapes = gen_device_group_shapes(num_devices)
        self.device_groups = gen_dgroups_for_stages_with_variance(num_stages=1,
                                                                  num_gpus=self.num_devices,
                                                                  group_shapes=self.group_shapes,
                                                                  variance=variance,
                                                                  max_permute_len=max_permute_len)

        self.curr = InterStagePlan(ns_idx=0, node_sequence=list(self.node_sequences[0]), dg_idx=0,
                                   device_groups=self.device_groups[0], num_stage=1, batches=gbs+1, gbs=gbs)

    def _find_next_batches(self) -> int:
        batches = self.curr.batches -1
        while batches >= 1 and self.curr.gbs % batches > 0:
            batches -= 1
        return batches

    def _find_next_dg(self) -> int:
        dg_idx = self.curr.dg_idx + 1
        return dg_idx

    def _find_next_stage_device_groups(self) -> int:
        num_stage = self.curr.num_stage + 1

        while True:
            self.device_groups = gen_dgroups_for_stages_with_variance(num_stages=num_stage,
                                                                  num_gpus=self.num_devices,
                                                                  group_shapes=self.group_shapes,
                                                                  variance=self.variance,
                                                                  max_permute_len=self.max_permute_len)
            if self.device_groups or num_stage > min(self.num_devices, self.num_layers):
                break
            num_stage += 1
        return num_stage

    def _find_next_node_sequence(self) -> int:
        ns_idx = self.curr.ns_idx + 1
        self.curr.num_stage = 1
        self._find_next_stage_device_groups()
        return ns_idx

    def __iter__(self):
        return self

    def __next__(self) -> InterStagePlan:
        self.curr.batches = self._find_next_batches()

        if self.curr.batches == 0:
            self.curr.dg_idx = self._find_next_dg()
            self.curr.batches = self.gbs

        if self.curr.dg_idx >= len(self.device_groups):
            self.curr.num_stage = self._find_next_stage_device_groups()
            self.curr.batches = self.gbs
            self.curr.dg_idx = 0

        if self.curr.num_stage > min(self.num_devices, self.num_layers):
            self.curr.ns_idx = self._find_next_node_sequence()
            self.curr.batches = self.gbs
            self.curr.dg_idx = 0

        if self.curr.ns_idx >= len(self.node_sequences):
            raise StopIteration

        self.curr.device_groups = self.device_groups[self.curr.dg_idx]
        self.curr.node_sequence = self.node_sequences[self.curr.ns_idx]
        return self.curr


class IntraStagePlanGenerator:
    def __init__(self, inter_stage_plan: InterStagePlan, stage_performance: StagePerformance,
                 layer_load_balancer: LayerLoadBalancer, max_tp_degree: int, max_bs: int):
        self.inter_stage_plan = inter_stage_plan
        self.device_groups = inter_stage_plan.device_groups
        self.gbs = inter_stage_plan.gbs
        self.batches = inter_stage_plan.batches
        self.stage_performance = stage_performance
        self.layer_load_balancer = layer_load_balancer
        self.max_tp_degree = max_tp_degree
        self.max_bs = max_bs

        self.curr = IntraStagePlan(strategies=[], memory_state=[], layer_partition=[], num_repartition=0)

    @property
    def has_next(self) -> bool:
        if self.curr.num_repartition == 1:
            return False

        while True:
            if not self.curr.strategies:
                self.curr.strategies = self._initial_strategies()
            else:
                self.curr.strategies = self._next_strategy(copy.deepcopy(self.curr.strategies))

            if not self.curr.strategies:
                return False

            if self._is_valid_strategies(self.curr.strategies):
                print(f'valid_strategies: {self.curr.strategies}')
                stage_memory_capacity = self.stage_performance.get_device_group_memory_capacity()
                intra_stage_compute_performance = self.stage_performance.get_intra_stage_compute_performance(
                    self.curr.strategies, self.gbs, self.batches)
                print(f'stage_memory_capacity: {stage_memory_capacity}')
                print(f'stage_compute_performance: {intra_stage_compute_performance}')

                layer_partition, num_repartition, memory_state = (
                    self.layer_load_balancer.partition_layer(self.inter_stage_plan, self.curr.strategies,
                                                             intra_stage_compute_performance, stage_memory_capacity))

                print(f'layer_partition: {layer_partition}')
                if layer_partition:
                    self.curr.layer_partition = layer_partition
                    self.curr.memory_state = memory_state
                    self.curr.num_repartition = num_repartition
                    return True
                else:
                    self.curr.memory_state = memory_state
                    continue

    def next(self) -> IntraStagePlan:
        return self.curr

    def _initial_strategies(self) -> List[Tuple[int, int]]:
        strategies = []
        for num_devices in self.device_groups:
            strategies.append((num_devices, 1))

        return strategies

    def _is_valid_strategies(self, strategies: List[Tuple[int, int]]) -> bool:
        for dp_deg, tp_deg in strategies:
            mbs = self.gbs // dp_deg // self.batches
            if mbs == 0 or mbs > self.max_bs:
                # log for debugging
                print(f'invalid_strategy: dp_deg({dp_deg}), batches({self.batches}), mbs(0)')
                return False
            if tp_deg > self.max_tp_degree:
                # log for debugging
                print(f'invalid_strategy: tp_deg({tp_deg})')
                return False
        return True

    def _next_strategy(self, strategies: List[Tuple[int, int]]) -> Union[List[Tuple[int, int]], None]:
        if self.curr.memory_state:
            memory_state = self.curr.memory_state
        else:
            memory_state = [1 / dp_deg for (dp_deg, tp_deg) in self.curr.strategies]

        memory_state_dict = {}
        for stage_id, memory_state in enumerate(memory_state):
            memory_state_dict[stage_id] = memory_state

        sorted_stage_id = sorted(memory_state_dict, key=lambda x: memory_state_dict[x])
        for stage_id in sorted_stage_id:
            dp_deg, tp_deg = strategies[stage_id]
            if dp_deg != 1:
                strategies[stage_id] = (dp_deg // 2, tp_deg * 2)
                return strategies

        return None
