# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
import argparse
from copy import copy
from typing import List, Tuple
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import parse_args
from data_loader import ProfileDataLoader
from gpu_cluster import GPUCluster
from model.cost_estimator import HomoCostEstimator
from model.cost_validation import EstimateCostValidator
from model.activation_parameter import GPTActivationAndParam
from search_space.plan import UniformPlanGenerator
from utils import ModelConfig
from search_space.plan import UniformPlan


def cost_homo_cluster(args: argparse.Namespace, gpu_cluster: GPUCluster, cost_estimator: HomoCostEstimator) -> List[Tuple[UniformPlan, float]]:
    estimate_costs = []
    for plan in UniformPlanGenerator(num_devices=gpu_cluster.get_total_num_devices(),
                                     max_tp=args.max_profiled_tp_degree, max_gbs=args.gbs):
        if plan.gbs != args.gbs:
            continue

        try:
            time_cost, stage_memory_cost, OOM = cost_estimator.get_cost(plan, device_types[0])
            estimate_costs.append((copy(plan), time_cost))

            print(f'\n{plan}')
            print(f"time: {time_cost}, memory(stage): {stage_memory_cost}")
        except KeyError as e:
            print(f'KeyError: {e}')

    return estimate_costs


if __name__ == "__main__":
    args = parse_args()
    gpu_cluster = GPUCluster(hostfile_path=args.hostfile_path, clusterfile_path=args.clusterfile_path)

    assert 10 <= gpu_cluster.get_inter_bandwidth(0) <= 500, \
        "intra-bandwidth for NVLink should exist within a range 10GB/s to 500GB/s"
    assert 1 <= gpu_cluster.get_intra_bandwidth(0) <= 50, \
        "inter-bandwidth should exist within a range 1GB/s to 50GB/s"

    data_loader = ProfileDataLoader(profile_dir=args.profile_data_path, runtime_cost_dir=args.evaluation_data_path)
    profile_data, device_types = data_loader.load_profile_data_all()
    if len(profile_data.keys()) > 0:
        print('\nProfiled data has been loaded.')

    assert len(profile_data.keys()) > 0, 'There is no profiled data at the specified path.'

    model_config = ModelConfig(model_name=args.model_name,
                               num_layers=args.num_layers,
                               sequence_length=args.sequence_length,
                               vocab_size=args.vocab_size,
                               hidden_size=args.hidden_size,
                               attention_head_size=args.attention_head_size)

    model_volume = GPTActivationAndParam(model_config, profile_data['model']['parameters'])
    cost_estimator = HomoCostEstimator(profile_data, model_config, model_volume, gpu_cluster)

    estimate_costs = cost_homo_cluster(args, gpu_cluster, cost_estimator)
    sorted_result = sorted(estimate_costs, key=lambda kv: kv[1])
    print('rank, cost, plan')
    for idx, result in enumerate(sorted_result):
        print(f'{idx + 1}, {result[1]}, {result[0]}')