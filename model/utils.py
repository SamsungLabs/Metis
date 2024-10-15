# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
from typing import List


def partition_layers_by_stage(total_layers: int, num_stages: int) -> List[int]:
    """
    Function to partition layers by stage.
    Parameters:
        total_layers (int): Total number of layers.
        num_stages (int): Number of stages to partition.
    Returns:
        list: A list containing the number of layers partitioned by stage.
    Example:
        total_layers = 10
        num_stages = 4
        partition_layers_by_stage(total_layers, num_stages)
        # Output: [3, 2, 2, 3]
    Note:
        - The input layer and output layer are included in the first and last stages, respectively.
        - The number of layers partitioned is evenly distributed across each stage.
    """
    uniform_partition_layers = (total_layers - 2) // num_stages
    remaining_layers = (total_layers - 2) % num_stages
    stage_layer_counts = [uniform_partition_layers] * num_stages
    for i in range(1, remaining_layers+1):
        stage_layer_counts[i] += 1

    stage_layer_counts[0] += 1
    stage_layer_counts[-1] += 1

    return stage_layer_counts