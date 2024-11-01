# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
from typing import List
from itertools import chain
from search_space.utils import permutations


def permute(s, max_permute_len):
    def find_num_min(m, groups):
        for i, e in enumerate(groups):
            if m != e:
                return i + 1
        return len(groups)

    # grouping
    groups = [(e,) for e in s]
    # Key idea 2: limit the permutation length
    curr_permute_len = len(groups)
    num_reduce = curr_permute_len - max_permute_len
    while num_reduce > 0:
        min_group_size = sum(groups[0])
        num_min_groups = find_num_min(groups[0], groups)
        # Example:
        #  - device groups: (1), (1), (1), (1), (1), (1), (2)
        #  - max_permute_len: 6
        #  - generated: (1,1), (1,1), (1,1), (2)
        if num_min_groups // 2 > num_reduce:
            num_reduce = num_min_groups // 2

        # Merge the two smallest groups
        merged_groups = []
        for i in range(0, len(groups), 2):
            curr_reduce_num = i // 2
            if num_reduce <= curr_reduce_num:
                # End of merge
                merged_groups.extend(groups[i:])
                break
            if i + 1 >= len(groups):
                merged_groups.append(groups[i])
            else:
                if sum(groups[i]) == min_group_size and sum(groups[i]) == sum(groups[i + 1]):
                    merged_group = tuple(groups[i] + groups[i + 1])
                    merged_groups.append(merged_group)
                else:
                    merged_groups.append(groups[i])
                    merged_groups.append(groups[i + 1])

        groups = merged_groups
        if num_reduce == len(groups) - max_permute_len:
            # we can't reduce anymore
            break
        num_reduce = len(groups) - max_permute_len

    # print("Merged groups:", groups)
    perms = permutations(groups)
    return perms


def gen_dgroups_recursive(num_stages: int, num_gpus: int, group_shapes: List):
    def f(current_sum, stage_idx, curr_sol, prev_shape_idx):
        # filtering
        if group_shapes[-1] * (num_stages - stage_idx) < num_gpus - current_sum:
            # max gpu < total gpu
            return
        if group_shapes[0] * (num_stages - stage_idx) > num_gpus - current_sum:
            # min gpu > total gpu
            return

        if stage_idx >= num_stages:
            if len(curr_sol) == num_stages and current_sum == num_gpus:
                yield curr_sol
            return

        for i in range(max(0, prev_shape_idx), len(group_shapes)):
            possible_gpu_num = group_shapes[i]
            if possible_gpu_num + current_sum > num_gpus:
                break
            my_sol = curr_sol + [possible_gpu_num]
            yield from f(current_sum + possible_gpu_num, stage_idx + 1, my_sol, i)

    for idx, possible_gpu_num in enumerate(group_shapes):
        yield from f(possible_gpu_num, 1, [possible_gpu_num], idx)


def gen_device_group_shapes(num_gpus: int) -> List[int]:
    group_shapes = []
    i = 0
    while 2 ** i <= num_gpus:
        group_shapes.append(2 ** i)
        i += 1
    return group_shapes


def gen_dgroups_for_stages_with_variance(num_stages: int, num_gpus: int, group_shapes: List[int], variance: float,
                                         max_permute_len: int) -> List:
    # Key idea 1: Limit the size of device group
    min_group_stage = max(num_gpus // num_stages, num_stages // num_gpus)
    min_group_stage *= variance
    group_shapes = [s for s in group_shapes if s >= min_group_stage]

    device_groups = []
    for s in gen_dgroups_recursive(num_stages, num_gpus, group_shapes):
        perm_s = permute(s, max_permute_len)
        for perm in perm_s:
            perm_ss = list(chain(*perm))
            device_groups.append(perm_ss)

    return device_groups
