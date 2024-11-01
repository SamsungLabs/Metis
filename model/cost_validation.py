# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
from data_loader import ProfileDataLoader
from search_space.plan import UniformPlan


class EstimateCostValidator:
    def __init__(self, data_loader: ProfileDataLoader, error_threshold: float):
        self.data_loader = data_loader
        self.error_threshold = error_threshold

        self.total, self.num_error = 0, 0
        self.costs = dict()

    def validate_cost_within_tolerance(self, plan: UniformPlan, estimate_cost: float) -> bool:
        runtime_cost = self.data_loader.load_eval_cost(plan)
        if not runtime_cost:
            return False

        str_plan = f'dp{plan.dp}_pp{plan.pp}_tp{plan.tp}_gbs{plan.gbs}_mbs{plan.mbs}'
        self.costs[str_plan] = dict()
        self.costs[str_plan]['estimate_cost'] = estimate_cost
        self.costs[str_plan]['runtime_cost'] = runtime_cost['interval-time']

        diff = abs(runtime_cost['interval-time'] - estimate_cost)
        diff_rate = diff / runtime_cost['interval-time']

        self.total += 1
        if diff_rate > self.error_threshold:
            self.num_error += 1
            return False

        return True
