import torch
import numpy as np
# from utils.functions import load_problem
from utils.insertion import random_insertion_parallel
from heatmap.inst import sum_cost
from upper_solver import or_tools
# problem = load_problem('tsp')
# get_cost_func = lambda input, pi: problem.get_costs(input, pi, return_local=True)

# @torch.no_grad()
# def eval(tsp_insts, n_tsps_per_route, opts):
#     opts.eval_batch_size = (tsp_insts.size(0))
#     p_size = tsp_insts.size(1)
#     seeds = tsp_insts
#     order = torch.arange(p_size)
#     pi_all = random_insertion_parallel(seeds, order)
#     pi_all = torch.tensor(pi_all.astype(np.int64), device=seeds.device).reshape(-1, p_size)
#     seeds = seeds.gather(1, pi_all.unsqueeze(-1).expand_as(seeds))
#     tours, costs_revised = reconnect(
#                                 get_cost_func=get_cost_func,
#                                 batch=seeds,
#                                 opts=opts,
#                                 revisers=opts.revisers,
#                                 )
#     assert costs_revised.size(0) == seeds.size(0)
#     costs_revised = sum_cost(costs_revised, n_tsps_per_route)
#     return costs_revised


@torch.no_grad()
def eval(tsp_insts, n_tsps_per_route, opts):
    opts.eval_batch_size = (tsp_insts.size(0))
    p_size = tsp_insts.size(1)
    seeds = tsp_insts
    order = torch.arange(p_size)
    tsp_insts_dim = tsp_insts.shape

    pi_all = random_insertion_parallel(seeds, order)
    pi_all_dim = pi_all.shape

    pi_all = torch.tensor(pi_all.astype(np.int64), device=seeds.device).reshape(-1, p_size)
    seeds = seeds.gather(1, pi_all.unsqueeze(-1).expand_as(seeds))

    tours, costs_revised = or_tools.upper_solve_parallel(seeds)
    if isinstance(costs_revised, (int, float)):
        costs_revised = torch.tensor(costs_revised, device=seeds.device)
    if costs_revised.dim() == 0:
        costs_revised = costs_revised.unsqueeze(0)  # 将标量张量转换为一维张量
    assert costs_revised.size(0) == seeds.size(0)
    costs_revised = sum_cost(costs_revised, n_tsps_per_route)
    return costs_revised

