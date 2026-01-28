import torch
import numpy as np
from multiprocessing import Pool
from functools import partial
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def compute_distance_matrix(points):
    """计算点集的欧几里得距离矩阵（NumPy版本）"""
    return np.sqrt(((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2).sum(axis=2))


def _solve_single(args):
    """单TSP求解函数（直接从seed计算距离矩阵）"""
    i, seed = args  # seed: (problem_size, 2)
    distance_matrix = compute_distance_matrix(seed)
    n_nodes = len(distance_matrix)

    # OR-Tools求解部分
    manager = pywrapcp.RoutingIndexManager(n_nodes, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 1000)  # 放大1000倍转为整数

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = 5

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))  # 添加终点
        route = route[:-1]  # 移除闭环重复节点

        if max(route) >= len(seed):
            raise ValueError(f"路径索引{max(route)}超出seed长度{len(seed)}")

        return i, (route, solution.ObjectiveValue() / 1000.0)  # 成本缩小回原比例
    raise RuntimeError(f"No solution for instance {i}")


def upper_solve_parallel(seeds, num_processes=4):
    """
    多进程批量求解TSP（直接输入seeds）
    Args:
        seeds: (batch_size, problem_size, 2) 的PyTorch张量
        num_processes: 并行进程数
    Returns:
        routes: (batch_size, problem_size) 的访问顺序
        costs: (batch_size,) 的路径成本
    """
    batch_size = seeds.shape[0]
    seeds_np = seeds.cpu().numpy()  # 转换为NumPy便于多进程共享

    # 并行求解
    with Pool(processes=num_processes) as pool:
        results = pool.map(_solve_single, enumerate(seeds_np))

    # 按原始顺序重组结果
    results.sort(key=lambda x: x[0])
    routes = [r[1][0] for r in results]
    costs = [r[1][1] for r in results]

    # 转换为PyTorch张量（保持原始设备）
    routes = torch.tensor(np.stack(routes), device=seeds.device)
    costs = torch.tensor(costs, device=seeds.device)

    return routes, costs