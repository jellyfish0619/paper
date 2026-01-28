import torch
import time
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from heatmap.inst import gen_inst, gen_pyg_data, trans_tsp  # 生成实例、图数据和转换TSP问题

DEVICE = 'cpu'  # 使用 GPU 设备


def create_data_model(coors, demand, capacity, num_vehicles=50):
    """创建数据模型（取消距离放大）"""
    dist_matrix = torch.cdist(coors, coors, p=2).numpy()  # 保持原始[0~1.414]范围

    return {
        'distance_matrix': dist_matrix.tolist(),  # 不转换为整数
        'demands': demand.tolist(),
        'vehicle_capacities': [capacity] * num_vehicles,
        'num_vehicles': num_vehicles,
        'depot': 0
    }


def solve_cvrp(n):
    coors, demand, capacity = gen_inst(n, 'cpu')
    data = create_data_model(coors, demand, capacity)

    """OR-Tools求解器（适配您的距离尺度）"""
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']),
        data['num_vehicles'],
        data['depot'])

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """直接返回原始距离（不放大）"""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 容量约束（保持您的设置）
    def demand_callback(from_index):
        return data['demands'][manager.IndexToNode(from_index)]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        data['vehicle_capacities'],
        True,
        'Capacity')

    # 搜索参数（保持快速求解）
    # search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # search_parameters.first_solution_strategy = (
    #     routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # search_parameters.time_limit.seconds = 50
    search_parameters = configure_search_parameters()
    start_time = time.time()
    solution = routing.SolveWithParameters(search_parameters)
    solve_time = time.time() - start_time

    return solution, routing, manager, solve_time, data, coors


def extract_routes(data, manager, routing, solution):
    """按照您的sum_cost逻辑提取路径信息"""
    routes = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            index = solution.Value(routing.NextVar(index))
        if len(route) > 0:  # 只保留非空路径
            routes.append(route)
    return routes


def calculate_route_costs(routes, distance_matrix):
    """完全实现您的sum_cost算法"""
    all_costs = []
    n_tsps_per_route = []

    for route in routes:
        route_costs = []
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            route_costs.append(distance_matrix[from_node][to_node])
        all_costs.extend(route_costs)
        n_tsps_per_route.append(len(route) - 1)

    # 您的sum_cost实现
    costs = torch.tensor(all_costs)
    ret = []
    start = 0
    for n in n_tsps_per_route:
        ret.append(costs[start: start + n].sum())
        start += n
    return torch.stack(ret), n_tsps_per_route


def configure_search_parameters():
    """效果优先的高级搜索参数配置"""
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.lns_time_limit.seconds = 30  # 启用大规模邻域搜索（LNS）
    search_parameters.solution_limit = 10000  # 限制解的数量，避免计算量过大
    search_parameters.time_limit.seconds = 300  # 增加时间限制，提高解的质量



    return search_parameters

