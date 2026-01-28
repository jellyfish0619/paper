
import torch

import os

from math import sqrt


def load_problem(name):
    from problems import TSP, LOCAL
    problem = {
        'local': LOCAL,
        'tsp': TSP,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem

def load_model(path, epoch=None, is_local=True): #加载模型
    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)



def load_vrp_data(file_path, device, max_customers=None):
    """从VRP文件加载数据，可选截取指定数量的客户点"""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    node_coords = []
    demands = []
    capacity = 0
    depot = 0  # 默认仓库节点为0

    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('CAPACITY'):
            capacity = float(line.split(':')[1].strip())
        elif line == 'NODE_COORD_SECTION':
            current_section = 'NODE_COORD'
        elif line == 'DEMAND_SECTION':
            current_section = 'DEMAND'
        elif line == 'DEPOT_SECTION':
            current_section = 'DEPOT'
        elif line == 'EOF':
            break
        else:
            if current_section == 'NODE_COORD':
                parts = line.split()
                node_id = int(parts[0]) - 1  # 转换为0-based索引
                x = float(parts[1])
                y = float(parts[2])
                node_coords.append([x, y])
            elif current_section == 'DEMAND':
                parts = line.split()
                node_id = int(parts[0]) - 1  # 转换为0-based索引
                demand = int(parts[1])  # 直接转为int
                demands.append(demand)
            elif current_section == 'DEPOT':
                parts = line.split()
                if parts[0] != '-1':
                    depot = int(parts[0]) - 1  # 转换为0-based索引

    # 确保仓库节点是第一个节点
    if depot != 0:
        node_coords[0], node_coords[depot] = node_coords[depot], node_coords[0]
        demands[0], demands[depot] = demands[depot], demands[0]
        depot = 0

    # 如果指定了max_customers，截取前max_customers+1个点（仓库+客户）
    if max_customers is not None:
        total_nodes = len(node_coords)
        if total_nodes > max_customers + 1:
            node_coords = node_coords[:max_customers + 1]
            demands = demands[:max_customers + 1]

    # 转换为张量
    coors = torch.tensor(node_coords, dtype=torch.float32, device=device)
    demand = torch.tensor(demands, dtype=torch.int, device=device)  # 改为int类型

    return coors, demand, capacity
