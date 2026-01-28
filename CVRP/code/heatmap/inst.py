import random

import torch
from torch_geometric.data import Data

def gen_inst(n, device): # n代表客户的数量
    # 定义不同客户数量对应的车辆容量
    torch.manual_seed(619)
    CAPACITIES = {
        200: 40.,    # 当客户数量为 200 时，车辆容量为 40
        500: 100.,   # 当客户数量为 500 时，车辆容量为 100
        1000: 200.,  # 当客户数量为 1000 时，车辆容量为 200
        2000: 300.,  # 当客户数量为 2000 时，车辆容量为 300
        5000: 300.,  # 当客户数量为 5000 时，车辆容量为 300
        7000: 300.   # 当客户数量为 7000 时，车辆容量为 300
    }

    # 生成随机坐标，表示客户和仓库的位置
    # coors 是一个 (n+1, 2) 的张量，n+1 表示 n 个客户和 1 个仓库，2 表示二维坐标 (x, y)
    coors = torch.rand(size=(n+1, 2), device=device)

    # 生成每个客户的需求量，需求量为 1 到 9 之间的随机整数
    # demand 是一个长度为 n+1 的张量，表示每个客户的需求量
    demand = torch.randint(1, 10, (n+1,), device=device)

    # 将仓库的需求量设置为 0，因为仓库不需要配送
    demand[0] = 0

    # 根据客户数量 n 获取对应的车辆容量
    capacity = CAPACITIES[n]

    # 返回生成的坐标、需求量和车辆容量
    return coors, demand, capacity

def gen_distance_matrix(coordinates):
    # 计算坐标之间的欧几里得距离矩阵
    # coordinates[:, None] - coordinates 计算每对坐标之间的差值
    # torch.norm 计算差值的 L2 范数（欧几里得距离）
    distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
    return distances


def gen_cos_sim_matrix(shift_coors):
    # 计算坐标之间的余弦相似度矩阵
    # shift_coors 是相对于仓库的偏移坐标

    # 计算点积矩阵
    dot_products = torch.mm(shift_coors, shift_coors.t())

    # 计算每个坐标的模（magnitude），并扩展为矩阵形式
    magnitudes = torch.sqrt(torch.sum(shift_coors ** 2, dim=1)).unsqueeze(1)
    magnitude_matrix = torch.mm(magnitudes, magnitudes.t()) + 1e-10  # 加上小值避免除零错误

    # 计算余弦相似度矩阵
    cosine_similarity_matrix = dot_products / magnitude_matrix
    return cosine_similarity_matrix


def gen_pyg_data(coors, demand, capacity, k_sparse, cvrplib=False):
    # 将 VRP 数据转换为 PyTorch Geometric 的 Data 对象，适合图神经网络处理

    # 获取节点数量（客户数量 + 仓库）
    n_nodes = demand.size(0)

    # 将需求量归一化，除以车辆容量
    norm_demand = demand / capacity

    # 计算相对于仓库的偏移坐标
    shift_coors = coors - coors[0]

    # 提取偏移坐标的 x 和 y 分量
    _x, _y = shift_coors[:, 0], shift_coors[:, 1]

    # 计算极坐标中的半径 r 和角度 theta
    r = torch.sqrt(_x**2 + _y**2)
    theta = torch.atan2(_y, _x)

    # 构建节点特征矩阵 x，包含归一化需求量、半径和角度
    x = torch.stack((norm_demand, r, theta)).transpose(1, 0)

    # 计算余弦相似度矩阵
    cos_mat = gen_cos_sim_matrix(shift_coors)

    if cvrplib:
        # 如果使用 CVRPLIB 数据集的处理方式
        # 对余弦相似度矩阵进行归一化
        cos_mat = (cos_mat + cos_mat.min()) / cos_mat.max()

        # 计算欧几里得距离矩阵
        euc_mat = gen_distance_matrix(coors)

        # 计算欧几里得亲和度矩阵（1 - 距离矩阵）
        euc_aff = 1 - euc_mat

        # 结合余弦相似度和欧几里得亲和度，选择 top-k 最相似的边
        topk_values, topk_indices = torch.topk(cos_mat + euc_aff, k=k_sparse, dim=1, largest=True)

        # 构建边索引 edge_index
        edge_index = torch.stack([
            torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device), repeats=k_sparse),
            torch.flatten(topk_indices)
        ])

        # 构建边属性 edge_attr1 和 edge_attr2
        edge_attr1 = euc_aff[edge_index[0], edge_index[1]].reshape(k_sparse * n_nodes, 1)
        edge_attr2 = cos_mat[edge_index[0], edge_index[1]].reshape(k_sparse * n_nodes, 1)
    else:
        # 如果不使用 CVRPLIB 数据集的处理方式
        # 直接使用余弦相似度矩阵选择 top-k 最相似的边
        topk_values, topk_indices = torch.topk(cos_mat, k=k_sparse, dim=1, largest=True)

        # 构建边索引 edge_index
        edge_index = torch.stack([
            torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device), repeats=k_sparse),
            torch.flatten(topk_indices)
        ])

        # 构建边属性 edge_attr1 和 edge_attr2
        edge_attr1 = topk_values.reshape(-1, 1)
        edge_attr2 = cos_mat[edge_index[0], edge_index[1]].reshape(k_sparse * n_nodes, 1)

    # 将边属性拼接在一起
    edge_attr = torch.cat((edge_attr1, edge_attr2), dim=1)

    # 构建 PyTorch Geometric 的 Data 对象
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data


def trans_tsp(coors, routes, min_reviser_size=20):
    # 将车辆路径转换为 TSP（旅行商问题）实例

    tsp_pis = []  # 存储每个子路径
    n_tsps_per_route = []  # 存储每个路径的子路径数量

    for route in routes:
        start = 0
        sub_route_count = 0
        for idx, node in enumerate(route):
            if idx == 0:
                continue  # 跳过起始点（仓库）
            if node == 0:  # 如果当前节点是仓库
                if route[idx - 1] != 0:  # 如果前一个节点不是仓库
                    tsp_pis.append(route[start: idx])  # 将子路径添加到 tsp_pis
                    sub_route_count += 1
                start = idx  # 更新子路径的起始点
        n_tsps_per_route.append(sub_route_count)  # 记录当前路径的子路径数量

    # 计算所有子路径的最大长度，并确保不小于 min_reviser_size
    max_tsp_len = max([len(tsp_pis[i]) for i in range(len(tsp_pis))])
    max_tsp_len = max(min_reviser_size, max_tsp_len)

    # 对每个子路径进行填充，使其长度一致
    padded_tsp_pis = []
    for pi in tsp_pis:
        padded_pi = torch.nn.functional.pad(pi, (0, max_tsp_len - len(pi)), mode='constant', value=0)
        padded_tsp_pis.append(padded_pi)

    # 将填充后的子路径堆叠成一个张量
    padded_tsp_pis = torch.stack(padded_tsp_pis)

    # 根据填充后的子路径索引提取坐标
    tsp_insts = coors[padded_tsp_pis]

    # 确保生成的 TSP 实例形状正确
    assert tsp_insts.shape == (sum(n_tsps_per_route), max_tsp_len, 2)

    # 返回 TSP 实例和每个路径的子路径数量
    return tsp_insts, n_tsps_per_route

# def trans_tsp(coors, routes, min_reviser_size=20):
#     tsp_pis = []
#     n_tsps_per_route = []
#
#     for route in routes:
#         start = 0
#         sub_route_count = 0
#         for idx, node in enumerate(route):
#             if idx == 0:
#                 continue  # 跳过起始点
#             if node == 0:  # 仓库节点
#                 if route[idx - 1] != 0:  # 有效子路径
#                     sub_route = route[start: idx]
#                     tsp_pis.append(sub_route)
#                     sub_route_count += 1
#                 start = idx
#         n_tsps_per_route.append(sub_route_count)
#
#     # 动态计算最大长度（不低于min_reviser_size）
#     max_len = max(len(pi) for pi in tsp_pis) if tsp_pis else 0
#     max_len = max(max_len, min_reviser_size)
#
#     # 动态填充（保持原始数据不变）
#     padded_pis = []
#     for pi in tsp_pis:
#         if len(pi) < max_len:
#             # 用-1填充（避免与有效索引冲突）
#             padded = torch.cat([
#                 pi,
#                 torch.full((max_len - len(pi)), -1, dtype=pi.dtype, device=pi.device)
#             ])
#         else:
#             padded = pi
#             padded_pis.append(padded)
#
#             # 转换坐标时处理填充值
#             valid_mask = torch.stack([pi != -1 for pi in padded_pis])
#             padded_pis = torch.stack(padded_pis)
#             padded_pis[padded_pis == -1] = 0  # 临时用0索引（后续会被mask）
#
#             tsp_insts = coors[padded_pis]  # shape: (n_routes, max_len, 2)
#             tsp_insts[~valid_mask] = 0  # 将填充部分的坐标设为0
#
#     return tsp_insts, n_tsps_per_route

def sum_cost(costs, n_tsps_per_route):
    # 计算每个路径的总成本

    # 确保成本的数量与子路径的总数一致
    assert len(costs) == sum(n_tsps_per_route)

    # 如果 costs 不是张量，则转换为张量
    if not isinstance(costs, torch.Tensor):
        costs = torch.tensor(costs)

    ret = []
    start = 0
    for n in n_tsps_per_route:
        # 对每个路径的子路径成本求和
        ret.append(costs[start: start + n].sum())
        start += n

    # 返回每个路径的总成本
    return torch.stack(ret)