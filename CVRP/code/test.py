import sys
import torch
import time
import argparse

import numpy as np
from heatmap.inst import gen_inst, gen_pyg_data, trans_tsp
from heatmap.eval import eval
from heatmap.sampler import Sampler
from net.classification_net import Net
from utils.or_tools_contrast import solve_cvrp, extract_routes, calculate_route_costs
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from utils.plot_heatmap import visualize_heatmap
from utils.function import load_vrp_data


checkpoint_dir = './checkpoints'
EPS = 1e-10
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 自动选择 GPU 或 CPU
LR = 3e-4
K_SPARSE = {
    200: 20,
    500: 50,
    1000: 100,
    2000: 200,
}


def infer_heatmap(model, pyg_data):
    heatmap = model(pyg_data)
    heatmap = heatmap / (heatmap.min() + 1e-5)  # 归一化
    heatmap = model.reshape(pyg_data, heatmap) + 1e-5
    return heatmap


def infer_instance(model, inst, opts):
    model.eval()
    coors, demand, capacity = inst
    n = demand.size(0) - 1

    pyg_data = gen_pyg_data(coors, demand, capacity, K_SPARSE[n])

    start_time = time.time()  # 记录求解开始时间
    heatmap = infer_heatmap(model, pyg_data)

    sampler = Sampler(demand, heatmap, capacity, 1, DEVICE)
    routes = sampler.gen_subsets(require_prob=False, greedy_mode=True)

    tsp_insts, n_tsps_per_route = trans_tsp(coors, routes)
    obj = eval(tsp_insts, n_tsps_per_route, opts).min()
    solve_time = time.time() - start_time  # 计算求解时间

    return obj, routes, coors.cpu().numpy(), solve_time, heatmap  # 确保 coors 是 NumPy 数组




@torch.no_grad()
def validation(n, net, opts):
    """
    评估模型性能
    """
    total_obj = 0
    total_time = 0
    for _ in range(opts.val_size):
        #随机生成
        inst = gen_inst(n, 'cuda' if torch.cuda.is_available() else 'cpu')
        # file_path = 'X-n1001-k43.vrp.txt'
        # inst = load_vrp_data(file_path, 'cuda' if torch.cuda.is_available() else 'cpu', max_customers = 500)

        obj, routes, coors, solve_time, heatmap = infer_instance(net, inst, opts)

        #打印热力图
        fig = visualize_heatmap(heatmap, title="500x500 Node Heatmap", cmap="magma",)

        plt.show()
        # **确保 coors 是 NumPy 数组**
        if isinstance(coors, torch.Tensor):
            coors = coors.cpu().numpy()

        # **确保 routes 是 NumPy 数组**
        routes = [np.array(route.cpu()) for route in routes]

        total_obj += obj
        total_time += solve_time
        # plot_routes(routes, coors)

    avg_obj = total_obj / opts.val_size
    avg_time = total_time / opts.val_size
    print(f'平均目标值: {avg_obj:.2f}, 平均求解时间: {avg_time:.4f}s')
    return avg_obj, avg_time

def plot_routes(routes, coordinates):
    title = "Vehicle Routing Solution"
    plt.figure(figsize=(10, 8))

    colors = ['#6495ED' for _ in routes]

    # 提取坐标
    x = np.array([coord[0] for coord in coordinates])
    y = np.array([coord[1] for coord in coordinates])

    # 画出所有客户点
    plt.scatter(x[1:], y[1:], c='black', s=100, zorder=3, label='Customers')

    # 画出仓库（假设索引 0 是仓库）
    plt.scatter(x[0], y[0], c='red', s=200, marker='s', zorder=4, label='Depot')

    # 绘制路径
    ax = plt.gca()
    for i, route in enumerate(routes):
        route_coords = np.array([coordinates[node] for node in route])
        segments = np.stack([route_coords[:-1], route_coords[1:]], axis=1)

        # 用 LineCollection 批量绘制线路
        lc = LineCollection(segments, colors=[colors[i]], linewidths=2, linestyle='-')
        ax.add_collection(lc)

        # 在路径的中点标记路线编号
        mid_idx = len(route_coords) // 2
        plt.text(route_coords[mid_idx][0], route_coords[mid_idx][1],
                 f'R{i + 1}', color=colors[i], fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # 画出方向箭头
        for j in range(len(route_coords) - 1):
            dx = route_coords[j + 1, 0] - route_coords[j, 0]
            dy = route_coords[j + 1, 1] - route_coords[j, 1]
            ax.annotate("", xy=(route_coords[j + 1, 0], route_coords[j + 1, 1]),
                        xytext=(route_coords[j, 0], route_coords[j, 1]),
                        arrowprops=dict(arrowstyle="->", color=colors[i], lw=2))

    # 设置图例和标题
    plt.title(title, fontsize=14)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.axis('equal')  # 保持纵横比相等
    plt.show()


def eval_plot(n, bs, steps_per_epoch, n_epochs, opts):
    net = Net(opts.units, 3, K_SPARSE[n], 2, depth=opts.depth).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs)

    if opts.checkpoint_path:
        checkpoint = torch.load(opts.checkpoint_path, map_location=DEVICE)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("已加载检查点")

    best_avg_obj, avg_time = validation(n, net, opts)
    print(f'初始评估: 平均目标值 = {best_avg_obj:.2f}, 平均求解时间 = {avg_time:.4f}s')


if __name__ == '__main__':
    import pprint as pp

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_size', type=int, default=1000)
    parser.add_argument('--val_size', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--steps_per_epoch', type=int , default=256)
    parser.add_argument('--checkpoint_path', type=str, default='C://Users//surface//Desktop//论文-毕业//code//checkpoints//cvrp-500-9-cos.pt')
    parser.add_argument('--units', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--depth', type=int, default=12)
    opts = parser.parse_args()

    torch.manual_seed(1)
    pp.pprint(vars(opts))
    #得出model的结果
    eval_plot(opts.problem_size, opts.batch_size, opts.steps_per_epoch, opts.n_epochs, opts)

    #得出or-tools求解结果

    solution, routing, manager, solve_time, data, coors = solve_cvrp(opts.problem_size)

    if solution:
        # 提取路径并按照您的算法计算
        routes = extract_routes(data, manager, routing, solution)
        route_costs, n_tsps = calculate_route_costs(routes, data['distance_matrix'])

        # 输出结果
        print(f"Total distance (sum_cost method): {route_costs.sum().item():.2f}")
        print(f"Solved in {solve_time:.2f}s")
        # plot_routes(routes, coors)
        # # 验证OR-Tools原始输出
        # ortools_distance = solution.ObjectiveValue()
        # print(f"OR-Tools raw objective: {ortools_distance:.2f} (should match {route_costs.sum().item():.2f})")
    else:
        print("No solution found!")