import torch
import numpy as np
import utils.random_insertion as ri # 导入自定义模块
import matplotlib.pyplot as plt

def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    if isinstance(arr, list):
        return np.array(arr)
    else:
        return arr

def random_insertion(cities, order=None):
    cities = _to_numpy(cities)
    order = _to_numpy(order)
    return ri.tsp_random_insertion(cities, order)

def random_insertion_parallel(cities, orders):
    cities = _to_numpy(cities)
    orders = _to_numpy(orders)
    return ri.tsp_random_insertion_parallel(cities, orders)

def random_insertion_non_euclidean(distmap, order):
    distmap = _to_numpy(distmap)
    order = _to_numpy(order)
    return ri.atsp_random_insertion(distmap, order)


def print_city_coordinates(cities, order):
    """
    打印每个点对应的坐标。
    :param cities: 城市坐标，形状为 (n, 2) 的 NumPy 数组。
    :param order: 城市访问顺序。
    """
    print("TSP 路径中每个点的坐标：")
    for i, city_index in enumerate(order):
        print(f"点 {city_index}: ({cities[city_index, 0]:.2f}, {cities[city_index, 1]:.2f})")




def plot_tsp_path(cities, order, title="TSP 路径可视化"):
    """
    可视化 TSP 路径。
    :param cities: 城市坐标，形状为 (n, 2) 的 NumPy 数组。
    :param order: 城市访问顺序。
    :param title: 图标题。
    """
    # 提取坐标
    x = cities[:, 0]
    y = cities[:, 1]

    # 绘制点
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c='red', label='城市点')

    # 绘制路径
    for i in range(len(order) - 1):
        start_index = order[i]
        end_index = order[i + 1]
        plt.plot([x[start_index], x[end_index]], [y[start_index], y[end_index]], 'b-')

    # 连接最后一个点和起点
    start_index = order[-1]
    end_index = order[0]
    plt.plot([x[start_index], x[end_index]], [y[start_index], y[end_index]], 'b-')

    # 添加标签
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, f'{i}', fontsize=12, ha='right')

    # 设置标题和标签
    plt.title(title)
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.legend()
    plt.grid(True)
    plt.show()


