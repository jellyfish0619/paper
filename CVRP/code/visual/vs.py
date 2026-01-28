import matplotlib.pyplot as plt


def plot_radical_chart(coors, demand): #绘制出求解前的原始图像
    # 提取仓库和客户的坐标
    warehouse = coors[0]  # 仓库坐标
    customers = coors[1:]  # 客户坐标

    # 绘制仓库和客户的位置
    plt.figure(figsize=(8, 6))
    plt.scatter(warehouse[0], warehouse[1], c='red', label='Warehouse', s=100, marker='s')  # 仓库用红色方块表示
    plt.scatter(customers[:, 0], customers[:, 1], c='blue', label='Customers', s=50)  # 客户用蓝色圆点表示
    for i, (x, y) in enumerate(customers):
        plt.text(x, y, f'{demand[i+1]}', fontsize=12, ha='right')  # 在客户点旁边显示需求量

    # 添加标签和标题
    plt.title("Vehicle Routes with Demand")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_routes_with_demand(coors, demand, routes):#绘制出求解后的图像
    # 提取仓库和客户的坐标
    warehouse = coors[0]  # 仓库坐标
    customers = coors[1:]  # 客户坐标

    # 绘制仓库和客户的位置
    plt.figure(figsize=(8, 6))
    plt.scatter(warehouse[0], warehouse[1], c='red', label='Warehouse', s=100, marker='s')  # 仓库用红色方块表示
    plt.scatter(customers[:, 0], customers[:, 1], c='blue', label='Customers', s=50)  # 客户用蓝色圆点表示

    # 添加客户需求量的文本标签
    for i, (x, y) in enumerate(customers):
        plt.text(x, y, f'{demand[i+1]}', fontsize=12, ha='right')  # 在客户点旁边显示需求量

    # 绘制路径
    colors = ['green', 'purple', 'orange', 'brown']  # 不同路径的颜色
    for i, route in enumerate(routes):
        route_coors = coors[route]  # 获取路径中的坐标
        plt.plot(route_coors[:, 0], route_coors[:, 1], marker='o', color=colors[i % len(colors)], label=f'Route {i+1}')

    # 添加标签和标题
    plt.title("Vehicle Routes with Demand")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

