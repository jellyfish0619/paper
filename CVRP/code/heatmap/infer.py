import sys
import torch

# 将当前目录添加到系统路径中，以便导入自定义模块
sys.path.insert(0, './')

# 导入自定义模块
from heatmap.inst import gen_pyg_data  # 用于生成图数据的函数
from heatmap.train import infer_heatmap  # 用于推理热图的函数
from net.classification_net import Net  # 导入分区网络模型

# 定义一个极小的常数，用于防止除零错误等
EPS = 1e-10

# 定义一个字典，表示不同问题规模下的稀疏参数k
K_SPARSE = {
    100: 10,
    200: 20,
    501: 50,
    1000: 100,
    2000: 200,
    5000: 200,
    7000: 200
}

# 加载分区模型的函数
def load_partitioner(n, device, ckpt_path, k_sparse=None, depth=None):
    # 如果未提供k_sparse，则根据问题规模n从K_SPARSE字典中获取
    k_sparse = K_SPARSE[n] if k_sparse is None else k_sparse
    # 如果未提供depth，则默认设置为12
    depth = 12 if depth is None else depth
    # 初始化网络模型，输入特征维度为48，输出特征维度为3，k_sparse为稀疏参数，2表示其他参数
    net = Net(48, 3, k_sparse, 2, depth=depth)
    # 如果未提供ckpt_path，则使用默认的预训练模型路径
    ckpt_path = f'./pretrained/Partitioner/cvrp/cvrp-{n}.pt' if ckpt_path == '' else ckpt_path
    # 加载模型权重
    ckpt = torch.load(ckpt_path, map_location=device)
    print('  [*] Loading model from {}'.format(ckpt_path))
    # 如果checkpoint中包含'model_state_dict'，则加载它，否则直接加载整个checkpoint
    if ckpt.get('model_state_dict', None):
        net.load_state_dict(ckpt['model_state_dict'])
    else:
        net.load_state_dict(ckpt)
    # 将模型移动到指定设备（如GPU或CPU）
    return net.to(device)

# 推理函数，用于生成热图
@torch.no_grad()  # 禁用梯度计算，节省内存和计算资源
def infer(model, coors, demand, capacity, k_sparse=None, is_cvrplib=False):
    # 将模型设置为评估模式
    model.eval()
    # 获取问题的规模n（需求点的数量）
    n = demand.size(0)-1
    # 如果未提供k_sparse，则根据问题规模n从K_SPARSE字典中获取
    k_sparse = K_SPARSE[n] if k_sparse is None else k_sparse
    # 生成图数据
    pyg_data = gen_pyg_data(coors, demand, capacity, k_sparse, cvrplib=is_cvrplib)
    # 推理热图
    heatmap = infer_heatmap(model, pyg_data)
    # 返回生成的热图
    return heatmap