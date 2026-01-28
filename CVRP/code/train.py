import sys

sys.path.insert(0, './')  # 将当前目录添加到系统路径，以便导入本地模块
import torch
import time
import argparse
import os
from tqdm import tqdm  # 用于显示进度条
from heatmap.inst import gen_inst, gen_pyg_data, trans_tsp  # 生成实例、图数据和转换TSP问题
from heatmap.eval import eval  # 评估函数
from heatmap.sampler import Sampler  # 采样器
from net.classification_net import Net  # 分区网络
from utils.function import load_model  # 加载模型的工具函数
#
checkpoint_dir = './checkpoints'
EPS = 1e-10  # 极小值，用于数值稳定性
DEVICE = 'cuda:0'  # 使用 GPU 设备
LR = 3e-4  # 学习率
K_SPARSE = {  # 稀疏度参数，根据问题规模设置
    200: 20,
    500: 50,
    1000: 100,
    2000: 200,
}


def infer_heatmap(model, pyg_data):
    """
    生成热力图。

    Args:
        model: 模型
        pyg_data: 图数据（PyG格式）

    Returns:
        heatmap: 热力图
    """
    heatmap = model(pyg_data)  # 通过模型生成热力图
    heatmap = heatmap / (heatmap.min() + 1e-5)  # 归一化热力图
    heatmap = model.reshape(pyg_data, heatmap) + 1e-5  # 将热力图转换为矩阵形式
    return heatmap


def train_batch(model, optimizer, n, bs, opts):
    """
    训练一个批次的数据。

    Args:
        model: 模型
        optimizer: 优化器
        n: 问题规模
        bs: 批次大小
        opts: 配置参数
    """
    model.train()  # 设置模型为训练模式
    loss_lst = []  # 存储每个样本的损失
    for _ in range(opts.batch_size):
        # 生成实例（坐标、需求和容量）
        coors, demand, capacity = gen_inst(n, DEVICE)
        # 将实例转换为图数据
        pyg_data = gen_pyg_data(coors, demand, capacity, K_SPARSE[n])
        # 生成热力图
        heatmap = infer_heatmap(model, pyg_data)
        # 使用采样器生成路径
        sampler = Sampler(demand, heatmap, capacity, bs, DEVICE)
        routes, log_probs = sampler.gen_subsets(require_prob=True)
        # 将路径转换为TSP实例
        tsp_insts, n_tsps_per_route = trans_tsp(coors, routes)
        # 评估路径的目标值
        objs = eval(tsp_insts, n_tsps_per_route, opts)
        baseline = objs.mean()  # 计算基线值
        log_probs = log_probs.to(DEVICE)
        # 计算强化学习损失
        reinforce_loss = torch.sum((objs - baseline) * log_probs.sum(dim=1)) / bs
        loss_lst.append(reinforce_loss)

    # 计算平均损失
    loss = sum(loss_lst) / opts.batch_size
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播
    if not opts.no_clip:
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=opts.max_norm, norm_type=2)
    optimizer.step()  # 更新参数


def infer_instance(model, inst, opts):
    """
    推理单个实例。

    Args:
        model: 模型
        inst: 实例（坐标、需求和容量）
        opts: 配置参数

    Returns:
        obj: 目标值
    """
    model.eval()  # 设置模型为评估模式
    coors, demand, capacity = inst
    n = demand.size(0) - 1  # 问题规模
    # 将实例转换为图数据
    pyg_data = gen_pyg_data(coors, demand, capacity, K_SPARSE[n])
    # 生成热力图
    heatmap = infer_heatmap(model, pyg_data)
    # 使用采样器生成路径（贪婪模式）
    sampler = Sampler(demand, heatmap, capacity, 1, DEVICE)
    routes = sampler.gen_subsets(require_prob=False, greedy_mode=True)
    # 将路径转换为TSP实例
    tsp_insts, n_tsps_per_route = trans_tsp(coors, routes)
    # 评估路径的目标值
    obj = eval(tsp_insts, n_tsps_per_route, opts).min()
    return obj


def train_epoch(n, bs, steps_per_epoch, net, optimizer, scheduler, opts):
    """
    训练一个 epoch。

    Args:
        n: 问题规模
        bs: 批次大小
        steps_per_epoch: 每个 epoch 的步数
        net: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        opts: 配置参数
    """
    for _ in tqdm(range(steps_per_epoch)):
        train_batch(net, optimizer, n, bs, opts)  # 训练一个批次
    scheduler.step()  # 更新学习率


@torch.no_grad()
def validation(n, net, opts):
    """
    验证模型性能。

    Args:
        n: 问题规模
        net: 模型
        opts: 配置参数

    Returns:
        avg_obj: 平均目标值
    """
    sum_obj = 0
    for _ in range(opts.val_size):
        # 生成实例
        inst = gen_inst(n, DEVICE)
        # 推理实例
        obj = infer_instance(net, inst, opts)
        sum_obj += obj
    avg_obj = sum_obj / opts.val_size  # 计算平均目标值
    print(avg_obj)
    return avg_obj


def train(n, bs, steps_per_epoch, n_epochs, opts):
    """
    训练模型。

    Args:
        n: 问题规模
        bs: 批次大小
        steps_per_epoch: 每个 epoch 的步数
        n_epochs: 总 epoch 数
        opts: 配置参数
    """
    # revisers = []  # 存储修订器
    # for reviser_size in opts.revision_lens:
    #     # 加载预训练的修订器
    #     reviser_path = f'pretrained/Reviser-stage2/reviser_{reviser_size}/epoch-299.pt'
    #     reviser, _ = load_model(reviser_path, is_local=True)
    #     revisers.append(reviser)
    # for reviser in revisers:
    #     reviser.to(DEVICE)  # 将修订器移动到 GPU
    #     reviser.eval()  # 设置修订器为评估模式
    #     reviser.set_decode_type(opts.decode_strategy)  # 设置解码策略
    # opts.revisers = revisers  # 将修订器添加到配置参数

    # 初始化模型
    net = Net(opts.units, 3, K_SPARSE[n], 2, depth=opts.depth).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR)  # 使用 AdamW 优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs)  # 余弦退火学习率调度器
    file_path = os.path.abspath(r'checkpoints\cvrp-500-1-cos.pt')
    # 加载检查点（如果有）
    if opts.checkpoint_path == '':
        starting_epoch = 1
    else:
        checkpoint = torch.load(opts.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        print("已加载检查点")



    sum_time = 0  # 总训练时间
    best_avg_obj = validation(n, net, opts)  # 初始验证性能
    print('epoch 0', best_avg_obj.item())
    for epoch in range(starting_epoch, n_epochs + 1):
        start = time.time()
        train_epoch(n, bs, steps_per_epoch, net, optimizer, scheduler, opts)  # 训练一个 epoch
        sum_time += time.time() - start
        avg_obj = validation(n, net, opts)  # 验证性能
        print(f'epoch {epoch}: ', avg_obj.item())
        if best_avg_obj > avg_obj:
            best_avg_obj = avg_obj
            print(f'Save checkpoint-{epoch}.')
            # 保存检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            checkpoint_path = os.path.join(checkpoint_dir, f'cvrp-{n}-{epoch}-cos.pt')
            torch.save(checkpoint, checkpoint_path)
    print('total training duration:', sum_time)  # 输出总训练时间


if __name__ == '__main__':
    import pprint as pp

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_size', type=int, default=500, help='问题规模')
    parser.add_argument('--revision_lens', nargs='+', default=[20], type=int, help='修订器的大小')
    parser.add_argument('--revision_iters', nargs='+', default=[5], type=int, help='修订迭代次数')
    parser.add_argument('--decode_strategy', type=str, default='greedy', help='解码策略')
    parser.add_argument('--width', type=int, default=10, help='宽度参数')
    parser.add_argument('--no_aug', action='store_true', help='禁用实例增强')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    parser.add_argument('--val_size', type=int, default=20, help='验证集大小')
    parser.add_argument('--n_epochs', type=int, default=20, help='总 epoch 数')
    parser.add_argument('--steps_per_epoch', type=int, default=256, help='每个 epoch 的步数')
    parser.add_argument('--checkpoint_path', type=str, default='E:\论文-毕业\code\checkpoints\cvrp-500-9-cos.pt', help='检查点路径')
    parser.add_argument('--max_norm', type=float, default=1, help='梯度裁剪的最大范数')
    parser.add_argument('--units', type=int, default=48, help='每层的单元数')
    parser.add_argument('--no_clip', action='store_true', help='禁用梯度裁剪')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--depth', type=int, default=12, help='网络深度')
    opts = parser.parse_args()
    opts.no_aug = True  # 禁用实例增强
    opts.no_prune = False  # 不禁用剪枝
    opts.problem_type = 'tsp'  # 问题类型为 TSP

    torch.manual_seed(opts.seed)  # 设置随机种子
    pp.pprint(vars(opts))  # 打印配置参数
    train(opts.problem_size, opts.width, opts.steps_per_epoch, opts.n_epochs, opts)  # 开始训练