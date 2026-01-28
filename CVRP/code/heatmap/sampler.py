import torch
from torch.distributions import Categorical  # 用于概率分布采样
import math

class Sampler():
    def __init__(self, demand, heatmap, capacity, bs, device):
        """
        初始化 Sampler 类。

        Args:
            demand: 每个节点的需求量 (n,)
            heatmap: 热力图 (n, n)，表示节点之间的转移概率
            capacity: 车辆的容量
            bs: 批次大小
            device: 计算设备（如 CPU 或 GPU）
        """
        self.n = demand.size(0)  # 节点数量
        self.demand = demand.to(device)  # 将需求量移动到指定设备
        self.heatmap = heatmap.to(device)  # 将热力图移动到指定设备
        self.capacity = capacity  # 车辆容量
        self.max_vehicle = math.ceil(sum(self.demand) / capacity) + 1  # 最大车辆数量
        self.total_demand = self.demand.sum()  # 总需求量
        self.bs = bs  # 批次大小
        self.ants_idx = torch.arange(bs)  # 批次索引
        self.device = device  # 计算设备

    def gen_subsets(self, require_prob=False, greedy_mode=False):
        """
        生成路径子集。

        Args:
            require_prob: 是否需要返回路径的概率
            greedy_mode: 是否使用贪婪模式

        Returns:
            如果 require_prob=True，返回路径和对应的概率；否则只返回路径。
        """
        if greedy_mode:
            assert not require_prob  # 贪婪模式下不需要概率

        # 初始化动作（选择的节点）
        actions = torch.zeros((self.bs,), dtype=torch.long, device=self.device)
        # 初始化访问掩码（标记哪些节点已被访问）
        visit_mask = torch.ones(size=(self.bs, self.n), device=self.device)
        visit_mask = self.update_visit_mask(visit_mask, actions)  # 更新访问掩码
        # 初始化已使用的容量
        used_capacity = torch.zeros(size=(self.bs,), device=self.device)
        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)  # 更新容量掩码

        # 初始化车辆计数和需求计数
        vehicle_count = torch.zeros((self.bs,), device=self.device)
        demand_count = torch.zeros((self.bs,), device=self.device)
        # 更新仓库掩码
        depot_mask, vehicle_count, demand_count = self.update_depot_mask(vehicle_count, demand_count, actions, capacity_mask, visit_mask)

        # 存储路径和概率
        paths_list = [actions]
        log_probs_list = []
        done = self.check_done(visit_mask, actions)  # 检查是否完成

        # 循环直到所有节点被访问
        while not done:
            # 选择下一个节点
            actions, log_probs = self.pick_node(actions, visit_mask, capacity_mask, depot_mask, require_prob, greedy_mode)
            paths_list.append(actions)  # 记录路径
            if require_prob:
                log_probs_list.append(log_probs)  # 记录概率
                visit_mask = visit_mask.clone()  # 克隆访问掩码
                depot_mask = depot_mask.clone()  # 克隆仓库掩码

            # 更新访问掩码、容量掩码和仓库掩码
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
            depot_mask, vehicle_count, demand_count = self.update_depot_mask(vehicle_count, demand_count, actions, capacity_mask, visit_mask)
            done = self.check_done(visit_mask, actions)  # 检查是否完成

        # 返回结果
        if require_prob:
            return torch.stack(paths_list).permute(1, 0), torch.stack(log_probs_list).permute(1, 0)
        else:
            return torch.stack(paths_list).permute(1, 0)

    def pick_node(self, prev, visit_mask, capacity_mask, depot_mask, require_prob, greedy_mode=False):
        """
        选择下一个节点。

        Args:
            prev: 上一个节点
            visit_mask: 访问掩码
            capacity_mask: 容量掩码
            depot_mask: 仓库掩码
            require_prob: 是否需要返回概率
            greedy_mode: 是否使用贪婪模式

        Returns:
            选择的节点和对应的概率（如果需要）。
        """
        log_prob = None
        heatmap = self.heatmap[prev]  # 获取当前节点的热力图
        dist = (heatmap * visit_mask * capacity_mask * depot_mask)  # 计算选择概率

        if not greedy_mode:
            try:
                # 使用 Categorical 分布采样
                dist = Categorical(dist)
                item = dist.sample()  # 采样节点
                log_prob = dist.log_prob(item) if require_prob else None  # 计算对数概率
            except:
                # 如果分布无效，使用 softmax 和 multinomial 采样
                dist = torch.softmax(torch.log(dist), dim=1)
                item = torch.multinomial(dist, num_samples=1).squeeze()
                log_prob = torch.log(dist[torch.arange(self.bs), item])
        else:
            # 贪婪模式：选择概率最大的节点
            _, item = dist.max(dim=1)

        return item, log_prob

    def update_depot_mask(self, vehicle_count, demand_count, actions, capacity_mask, visit_mask):
        """
        更新仓库掩码。

        Args:
            vehicle_count: 车辆计数
            demand_count: 需求计数
            actions: 当前动作（选择的节点）
            capacity_mask: 容量掩码
            visit_mask: 访问掩码

        Returns:
            更新后的仓库掩码、车辆计数和需求计数。
        """
        depot_mask = torch.ones((self.bs, self.n), device=self.device)  # 初始化仓库掩码

        # 更新车辆计数和需求计数
        vehicle_count[actions == 0] += 1  # 如果返回仓库，车辆计数加 1
        demand_count += self.demand[actions]  # 更新需求计数

        # 计算剩余需求
        remaining_demand = self.total_demand - demand_count

        # 更新仓库掩码
        depot_mask[remaining_demand > self.capacity * (self.max_vehicle - vehicle_count), 0] = 0  # 如果剩余需求超过容量，屏蔽仓库
        depot_mask[((visit_mask[:, 1:] * capacity_mask[:, 1:]) == 0).all(dim=1), 0] = 1  # 如果没有可用节点，解除仓库屏蔽

        return depot_mask, vehicle_count, demand_count

    def update_visit_mask(self, visit_mask, actions):
        """
        更新访问掩码。

        Args:
            visit_mask: 当前访问掩码
            actions: 当前动作（选择的节点）

        Returns:
            更新后的访问掩码。
        """
        visit_mask[torch.arange(self.bs, device=self.device), actions] = 0  # 标记已访问的节点
        visit_mask[:, 0] = 1  # 仓库可以被重新访问
        visit_mask[(actions == 0) * (visit_mask[:, 1:] != 0).any(dim=1), 0] = 0  # 如果还有未访问的节点，屏蔽仓库
        return visit_mask

    def update_capacity_mask(self, cur_nodes, used_capacity):
        """
        更新容量掩码。

        Args:
            cur_nodes: 当前节点
            used_capacity: 已使用的容量

        Returns:
            更新后的已使用容量和容量掩码。
        """
        capacity_mask = torch.ones(size=(self.bs, self.n), device=self.device)  # 初始化容量掩码

        # 更新已使用的容量
        used_capacity[cur_nodes == 0] = 0  # 如果返回仓库，重置容量
        used_capacity = used_capacity + self.demand[cur_nodes]  # 更新容量

        # 更新容量掩码
        remaining_capacity = self.capacity - used_capacity  # 计算剩余容量
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.n)  # 扩展剩余容量
        demand_repeat = self.demand.unsqueeze(0).repeat(self.bs, 1)  # 扩展需求量
        capacity_mask[demand_repeat > remaining_capacity_repeat] = 0  # 如果需求量超过剩余容量，屏蔽节点

        return used_capacity, capacity_mask

    def check_done(self, visit_mask, actions):
        """
        检查是否完成路径生成。

        Args:
            visit_mask: 访问掩码
            actions: 当前动作（选择的节点）

        Returns:
            是否完成路径生成。
        """
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()  # 所有非仓库节点被访问且返回仓库