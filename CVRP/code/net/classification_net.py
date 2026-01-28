import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import torch_geometric.nn as gnn

def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

# GNN for edge embeddings
class EmbNet(nn.Module):
    def __init__(self, depth=12, feats=2, edge_feats=1, units=48, act_fn='silu', agg_fn='mean'):
        """
        初始化 EmbNet 模块，用于生成边的嵌入表示。

        Args:
            depth (int): 网络的深度（层数）。
            feats (int): 节点特征的维度。
            edge_feats (int): 边特征的维度。
            units (int): 每层的隐藏单元数。
            act_fn (str): 激活函数名称。
            agg_fn (str): 聚合函数名称（如 'mean'）。
        """
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        self.act_fn = getattr(F, act_fn)  # 获取激活函数
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')  # 获取聚合函数

        # 节点特征的处理层
        self.v_lin0 = nn.Linear(self.feats, self.units)  # 初始线性变换
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])  # 多层线性变换
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])  # 批归一化层

        # 边特征的处理层
        self.e_lin0 = nn.Linear(edge_feats, self.units)  # 初始线性变换
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])  # 多层线性变换
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])  # 批归一化层

    def reset_parameters(self):
        """重置模型参数（未实现）。"""
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr):
        """
        前向传播函数，生成边的嵌入表示。

        Args:
            x: 节点特征 (num_nodes, feats)
            edge_index: 边索引 (2, num_edges)
            edge_attr: 边特征 (num_edges, edge_feats)

        Returns:
            w: 边的嵌入表示 (num_edges, units)
        """
        x = x  # 节点特征
        w = edge_attr  # 边特征

        # 节点特征的初始变换
        x = self.v_lin0(x)
        x = self.act_fn(x)  # 激活函数

        # 边特征的初始变换
        w = self.e_lin0(w)
        w = self.act_fn(w)  # 激活函数

        # 多层图神经网络
        for i in range(self.depth):
            x0 = x  # 保存当前节点特征
            x1 = self.v_lins1[i](x0)  # 线性变换
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)

            w0 = w  # 保存当前边特征
            w1 = self.e_lins0[i](w0)  # 线性变换
            w2 = torch.sigmoid(w0)  # Sigmoid 激活函数

            # 更新节点特征
            x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            # 更新边特征
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))

        return w  # 返回边的嵌入表示


# General class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        """返回模型所在的设备（CPU 或 GPU）。"""
        return self._dummy.device

    def __init__(self, units_list, act_fn):
        """
        初始化 MLP 模块。

        Args:
            units_list (list): 每层的单元数列表。
            act_fn (str): 激活函数名称。
        """
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)  # 用于获取设备信息
        self.units_list = units_list
        self.depth = len(self.units_list) - 1  # 网络深度
        self.act_fn = getattr(F, act_fn)  # 获取激活函数
        self.lins = nn.ModuleList(
            [nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])  # 线性层
        self.dropout = nn.Dropout(0.5)  # Dropout 层，防止过拟合

    def forward(self, x, k_sparse):
        """
        前向传播函数。

        Args:
            x: 输入特征 (batch_size, input_dim)
            k_sparse: 稀疏度参数

        Returns:
            x: 输出特征 (batch_size, output_dim)
        """
        for i in range(self.depth):
            x = self.lins[i](x)  # 线性变换
            if i < self.depth - 1:
                x = self.act_fn(x)  # 激活函数
                x = self.dropout(x)  # Dropout
            else:
                x = x.reshape(-1, k_sparse)  # 调整形状
                x = torch.softmax(x, dim=1)  # Softmax 归一化
                x = x.flatten()  # 展平
        return x


# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, k_sparse, depth=3, units=48, preds=1, act_fn='silu'):
        """
        初始化 ParNet 模块，用于预测参数化 theta。

        Args:
            k_sparse: 稀疏度参数
            depth: 网络深度
            units: 每层的单元数
            preds: 输出维度
            act_fn: 激活函数名称
        """
        self.units = units
        self.preds = preds
        self.k_sparse = k_sparse
        super().__init__([self.units] * depth + [self.preds], act_fn)

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x: 输入特征

        Returns:
            输出特征
        """
        return super().forward(x, self.k_sparse).squeeze(dim=-1)  # 去除多余的维度


# 主网络
class Net(nn.Module):
    def __init__(self, units, feats, k_sparse, edge_feats=1, depth=12):
        """
        初始化主网络。

        Args:
            units: 每层的单元数
            feats: 节点特征的维度
            k_sparse: 稀疏度参数
            edge_feats: 边特征的维度
            depth: 网络深度
        """
        super().__init__()
        self.emb_net = EmbNet(depth=depth, units=units, feats=feats, edge_feats=edge_feats)  # 边的嵌入网络
        self.par_net_heu = ParNet(units=units, k_sparse=k_sparse)  # 参数化网络

    def forward(self, pyg):
        """
        前向传播函数。

        Args:
            pyg: 图数据（包含节点特征、边索引和边特征）

        Returns:
            heu: 启发式向量
        """
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        emb = self.emb_net(x, edge_index, edge_attr)  # 生成边的嵌入
        heu = self.par_net_heu(emb)  # 生成启发式向量
        return heu

    @staticmethod
    def reshape(pyg, vector):
        """
        将启发式向量转换为矩阵形式。

        Args:
            pyg: 图数据
            vector: 启发式向量

        Returns:
            matrix: 矩阵形式的启发式向量
        """
        n_nodes = pyg.x.shape[0]  # 节点数
        device = pyg.x.device  # 设备信息
        matrix = torch.zeros(size=(n_nodes, n_nodes), device=device)  # 初始化矩阵
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = vector  # 填充矩阵
        try:
            assert (matrix.sum(dim=1) >= 0.99).all()  # 检查矩阵是否有效
        except:
            torch.save(matrix, './error_reshape.pt')  # 保存错误矩阵
        return matrix



class PPOactor(nn.Module):
    """
      Actor class for **PPO** with stochastic, learnable, **state-independent** log standard deviation.

      :param mid_dim[int]: the middle dimension of networks
      :param state_dim[int]: the dimension of state (the number of state vector)
      :param action_dim[int]: the dimension of action (the number of discrete action)
      """

    def __init__(self, state_dim, mid_dim, action_dim, init_a_std_log=-0.5):
        super().__init__()

        nn_middle = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
        )

        self.net = nn.Sequential(
            nn_middle,
            nn.Linear(mid_dim, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, action_dim),
        )

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.a_std_log = nn.Parameter(
            torch.ones((1, action_dim)).mul_(init_a_std_log), requires_grad=True
        )  # calculated from action space
        self.register_parameter("a_std_log", self.a_std_log)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

        self.reset_parameter()

    def reset_parameter(self):
        for name, module in self.net.named_modules():
            if isinstance(module, torch.nn.Linear):
                layer_norm(module)
        # rescale last layer
        last_layer = self.net[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        layer_norm(last_layer, 0.01)

    def forward(self, state):
        """
        The forward function.

        :param state[np.array]: the input state.
        :return: the output tensor.
        """
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state):
        """
        The forward function with Gaussian noise.

        :param state[np.array]: the input state.
        :return: the action and added noise.
        """
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def get_logprob_entropy(self, state, action):
        """
        Compute the log of probability with current network.

        :param state[np.array]: the input state.
        :param action[float]: the action.
        :return: the log of probability and entropy.
        """
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        dist = torch.distributions.Normal(a_avg, a_std)
        logprob = dist.log_prob(action).sum(1)
        dist_entropy = -dist.entropy().mean()
        del dist

        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise):  # noise = action - a_noise
        """
        Compute the log of probability with old network.

        :param _action[float]: the action.
        :param noise[float]: the added noise when exploring.
        :return: the log of probability with old network.
        """
        delta = noise.pow(2) * 0.5
        return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # old_logprob

    def get_old_logprob_act(self, old_action, old_noise, action):
        """
        Compute the log of probability with out new noise.

        :param _action[float]: the action.
        :param noise[float]: the added noise when exploring.
        :return: the log of probability with old network.
        """
        a_std = self.a_std_log.exp()
        noise = (old_action - action) / a_std - old_noise
        delta = noise.pow(2) * 0.5
        return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # old_logprob

class PPOcritic(nn.Module):
    """
        The Critic class for **PPO**.

        :param mid_dim[int]: the middle dimension of networks
        :param state_dim[int]: the dimension of state (the number of state vector)
        :param action_dim[int]: the dimension of action (the number of discrete action)
        """

    def __init__(self, state_dim, mid_dim, _action_dim):
        super().__init__()

        nn_middle = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
        )

        self.net = nn.Sequential(
            nn_middle,
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, 1),
        )
        self.reset_parameter()

    def reset_parameter(self):
        for name, module in self.net.named_modules():
            if isinstance(module, torch.nn.Linear):
                layer_norm(module)

    def forward(self, state):
        """
        The forward function to ouput the value of the state.

        :param state[np.array]: the input state.
        :return: the output tensor.
        """
        return self.net(state)  # advantage value
