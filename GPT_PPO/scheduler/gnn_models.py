# scheduler/gnn_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TaskGNN(nn.Module):
    """
    TaskGNN用于处理单个任务的阶段图，生成任务的嵌入表示。
    每个任务被划分为多个阶段，形成一个图，每个阶段是图中的一个节点。
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, num_layers=2, dropout=0.6):
        """
        初始化TaskGNN模块。

        参数:
            input_dim (int): 每个阶段节点的输入特征维度。
            hidden_dim (int): GATConv的隐藏层维度。
            output_dim (int): GATConv的输出维度。
            num_heads (int, optional): 注意力头的数量。默认为4。
            num_layers (int, optional): GNN的层数。默认为2。
            dropout (float, optional): Dropout概率。默认为0.6。
        """
        super(TaskGNN, self).__init__()
        self.convs = nn.ModuleList()
        # 第一层GATConv
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        # 中间层GATConv
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
        # 最后一层GATConv，不使用多头
        self.convs.append(GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False, dropout=dropout))
        # 线性层，用于进一步转换
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index, positions=None):
        """
        前向传播。

        参数:
            x (Tensor): 节点特征矩阵，形状 [num_nodes, input_dim]。
            edge_index (LongTensor): 边索引，形状 [2, num_edges]。
            positions (Tensor, optional): 节点位置编码，形状 [num_nodes]。

        返回:
            Tensor: 节点嵌入，形状 [num_nodes, output_dim]。
        """
        # 如果有位置编码，添加到输入特征
        if positions is not None:
            # 假设位置编码已被加入到特征x中
            pass

        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
        x = self.convs[-1](x, edge_index)
        x = F.elu(x)
        x = self.fc(x)
        return x  # [num_nodes, output_dim]

class TaskAggregationGNN(nn.Module):
    """
    TaskAggregationGNN用于聚合所有任务的嵌入，生成全局任务嵌入表示。
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, num_layers=2, dropout=0.6):
        """
        初始化TaskAggregationGNN模块。

        参数:
            input_dim (int): 每个任务嵌入的输入维度。
            hidden_dim (int): GATConv的隐藏层维度。
            output_dim (int): GATConv的输出维度。
            num_heads (int, optional): 注意力头的数量。默认为4。
            num_layers (int, optional): GNN的层数。默认为2。
            dropout (float, optional): Dropout概率。默认为0.6。
        """
        super(TaskAggregationGNN, self).__init__()
        self.convs = nn.ModuleList()
        # 第一层GATConv
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        # 中间层GATConv
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
        # 最后一层GATConv，不使用多头
        self.convs.append(GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False, dropout=dropout))
        # 线性层，用于进一步转换
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index):
        """
        前向传播。

        参数:
            x (Tensor): 节点嵌入矩阵，形状 [num_tasks, input_dim]。
            edge_index (LongTensor): 边索引，形状 [2, num_edges]。

        返回:
            Tensor: 全局任务嵌入，形状 [num_tasks, output_dim]。
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
        x = self.convs[-1](x, edge_index)
        x = F.elu(x)
        x = self.fc(x)
        return x  # [num_tasks, output_dim]

class ClusterEncoder(nn.Module):
    """
    ClusterEncoder用于编码集群的状态指标，生成集群状态嵌入。
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        初始化ClusterEncoder模块。

        参数:
            input_dim (int): 集群指标的输入维度。
            hidden_dim (int): 隐藏层维度。
            output_dim (int): 输出嵌入的维度。
        """
        super(ClusterEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        前向传播。

        参数:
            x (Tensor): 集群指标，形状 [batch_size, input_dim]。

        返回:
            Tensor: 集群嵌入，形状 [batch_size, output_dim]。
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x  # [batch_size, output_dim]

class PolicyNetwork(nn.Module):
    """
    策略网络，基于任务和集群的嵌入向量，输出动作的概率分布。
    """

    def __init__(self, input_dim, hidden_dim, num_ops, num_tasks, num_nodes):
        """
        初始化PolicyNetwork。

        参数:
            input_dim (int): 输入状态向量的维度。
            hidden_dim (int): 隐藏层维度。
            num_ops (int): 动作类型的数量。
            num_tasks (int): 任务ID的数量。
            num_nodes (int): 节点ID的数量。
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)

        # 定义输出层，输出四个分布：操作类型、任务ID、源节点ID、目标节点ID
        self.fc_op = nn.Linear(hidden_dim // 2, num_ops)  # 操作类型
        self.fc_task_id = nn.Linear(hidden_dim // 2, num_tasks)  # 任务ID
        self.fc_src_node_id = nn.Linear(hidden_dim // 2, num_nodes)  # 源节点ID
        self.fc_des_node_id = nn.Linear(hidden_dim // 2, num_nodes)  # 目标节点ID

    def forward(self, state):
        """
        前向传播。

        参数:
            state (Tensor): 状态向量，形状 [batch_size, input_dim]。

        返回:
            tuple: 四个概率分布张量。
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        op_logits = self.fc_op(x)  # [batch_size, num_ops]
        task_id_logits = self.fc_task_id(x)  # [batch_size, num_tasks]
        src_node_id_logits = self.fc_src_node_id(x)  # [batch_size, num_nodes]
        des_node_id_logits = self.fc_des_node_id(x)  # [batch_size, num_nodes]

        # 转换为概率分布
        op_probs = F.softmax(op_logits, dim=-1)
        task_id_probs = F.softmax(task_id_logits, dim=-1)
        src_node_id_probs = F.softmax(src_node_id_logits, dim=-1)
        des_node_id_probs = F.softmax(des_node_id_logits, dim=-1)

        return op_probs, task_id_probs, src_node_id_probs, des_node_id_probs

class ValueNetwork(nn.Module):
    """
    价值网络，基于任务和集群的嵌入向量，输出状态的价值估计。
    """

    def __init__(self, input_dim, hidden_dim):
        """
        初始化ValueNetwork。

        参数:
            input_dim (int): 输入状态向量的维度。
            hidden_dim (int): 隐藏层维度。
        """
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)  # 输出一个价值估计

    def forward(self, state):
        """
        前向传播。

        参数:
            state (Tensor): 状态向量，形状 [batch_size, input_dim]。

        返回:
            Tensor: 价值估计，形状 [batch_size, 1]。
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)  # [batch_size, 1]
        return value  # [batch_size, 1]
