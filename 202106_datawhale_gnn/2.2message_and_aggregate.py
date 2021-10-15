import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import Planetoid
from torch_sparse import SparseTensor


# 创建一个仅包含一次“消息传递过程”的图神经网络的方法
# 没有循环，传递了一次就没了，就结束了
class GCNConv(MessagePassing):
    """
    消息传递 与 消息聚合 可以融合在一起
    """
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add', flow='source_to_target')
        # Add 是 聚合操作
        # flow 是 流向，从源节点到目标节点传播信息
        print("1.初始化")
        self.lin = torch.nn.Linear(in_channels, out_channels)  # 线性变换

    def forward(self, x, edge_index):
        print("2.输入节点进行处理")
        # x shape [N, in_channels]
        # edge_index shape [2, E]  E 是 边数，边索引

        # 第一步：添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 第二步：线性变换，对节点进行线性变换，变换维度
        x = self.lin(x)  # 输入节点维度：1433，输出维度为：64

        # 第三步：使用度计算正则化
        row, col = edge_index  # 边的信息 第一行 第二行
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # 归一化系数

        # 第四步、第五步：开始启动 消息传递
        adjmat = SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(edge_index.shape[1]))
        # 如果 x 里面包含了 x_i(中心节点), x_j（邻居节点），那么message可以指定接收哪个属性的信息
        return self.propagate(adjmat, x=x, norm=norm, deg=deg.view((-1, 1)))

    def message(self, x_j, norm, deg_i):
        """
        重写 message
        """
        print("4.先对 massage进行处理")
        # x_j 就是每一个节点，norm：正则化结果, deg_i：度的信息 shape [E, 1]
        # message的消息从 propagate得来的
        # update使用默认的就可以了，所以没有进行修改
        # 对 x_j 乘以对应的正则化系数，就是说 每个邻居节点进行归一化，然后进行update
        return norm.view(-1, 1) * x_j * deg_i

    def aggregate(self, inputs, index, ptr, dim_size):
        """
        重写 aggregate
        这里 就是打印了一下信息，确认了这个被调用了，然后就没了一切都没有修改
        """
        print("5.聚合信息")
        print("self.aggr: ", self.aggr)
        print("aggregate is called")
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)

    def message_and_aggregate(self, adj_t, x, norm):
        print("3.如果这个实现了，那么会先调用这个")
        print('`message_and_aggregate` is called')
        # 只是打印消息，并没有什么具体的操作


class GCNConv2(MessagePassing):
    """
    对 update 进行 重写
    """
    def __init__(self, in_channels, out_channels):
        super(GCNConv2, self).__init__(aggr='add', flow='source_to_target')
        # Add 是 聚合操作
        # flow 是 流向，从源节点到目标节点传播信息
        print("1.初始化")
        self.lin = torch.nn.Linear(in_channels, out_channels)  # 线性变换

    def forward(self, x, edge_index):
        print("2.输入节点进行处理")
        # x shape [N, in_channels]
        # edge_index shape [2, E]  E 是 边数，边索引

        # 第一步：添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 第二步：线性变换，对节点进行线性变换，变换维度
        x = self.lin(x)  # 输入节点维度：1433，输出维度为：64

        # 第三步：使用度计算正则化
        row, col = edge_index  # 边的信息 第一行 第二行
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # 归一化系数

        # 第四步、第五步：开始启动 消息传递
        adjmat = SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(edge_index.shape[1]))
        # 如果 x 里面包含了 x_i(中心节点), x_j（邻居节点），那么message可以指定接收哪个属性的信息
        return self.propagate(adjmat, x=x, norm=norm, deg=deg.view((-1, 1)))

    def message(self, x_j, norm, deg_i):
        """
        重写 message
        """
        print("4.先对 massage进行处理")
        # x_j 就是每一个节点，norm：正则化结果, deg_i：度的信息 shape [E, 1]
        # message的消息从 propagate得来的
        # update使用默认的就可以了，所以没有进行修改
        # 对 x_j 乘以对应的正则化系数，就是说 每个邻居节点进行归一化，然后进行update
        return norm.view(-1, 1) * x_j * deg_i

    def aggregate(self, inputs, index, ptr, dim_size):
        """
        重写 aggregate
        这里 就是打印了一下信息，确认了这个被调用了，然后就没了一切都没有修改
        """
        print("5.聚合信息")
        print("self.aggr: ", self.aggr)
        print("aggregate is called")
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)

    def message_and_aggregate(self, adj_t, x, norm):
        print("3.如果这个实现了，那么会先调用这个")
        print('`message_and_aggregate` is called')
        # 只是打印消息，并没有什么具体的操作

    def update(self, inputs, deg):
        # 接收了 aggregate的 inputs，还接收了 propagate传过来的deg，也就是update可以随意接收
        print(deg)
        print("6.更新节点信息")
        return inputs


# 实际初始化 和 具体调用

dataset = Planetoid(root='dataset/Cora', name='Cora')  # 加载数据集
data = dataset[0]  # 获取图
# data.num_features：原始节点特征维度， 64是线性化变换之后的节点的特征维度
net = GCNConv(data.num_features, 64)  # 初始化网络
h_nodes = net(data.x, data.edge_index)  # 运行 图神经网络
# torch.Size([2708, 64])
print("6.结束")

"""
1.初始化
2.输入节点进行处理
3.如果这个实现了，那么会先调用这个
`message_and_aggregate` is called
6.结束

显然，只有 message_and_aggregate 这个被执行了，message和aggregate 虽然实现了，但是却没有执行，说明二选一
"""

dataset = Planetoid(root='dataset/Cora', name='Cora')  # 加载数据集
data = dataset[0]  # 获取图
# data.num_features：原始节点特征维度， 64是线性化变换之后的节点的特征维度
net = GCNConv2(data.num_features, 64)  # 初始化网络
h_nodes = net(data.x, data.edge_index)  # 运行 图神经网络
# torch.Size([2708, 64])
print("7.结束")

"""
1.初始化
2.输入节点进行处理
3.如果这个实现了，那么会先调用这个
`message_and_aggregate` is called
tensor([[4.],
        [4.],
        [6.],
        ...,
        [2.],
        [5.],
        [5.]])
6.更新节点信息
7.结束
"""