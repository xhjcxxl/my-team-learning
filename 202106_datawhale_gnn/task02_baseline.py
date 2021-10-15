import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import Planetoid
from torch_sparse import SparseTensor


class GCNConv(MessagePassing):
    """
    对 update 进行 重写
    """
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='mean', flow='source_to_target')
        # Add 是 聚合操作
        # flow 是 流向，从源节点到目标节点传播信息
        print("1.初始化")
        self.lin = torch.nn.Linear(in_channels, out_channels)  # 线性变换
        self.tanh = torch.nn.Tanh()  # 激活函数

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

    def message_and_aggregate(self, adj_t, x, norm, deg):
        print("3.如果这个实现了，那么会先调用这个")
        print('`message_and_aggregate` is called')
        # 参考别人的
        adj_t = adj_t.to_dense()
        N = len(adj_t)
        out = []
        x0 = x[:]
        for i in range(N):
            # 计算每个 xi 的neighbor传过来的信息的平均值
            x_sum = torch.matmul(x.T, adj_t[i])
            x_avg = x_sum / deg[i]
            out.append(x_avg)
        out = torch.stack(out)
        return [out, x0]

    def update(self, inputs, deg):
        # 接收了 aggregate的 inputs，还接收了 propagate传过来的deg，也就是update可以随意接收
        print(deg)
        print("6.更新节点信息")
        x0 = inputs[1]  # 获取 中心节点
        output = self.tanh(inputs[0]) + x0  # 使用激活函数，更新数据，更新节点信息
        return output


# 实际初始化 和 具体调用
dataset = Planetoid(root='dataset/Cora', name='Cora')  # 加载数据集
data = dataset[0]  # 获取图
# data.num_features：原始节点特征维度， 64是线性化变换之后的节点的特征维度
net = GCNConv(data.num_features, 64)  # 初始化网络
h_nodes = net(data.x, data.edge_index)  # 运行 图神经网络
# torch.Size([2708, 64])
print("6.结束")
print(h_nodes)
