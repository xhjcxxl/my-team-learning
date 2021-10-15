# task2学习笔记

## 消息传递范式

消息传递范式包含三个步骤：

- 邻接节点信息变换(message)
- **邻接节点信息聚合到中心节点**(aggregate)
- **聚合信息变换**(update)

我记得之前看DGL的时候，也是用的消息传递范式，消息传递范式的好处在于，基本上GNN，GAT，GCN等大部分的图神经网络都可以使用消息传递范式来表示出来。所以，这个消息传递范式的功能非常强大。

## 构建消息传递的神经网络

PYG提供了一个MessagePassing 这个基类，然后在我们需要的神经网络上，需要继承这个基类，并且需要实现

- message() 方法（**传入节点，边，边传递给节点消息的方法，一般邻居节点和边的方法都会在这里进行处理好，然后后面只需要根据设定的方式更新节点就好了**）
- update() 方法（**对节点进行数据更新**）
- 消息聚合方法（就是怎么把**邻居节点和中心节点进行组合**，这个地方方法各种各样，也导致GNN方法各种各样）

而实际上，我们需要设定的是如下：`MessagePassing(aggr="add", flow="source_to_target", node_dim=-2)`

### MessagePassing基类

- `MessagePassing(aggr="add", flow="source_to_target", node_dim=-2)`（对象初始化方法）：
  - `aggr`：定义要使用的**聚合方案**（"add"、"mean "或 "max"）；
  - `flow`：定义**消息传递的流向**（"source_to_target "或 "target_to_source"）；
  - `node_dim`：**定义沿着哪个维度传播**，默认值为`-2`，也就是节点表征张量（Tensor）的哪一个维度是节点维度。节点表征张量`x`形状为`[num_nodes, num_features]`，其第0维度（也是第-2维度）是节点维度，其第1维度（也是第-1维度）是节点表征维度，所以我们可以设置`node_dim=-2`。
  - 注：`MessagePassing(……)`等同于`MessagePassing.__init__(……)`

- `MessagePassing.propagate(edge_index, size=None, **kwargs)`（调用 消息传递，启动消息传递）：
  - 开始传递消息的起始调用，在此方法中`message`、`update`等**方法被调用**。
  - 它以`edge_index`（边的端点的索引）和`flow`（消息的流向）以及一些额外的数据为参数。
  - 请注意，`propagate()`不局限于基于形状为`[N, N]`的对称邻接矩阵进行“消息传递过程”。基于非对称的邻接矩阵进行消息传递（当图为二部图时），需要传递参数`size=(N, M)`。
  - 如果设置`size=None`，则认为邻接矩阵是对称的。

- `MessagePassing.message(...)`：
  - 首先确定要给节点$i$传递消息的**边的集合**：
    - 如果`flow="source_to_target"`，则是$(j,i) \in \mathcal{E}$的边的集合；
    - 如果`flow="target_to_source"`，则是$(i,j) \in \mathcal{E}$的边的集合。
  - 接着为**各条边创建要传递给节点$i$的消息，即实现$\phi$函数**。
  - 我们用$i$表示“消息传递”中的中心节点，用$j$表示“消息传递”中的邻接节点。
  - **propagate 传入什么信息，这里就会收到什么信息，但是如果propagate收到的x节点带有 x_i 和 x_j属性，那么mesage可以直接指定接收 x_i，或者x_j节点， 因为propagate会自动把x的属性分开（这一点最重要）**

- `MessagePassing.aggregate(...)`：
  - 将从**源节点传递过来的消息聚合在目标节点**上，一般可选的聚合方式有`sum`, `mean`和`max`（就是 **邻居节点的数据如何和中心节点如何组合**）。
- `MessagePassing.message_and_aggregate(...)`：
  - 在一些场景里，邻接节点信息变换和邻接节点信息聚合这两项操作可以融合在一起，那么我们可以在此方法里定义这两项操作，从而让程序运行更加高效。
- `MessagePassing.update(aggr_out, ...)`: 
  - 为每个节点$i \in \mathcal{V}$更新节点表征，即实现$\gamma$函数。此方法以`aggregate`方法的输出为第一个参数，并接收所有传递给`propagate()`方法的参数。（**聚合之后，就需要更新中心节点的数据**）

### GCNConv练习

具体代码见：2.1、2.2

- 如果重写了 message_and_aggregate，传入propagate的是：SparseTensor，那么就会把 message_and_aggregate 这个被执行了，message和aggregate 虽然实现了，但是却没有执行，说明二选一

- 如果没有实现 message_and_aggregate，就会直接执行 message 和 aggregate

- 另外：update 接收了 aggregate的 inputs，还接收了 propagate传过来的deg，也就是 update 可以随意接收

## 作业

### MessagePassing基类的运行流程

流程有两种：

init(初始化) -> forward(调用函数，前期处理) -> propagate(开始消息传递) -> message_and_aggregate(实际消息传递实现) -> update(更新节点信息)

init(初始化) -> forward(调用函数，前期处理) -> propagate(开始消息传递) -> message(邻居节点消息处理) -> aggregate(邻居节点消息聚合) -> update(更新节点信息)

### 复现一个一层的图神经网络的构造

参考其他同学的方法，简单修改复现

```python
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
```

输出结果：

```python
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
6.结束
tensor([[ 0.0204,  0.0565, -0.0160,  ..., -0.0136, -0.0857,  0.0308],
        [-0.0374, -0.0480, -0.0352,  ..., -0.0295, -0.0927,  0.0255],
        [ 0.0006, -0.1815,  0.0774,  ...,  0.0522,  0.0284,  0.0457],
        ...,
        [-0.0004,  0.1197, -0.0931,  ...,  0.0609,  0.0027, -0.1604],
        [-0.0932,  0.0766, -0.0123,  ...,  0.1310,  0.0281, -0.0930],
        [-0.0972,  0.0849, -0.0777,  ...,  0.1255, -0.0044,  0.0330]],
       grad_fn=<AddBackward0>)
```