# 学习笔记

## 创建超大规模的数据集类

有些数据集无法直接装进内存，所以需要一个特别的数据集来处理这些样本，能够实现按需加载样本到内存。

可以通过继承 torch_geometric.data.Dataset，然后实现基类的方法，以及另外的方法：

- len():返回数据集中样本的个数
- get():实现加载单个图的操作

## 图样本封装成 batch 和 dataloader

### 合并小图成大图

pyg中，通过将小图作为连通组件的形式进行合并，构建一个大图，好处在于：

- 依靠消息传递方案的GNN运算不需要被修改，因为消息仍然不能在属于不同图的两个节点之间交换

- 没有额外的计算或内存的开销

### 小图的属性增值与拼接

将小图存储到大图中时需要对小图的属性做一些修改，一个最显著的例子就是要对节点序号增值

#### 图的匹配

如果要在一个data对象中存储多个图，用于图匹配，然后需要需要确保这些图可以正确封装成batch，比如将一个源图$G_s$ 和一个目标图$G_t$
，存储在一个Data类中:

```python
class PairData(Data):
    def __init__(self, edge_index_s, x_s, edge_index_t, x_t):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
```

这时，edge_index_s 会自动根据 源图$G_s$的节点数做增值，也会对 目标图做节点增值。

```python
class PairData(Data):
    def __init__(self, edge_index_s, x_s, edge_index_t, x_t):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t

    def __inc__(self, key, value):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value)
```

可以使用脚本进行测试：

```python
edge_index_s = torch.tensor([
    [0, 0, 0, 0],
    [1, 2, 3, 4],
])
x_s = torch.randn(5, 16)  # 5 nodes.
edge_index_t = torch.tensor([
    [0, 0, 0],
    [1, 2, 3],
])
x_t = torch.randn(4, 16)  # 4 nodes.

data = PairData(edge_index_s, x_s, edge_index_t, x_t)
data_list = [data, data]
loader = DataLoader(data_list, batch_size=2)
batch = next(iter(loader))

print(batch)
# Batch(edge_index_s=[2, 8], x_s=[10, 16], edge_index_t=[2, 6], x_t=[8, 16])

print(batch.edge_index_s)
# tensor([[0, 0, 0, 0, 5, 5, 5, 5], [1, 2, 3, 4, 6, 7, 8, 9]])

print(batch.edge_index_t)
# tensor([[0, 0, 0, 4, 4, 4], [1, 2, 3, 5, 6, 7]])
```

#### 二部图

二部图的邻接矩阵定义两种类型的节点之间的连接关系。对二部图的封装成批过程中，edge_index 中边的源节点与目标节点做的增值操作应是不同的。我们将二部图中两类节点的特征特征张量分别存储为$x_s$和$x_t$。

```python
class BipartiteData(Data):
    def __init__(self, edge_index, x_s, x_t):
        super(BipartiteData, self).__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
```

如果要让二部图实现封装成batch，需要在pyg中进行设置：

```python
def __inc__(self, key, value):
    if key == 'edge_index':
        return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
    else:
        return super().__inc__(key, value)
```

测试：

```python
edge_index = torch.tensor([
    [0, 0, 1, 1],
    [0, 1, 1, 2],
])
x_s = torch.randn(2, 16)  # 2 nodes.
x_t = torch.randn(3, 16)  # 3 nodes.

data = BipartiteData(edge_index, x_s, x_t)
data_list = [data, data]
loader = DataLoader(data_list, batch_size=2)
batch = next(iter(loader))

print(batch)
# Batch(edge_index=[2, 8], x_s=[4, 16], x_t=[6, 16])

print(batch.edge_index)
# tensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 2, 3, 4, 4, 5]])
```

#### 在新的维度上做拼接

如果Data对象的属性需要在一个新的维度上做拼接，比如封装成batch，就需要在`__cat_dim__()`中进行修改：

```python
class MyData(Data):
    def __cat_dim__(self, key, item):
        if key == 'foo':
            return None
        else:
            return super().__cat_dim__(key, item)

edge_index = torch.tensor([
   [0, 1, 1, 2],
   [1, 0, 2, 1],
])
foo = torch.randn(16)

data = MyData(edge_index=edge_index, foo=foo)
data_list = [data, data]
loader = DataLoader(data_list, batch_size=2)
batch = next(iter(loader))

print(batch)
# Batch(edge_index=[2, 8], foo=[2, 16])
```

### 创建超大规模数据集类实践

对数据 PCQM4M-LSC，构建数据集类，PCQM4M-LSC是一个分子图的量子特性回归数据集；

```python
import os
import os.path as osp

import pandas as pd
import torch
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import download_url, extract_zip
from rdkit import RDLogger
from torch_geometric.data import Data, Dataset
import shutil

RDLogger.DisableLog('rdApp.*')

class MyPCQM4MDataset(Dataset):

    def __init__(self, root):
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m_kddcup2021.zip'
        super(MyPCQM4MDataset, self).__init__(root)

        filepath = osp.join(root, 'raw/data.csv.gz')
        data_df = pd.read_csv(filepath)
        self.smiles_list = data_df['smiles']
        self.homolumogap_list = data_df['homolumogap']

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.move(osp.join(self.root, 'pcqm4m_kddcup2021/raw/data.csv.gz'), osp.join(self.root, 'raw/data.csv.gz'))

    def len(self):
        return len(self.smiles_list)

    def get(self, idx):
        smiles, homolumogap = self.smiles_list[idx], self.homolumogap_list[idx]
        graph = smiles2graph(smiles)
        assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert(len(graph['node_feat']) == graph['num_nodes'])

        x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        y = torch.Tensor([homolumogap])
        num_nodes = int(graph['num_nodes'])
        data = Data(x, edge_index, edge_attr, y, num_nodes=num_nodes)
        return data

    # 获取数据集划分
    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'pcqm4m_kddcup2021/split_dict.pt')))
        return split_dict

if __name__ == "__main__":
    dataset = MyPCQM4MDataset('dataset2')
    from torch_geometric.data import DataLoader
    from tqdm import tqdm
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    for batch in tqdm(dataloader):
        pass
```

## 图预测任务实践

基于GIN的图表示学习神经网络，和自定义的数据集来实现分子图的量子性质预测任务;

执行代码为:

```python
#!/bin/sh

python main.py  --task_name GINGraphPooling\    # 为当前试验取名
                --device 0\                     
                --num_layers 5\                 # 使用GINConv层数
                --graph_pooling sum\            # 图读出方法
                --emb_dim 256\                  # 节点嵌入维度
                --drop_ratio 0.\
                --save_test\                    # 是否对测试集做预测并保留预测结果
                --batch_size 512\
                --epochs 100\
                --weight_decay 0.00001\
                --early_stop 10\                # 当有`early_stop`个epoches验证集结果没有提升，则停止训练
                --num_workers 4\
                --dataset_root dataset          # 存放数据集的根目录
```

通过修改不同的参数进行执行，来实现测试不同的超参数

## 总结

还是学习怎么构建超大数据集，通过继承已有的基础数据集来实现。

只能看着， 实验室的服务器连不上，只能看着pdf进行学习，希望后面能够进行测试
