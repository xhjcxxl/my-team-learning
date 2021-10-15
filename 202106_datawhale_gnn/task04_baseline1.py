import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Sequential, GCNConv
from torch_geometric.data import InMemoryDataset, download_url
import os.path as osp
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.io import read_planetoid_data


class PlanetoidPubMed(InMemoryDataset):
    r""" 节点代表文章，边代表引文关系。
   		 训练、验证和测试的划分通过二进制掩码给出。
    参数:
        root (string): 存储数据集的文件夹的路径
        transform (callable, optional): 数据转换函数，每一次获取数据时被调用。
        pre_transform (callable, optional): 数据转换函数，数据保存到文件前被调用。
    """
    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root, split="public", num_train_per_class=20,
                 num_val=500, num_test=1000, transform=None, pre_transform=None):

        super(PlanetoidPubMed, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.split = split
        assert self.split in ['public', 'full', 'random']

        if split == 'full':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])
        elif split == 'random':
            data = self.get(0)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True

            self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return ['ind.pubmed.{}'.format(name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, 'pubmed')
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels_list, num_classes):
        super(GAT, self).__init__()
        torch.manual_seed(123)
        hns = [num_features] + hidden_channels_list
        conv_list = []
        for idx in range(len(hidden_channels_list)):
            conv_list.append((GCNConv(hns[idx], hns[idx + 1]), 'x, edge_index -> x'))
            conv_list.append(nn.ReLU(inplace=True), )
        self.convseq = Sequential('x, edge_index', conv_list)
        self.linear = nn.Linear(hidden_channels_list[-1], num_classes)

    def forward(self, x, edge_index):
        x = self.convseq(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)
        return x


dataset = PlanetoidPubMed(root='dataset/PlanetoidPubMed/', transform=NormalizeFeatures())
print('dataset.num_features:', dataset.num_features)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)


def train():
    model.train()
    optimizer.zero_grad()  # 清除梯度
    out = model(data.x, data.edge_index)  # 输出结果
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # 计算loss
    loss.backward()  # loss回传
    optimizer.step()  # 梯度更新
    return loss


def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc

# hidden_channels_list: 第一个数据是：层数，第二个数据是：输出通道
model = GAT(num_features=dataset.num_features, hidden_channels_list=[100, 64], num_classes=dataset.num_classes).to(
    device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("train ok!")

print("testing...")
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
print("test ok!")

"""原始的
Epoch: 197, Loss: 0.0119
Epoch: 198, Loss: 0.0143
Epoch: 199, Loss: 0.0091
Epoch: 200, Loss: 0.0123
train ok!
testing...
Test Accuracy: 0.7430
test ok!
"""

"""GCNConv:[200, 100]
Epoch: 200, Loss: 0.0276
train ok!
testing...
Test Accuracy: 0.7790
test ok!
"""

"""GCNConv:[100, 100]
Epoch: 196, Loss: 0.0151
Epoch: 197, Loss: 0.0189
Epoch: 198, Loss: 0.0148
Epoch: 199, Loss: 0.0120
Epoch: 200, Loss: 0.0211
train ok!
testing...
Test Accuracy: 0.7860
test ok!
"""

"""CONConv:[100, 64]
Epoch: 197, Loss: 0.0152
Epoch: 198, Loss: 0.0191
Epoch: 199, Loss: 0.0228
Epoch: 200, Loss: 0.0192
train ok!
testing...
Test Accuracy: 0.7710
test ok!
"""
