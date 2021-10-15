import os.path as osp
from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, Sequential, GCNConv
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

"""
# 加载数据集
dataset = Planetoid('dataset/Cora', 'Cora', transform=T.NormalizeFeatures())
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = data.y = None  # 不再有用
data = train_test_split_edges(data)

# print(data.edge_index.shape)
# torch.Size([2, 10556])

for key in data.keys:
    print(key, getattr(data, key).shape)

x torch.Size([2708, 1433])
val_pos_edge_index torch.Size([2, 263])
test_pos_edge_index torch.Size([2, 527])
train_pos_edge_index torch.Size([2, 8976])
train_neg_adj_mask torch.Size([2708, 2708])
val_neg_edge_index torch.Size([2, 263])
test_neg_edge_index torch.Size([2, 527])
"""


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, out_channels):
        super(Net, self).__init__()
        hns = [in_channels] + hidden_channels_list
        conv_list = []
        for idx in range(len(hidden_channels_list)):
            conv_list.append((GCNConv(hns[idx], hns[idx + 1]), 'x, edge_index -> x'))
            conv_list.append(nn.ReLU(inplace=True), )
        self.convseq = Sequential('x, edge_index', conv_list)
        self.linear = nn.Linear(hidden_channels_list[-1], out_channels)

    # 编码：对节点和边进行特征变换
    def encode(self, x, edge_index):
        x = self.convseq(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)
        return x

    # 解码：预测存在边 的概率
    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    # 解码：对所有的节点对预测存在边的几率
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


# 用于生成完整训练集的标签
def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train(data, model, optimizer):
    model.train()

    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))

    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index).to(data.x.device)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss


def test(data, model):
    model.eval()

    z = model.encode(data.x, data.train_pos_edge_index)

    results = []
    for prefix in ['val', 'test']:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        results.append(roc_auc_score(link_labels.cpu().detach().numpy(), link_probs.cpu().detach().numpy()))
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Planetoid('dataset/Cora', 'Cora', transform=T.NormalizeFeatures())
    data = dataset[0]
    ground_truth_edge_index = data.edge_index.to(device)
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)
    data = data.to(device)

    model = Net(dataset.num_features, hidden_channels_list=[200, 100], out_channels=32).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    best_val_auc = test_auc = 0
    for epoch in range(1, 101):
        loss = train(data, model, optimizer)
        val_auc, tmp_test_auc = test(data, model)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc = tmp_test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')

    z = model.encode(data.x, data.train_pos_edge_index)
    final_edge_index = model.decode_all(z)

"""原始的
Epoch: 096, Loss: 0.4488, Val: 0.9156, Test: 0.8989
Epoch: 097, Loss: 0.4492, Val: 0.9157, Test: 0.8992
Epoch: 098, Loss: 0.4383, Val: 0.9161, Test: 0.8996
Epoch: 099, Loss: 0.4431, Val: 0.9162, Test: 0.8994
Epoch: 100, Loss: 0.4483, Val: 0.9163, Test: 0.9001
"""

"""Sequential
Epoch: 096, Loss: 0.5483, Val: 0.7997, Test: 0.7351
Epoch: 097, Loss: 0.5374, Val: 0.8008, Test: 0.7351
Epoch: 098, Loss: 0.5407, Val: 0.8007, Test: 0.7351
Epoch: 099, Loss: 0.5474, Val: 0.7977, Test: 0.7351
Epoch: 100, Loss: 0.5467, Val: 0.7991, Test: 0.7351
"""

if __name__ == "__main__":
    main()
