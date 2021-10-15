import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# transform: 将数据输入到神经网络之前修改数据，这一功能可用于 实现数据规范化或数据增强
# NormalizeFeatures 数据归一化
dataset = Planetoid(root='dataset/Cora', name='Cora', transform=NormalizeFeatures())
data = dataset[0]


class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(MLP, self).__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)  # 线性变换
        self.lin2 = Linear(hidden_channels, dataset.num_classes)  # 分类

    def forward(self, x):
        x = self.lin1(x)  # 线性变换
        x = x.relu()  # 激活函数
        x = F.dropout(x, p=0.5, training=self.training)  # dropout
        x = self.lin2(x)  # 线性变换，分类
        return x


model = MLP(hidden_channels=16)  # 定义模型
print(model)

# model = MLP(hidden_channels=16)
criterion = torch.nn.CrossEntropyLoss()  # 损失函数
# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.


# 训练

def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


for epoch in range(1, 201):  # 迭代
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("train ok!")


# 测试
def test():
    model.eval()
    out = model(data.x)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc


test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
