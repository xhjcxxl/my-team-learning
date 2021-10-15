from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='dataset/Cora', name='Cora')

print(len(dataset))  # 数据集的图的个数
print(dataset.num_classes)  # 任务标签类型
print(dataset.num_node_features)  # 节点属性的维度

data = dataset[0]  # 获取具体数据 的图
print(data)  # 只有一个图，这个图有2708个节点

print(data.is_undirected())  # 是否是无向图

print(data.train_mask.sum().item())  # 训练级节点个数
print(data.val_mask.sum().item())  # 验证集节点个数
print(data.test_mask.sum().item())  # 测试集节点个数

# 实际使用
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
