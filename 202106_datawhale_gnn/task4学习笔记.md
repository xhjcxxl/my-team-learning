# task学习笔记

## 学习任务1

先学习了 数据集处理的构造类，通过继承一个内部数据集类，来学习怎么对数据进行处理，包括

- 下载数据
- 数据处理（数据过滤，数据处理）
- 保存数据（返回结果）

## 学习任务2

利用之前学习的数据集，来实现 节点分类预测，和边分类预测

- 节点分类预测

就是说，使用GAT或者GCN，使用两层或者多层卷积，构建一个网络，然后对节点进行分类

同时，也学习了一个序列容器，能够批量生成多个层，然后让模型按照顺序进行构建模型，并在网络中进行训练，来达到深层模型的目的，一个一个的写太慢了，重复度够高的话，就可以用这个序列容器来批量编写多个层结构

- 边分类预测

就是去预测两个节点之间是否存在边

这里的话，一个重要的地方就是，由于我们的数据集里面只存在 正样本，也就是说，所有的样本中，都是有边的，那么我们需要构建负样本，构建一些样本的节点是没有边的，这样才能正负样本都具有，才能更好的训练模型。

有一个现成的 采样负样本边的方法，train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1)

- 第一个参数为torch_geometric.data.Data对象，
- 第二参数为验证集所占比例，
- 第三个参数为测试集所占比例

会自动地采样得到负样本，并将正负样本分成训练集、验证集和测试集三个集合。

所以非常方便，不需要我们自己去写了。

## 作业

- 作业1

就是把GAT网络换成GCN等其他卷积层，并且调整 序列网络的层数，来帮助模型更好的训练，具体的代码在 `task04baseline1`

部分结果如下：

```python
# 原始的
Epoch: 197, Loss: 0.0119
Epoch: 198, Loss: 0.0143
Epoch: 199, Loss: 0.0091
Epoch: 200, Loss: 0.0123
train ok!
testing...
Test Accuracy: 0.7430
test ok!

# GCNConv:[200, 100]
Epoch: 200, Loss: 0.0276
train ok!
testing...
Test Accuracy: 0.7790
test ok!

# GCNConv:[100, 100]
Epoch: 196, Loss: 0.0151
Epoch: 197, Loss: 0.0189
Epoch: 198, Loss: 0.0148
Epoch: 199, Loss: 0.0120
Epoch: 200, Loss: 0.0211
train ok!
testing...
Test Accuracy: 0.7860
test ok!

# CONConv:[100, 64]
Epoch: 197, Loss: 0.0152
Epoch: 198, Loss: 0.0191
Epoch: 199, Loss: 0.0228
Epoch: 200, Loss: 0.0192
train ok!
testing...
Test Accuracy: 0.7710
test ok!
```

- 作业2

将 Sequential 用到 边预测任务中，然后调整

```
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
```

- 作业3

个人认为：验证集和测试集里面也需要负样本，如果不进行采样的话，可能会存在偏差
