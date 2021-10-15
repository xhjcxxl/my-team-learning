from torch_geometric.datasets import Reddit
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler

dataset = Reddit('dataset/Reddit')
data = dataset[0]
print(dataset.num_classes)
print(data.num_nodes)
print(data.num_edges)
print(data.num_features)

"""
41
232965
114615873
602
"""
