class Data(object):

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, **kwargs):
        r"""
        Args:
            x (Tensor, optional): 节点属性矩阵，大小为\`[num_nodes, num_node_features]\`
            edge_index (LongTensor, optional): 边索引矩阵，大小为\`[2, num_edges]\`，第0行为尾节点，第1行为头节点，头指向尾
            edge_attr (Tensor, optional): 边属性矩阵，大小为\`[num_edges, num_edge_features]\`
            y (Tensor, optional): 节点或图的标签，任意大小（，其实也可以是边的标签）

        """
        self.x = x  # 节点
        self.edge_index = edge_index  # 边 索引矩阵
        self.edge_attr = edge_attr  # 边 属性矩阵
        self.y = y  # 边的 label

        for key, item in kwargs.items():
            if key == 'num_nodes':  # 节点个数
                self.__num_nodes__ = item  # 添加节点个数
            else:
                self[key] = item  # 设置其他属性

    @classmethod
    def from_dict(cls, dictionary):
        # 就是说，创建一个类方法，调用的是这个类
        data = cls()  # 实例化一个类，然后给类里面的数据进行赋值
        for key, item in dictionary.items():
            data[key] = item
        return data

    def to_dict(self):
        # 从data中，获取数据类型
        return {key: item for key, item in self}


# 实现
# 一般前面五个属性需要实现
graph = Data(x=x, edge_aindex=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_nodes, other_attr=other_attr)

# 将字典转为data数据
graph_dict = {
    'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr, 'y': y, 'num_nodes': num_nodes, 'other_attr': other_attr
}
graph_data = Data.from_dict(graph_dict)  # 直接调用就可以了，会在data类里面进行实现的
