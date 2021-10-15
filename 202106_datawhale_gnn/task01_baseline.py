from torch_geometric.data.data import Data


class mydata(Data):

    def __init__(self, org_node=None, author_node=None, paper_node=None,
                 org_aut_edge_index=None, aut_paper_edge_index=None, **kwargs):
        """
        三类节点：
            org_node：节点属性矩阵，维度[num_node, num_node_features]
            author_node：节点属性矩阵，维度[num_node, num_node_features]
            paper_node：节点属性矩阵，维度[num_node, num_node_features]
        两类边：
            org_aut_edge_index：边索引矩阵，维度[2, num_edges], 第0⾏为尾节点，第1⾏为头节点，头指向尾
            aut_paper_edge_index：边索引矩阵，维度[2, num_edges], 第0⾏为尾节点，第1⾏为头节点，头指向尾
        """
        # 设置参数
        self.org_node = org_node
        self.author_node = author_node
        self.paper_node = paper_node

        # 设置边
        self.org_aut_edges = org_aut_edge_index
        self.aut_paper_edges = aut_paper_edge_index

        # 对其他键值对进行处理
        for key, values in kwargs.items():
            if key == "num_nodes":
                self.__num_nodes__ = values
            else:
                self[key] = values

    @property
    def num_org_nodes(self):
        if self.org_node is None:
            return 0
        else:
            return self.org_node.shape[0]

    @property
    def num_author_nodes(self):
        if self.author_node is None:
            return 0
        else:
            return self.author_node.shape[0]

    @property
    def num_paper_nodes(self):
        if self.paper_node is None:
            return 0
        else:
            return self.paper_node.shape[0]
