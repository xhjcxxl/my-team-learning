# task1学习笔记

## 环境安装

```python
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-geometric
```

## 关于课后练习题

### 题目

请通过继承Data类实现一个类，专门用于表示“机构-作者-论文”的网络。该网络包含“机构“、”作者“和”论文”三类节点，以及“作者-机构“和“作者-论文“两类边。对要实现的类的要求：1）用不同的属性存储不同节点的属性；2）用不同的属性存储不同的边（边没有属性）；3）逐一实现获取不同节点数量的方法。

### 思考

根据题目，要实现一个继承类，这个没啥问题，但是后面说了有三类节点和两类边，其实我们只需要实现这个类就行了，并不需要传入数据把这个类实例化，说实话，最开始我并不知道在这里面的图的格式是怎么样的，通过队友的演示，知道了这里面输入的图的格式类型，看了队友的实例，明白了怎么去做。

其实，就是去定义一个类，然后这里类继承于data，然后需要把三种节点和两类边分开存放。

初始化的时候，包含了去定义三类节点和两类边，然后定义了一些函数，专门用于读取不同节点的数量的方法，这也就是三个问题。

### 总结

怎么说呢，刚开始没理解题意，确实没有怎么仔细思考，希望后面时间更多一点去仔细看看。