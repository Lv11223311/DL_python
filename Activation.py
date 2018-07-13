import numpy as np 


# sigmoid：将所有节点映射到（0,1），可以做二分类
# 应用于特征相差比较复杂或者是相差不是特别打的时候效果比较好。
# sigmoid的倒数最大值为0.25.这意味着用来进行方向传播时，返回网络的error将会在每一层收缩至少75%，若有很多层，权重更新会很小
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

# Tanh: 又称双正切函数，取值为[-1, 1]。
# 在特征相差明显时效果比较好，在循环过程中会不断扩大特征效果
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Relu:现较常用的激活函数， 将小于0的输入归为0,大于0者等于本身
# 优点为计算量小，一部分神经元为0造成了网络的稀疏性， 并且减少了参数的相互依存关系，缓解了过拟合问题的发生。
# 缺点为梯度较大时可能大多节点为0, 产生大量无效计算，导致模型无法学到有效特征
def relu(x):
    return np.maximum(0, x)

# Softmax:将所有节点映射到（0, 1）， 可以做多分类
# 将比sigmoid，做了归一化处理
def softmax(x):
    expx = np.exp(x)
    return [i/expx for i in expx]



