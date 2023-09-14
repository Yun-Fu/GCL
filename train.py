from dataset import get_dataset,split_nodes
from torch_geometric.nn.conv import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Encoder
train_rate=0.8
val_rate=0.1
lr=0.01
epochs=100
act='relu'
conv='GCN'
if conv=='GCN':#g根据命令行参数自动选择模型
    conv=GCNConv
if act=='relu':
    act=nn.ReLU()#注意nn.functional.relu()是函数不是对象，所以就不能临时确定了？nn.Relu是一个类所以可以临时确定

#1.得到数据
dataset=get_dataset()
data=dataset[0]
#2.划分训练数据，验证集和测试集节点
train_mask,val_mask,test_mask=split_nodes(data.num_nodes,train_rate=train_rate,val_rate=val_rate)
#3.批处理？分批放进节点的话就必须使用graphsage的邻居采样才能保证能够生成n阶表示？直接用全图？

#4.定模型
model=Encoder()
#5.定训练步骤
optimizer=torch.optim.Adam(Encoder.parameters(),lr=lr)
def train(model):
    model.train()
    #6.定义增强方式,删边，删点，删属性，但如何混合增强呢？？
