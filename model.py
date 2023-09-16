import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, act, conv, k) -> None:
        super().__init__()
        self.k=k
        self.convs=nn.ModuleList()
        self.convs.append(conv(in_channels,out_channels))
        for i in range(k-1):
            self.convs.append(conv(out_channels,out_channels))
        self.convs.append(conv(out_channels,out_channels))

        self.act=act

    def  forward(self,x,edge_index):
        for i in range(self.k):
            x=self.convs[i](x,edge_index)
            x=self.act(x)
        return x
    
#定对比学习框架
class GRACE(nn.Module):
    def __init__(self, encoder:Encoder, hidden_channels, project_channels, t) -> None:
        super().__init__()
        self.t=t#温度一般设置为多少？？为什么需要温度除相似度分母？？
        self.encoder=encoder
        self.fc1=nn.Linear(hidden_channels,project_channels)#为什么分两个？一个专门针对丢边，一个针对丢属性？
        self.fc2=nn.Linear(project_channels,hidden_channels)#共用的两层转换头，不是每个view对应一个，两个view共享的
        #其次，节点表示经过encoder就可以得到h了，这里只是为了对比才要经历两层mlp得到z！

    def forward(self,x,e):
        h=self.encoder(x,e)
        return h
    
    def projection(self,h):
        h=F.elu(self.fc1(h))
        return self.fc2(h)
    
    def sim(self,z1,z2):#计算余弦相似度
        z1=F.normalize(z1)#因为要计算余弦相似度，所以这里归一化？
        z2=F.normalize(z2)#为什么一定要normalize？
        return torch.mm(z1,z2.T)
        
    def semi_loss(self,z1,z2):
        f=lambda x:torch.exp(x/self.t)
        inter_sim=f(self.sim(z1,z1))
        intra_sim=f(self.sim(z1,z2))
        return -torch.log(intra_sim.diag()/(intra_sim.sum(-1)+inter_sim.sum(-1)-inter_sim.diag()))#这是一张图所有节点的对比损失,还要取平均才是整张图的损失
    
    def loss(self,h1,h2):
        z1=self.projection(h1)#fc1，fc2虽然没在编码1模型中，但却使用fc1,fc2计算的损失，所以后向传播时fc1，fc2也会同时被更新
        z2=self.projection(h2)

        l1=self.semi_loss(z1,z2)
        l2=self.semi_loss(z2,z1)#计算(u,v)和(v,u)双向损失
        ret=(l1+l2)*0.5
        return ret.mean()
    
class logReg(nn.Module):
    def __init__(self, in_channels,out_channels) -> None:
        super().__init__()
        self.fc=nn.Linear(in_channels,out_channels) 
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.fc.weight)#为什么一定要初始化？
        nn.init.zeros_(self.fc.bias)

    def forward(self,x):
        return self.fc(x)#直接一个线性层返回出去，没有再加非线性函数了，外面直接用的softmax激活