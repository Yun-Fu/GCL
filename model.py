import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, conv, k, act) -> None:
        super().__init__()
        self.k=k
        self.convs=nn.ModuleList()
        self.convs.append(conv(in_channels,hidden_channels))
        for i in range(k-1):
            self.convs.append(conv(hidden_channels,hidden_channels))
        self.convs.append(conv(hidden_channels,out_channels))

        self.act=act

    def  forward(self,x,edge_index):
        for i in range(self.k):
            x=self.convs[i](x,edge_index)
            x=self.act(x)
        return x
    
#定对比学习框架
class GRACE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, conv, k, act) -> None:
        super().__init__()
        self.enc=Encoder(in_channels, hidden_channels, out_channels, conv, k, act)

    