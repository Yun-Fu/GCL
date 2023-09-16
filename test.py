import torch_geometric.transforms as T
from torch_geometric.datasets import Coauthor
import os.path as osp
import torch.nn.functional as F
import torch
a=torch.rand((4,7))
b=a.diag()
print(b)
exit()
def my_function(**kwargs):
    if 'a' in kwargs:
        print("a:", kwargs['a'])
    if 'b' in kwargs:
        print("b:", kwargs['b'])

my_function(a=1, b=2, c=3)

def my_function(*args, **kwargs):
    for arg in args:#*表示按照元组存参数值
        print(arg)
    for key, value in kwargs.items():
        print(key, value)
    print(kwargs['a'],kwargs['b'])#**表示按照字典存参数名称和参数值

my_function(1, 2, 3, a=4, b=5)

exit()
a=F.relu
b=torch.tensor([-1,2,-1,-3,23])
print(a(b))
exit()
 
path='./'
name='cs'
dataset=Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
dataset=Coauthor(root=path,name=name)
data=dataset[0]
print(data.x[1][data.x[1].nonzero().squeeze(-1)])
print(data.x[1].nonzero().size(0))
print(data.x[2].nonzero().size(0))
print(data.x[1][data.x[1].argmax()])
print(data.x[2][data.x[2].argmax()])
print(data.x[1].max())
print(data.x[2].max())