import torch
def augment(x,edge_index,augs:list[str]):
    str2func={
        'edges':random_drop_edges,
        'nodes':random_drop_nodes,
        'attrs':random_drop_attrs
    }
    for aug in augs:
        return
    
def random_drop_edges(edge_index,pe=0.5):
    #如果是边矩阵，怎么删点？直接看row==i的置为false？
    p=torch.rand(edge_index.size(1),device=edge_index.device)#不能一个放在cpu，一个放在gpu
    mask=p>pe
    mask=mask.to(edge_index.device)
    return edge_index[:,mask]


def random_drop_nodes(num_nodes,edge_index,pn=0.5):
    p=torch.rand(num_nodes,device=edge_index.device)
    node_mask=p>pn#p>pn的不会被丢弃，小于就被丢了
    #不要再直接去找所有可能被删除的点的行和列了，试着根据两条边的点是否都在判断每条边是否被丢弃
    edge_mask=node_mask[edge_index[0]]&node_mask[edge_index[1]]#通过与操作判断边的两端都在来单独判断每条边是否都在
    return edge_index[:,edge_mask]


# def random_drop_nodes(num_nodes,edge_index,pn=0.5):
#     p=torch.rand(num_nodes)
#     drop_ids=torch.arange(num_nodes)[p>pn]
#     row,col=edge_index
#     mask_r=torch.isin(row,drop_ids)#因为有多个要去除的节点，会不能在直接用==来制造mask了，用isin
#     mask_c=torch.isin(col,drop_ids)
#     mask=mask_r|mask_c#行和列出现了一次就都要丢掉
#     mask=mask.to(edge_index.device)
#     return edge_index[:,~mask]

def random_drop_attrs(x,pa=0.5):
    p=torch.rand(x.size(1),device=x.device)#只用生成一个属性掩码，所有节点共用一个，不用每个节点都生成一个
    mask=p>pa#这里虽然生成了bool类型的mask，但不能直接用mask索引属性，因为false的属性会被1丢掉，而我们其实真正想做的是把false的变成0而不是丢掉
    z=torch.zeros_like(x)
    x=x.clone()#!!!其次，这里不能直接在x的基础上改，否则外面的x也会改，tensor是引用型变量，所以要先克隆！！
    x=torch.where(mask,x,z)
    return x

def random_sample_subgraph():#随机采样子图是每个节点采样一个？采样n个子图？
    return
if __name__=='__main__':
    edge_index=torch.tensor([[1,2,3,4,2,3,2,4,1,2,0,0,1],[1,3,0,1,0,1,3,2,2,4,3,0,2]])
    x=torch.rand((5,5))
    print(random_drop_edges(edge_index))
    print(random_drop_nodes(5,edge_index))
    print(x)
    print(random_drop_attrs(x))