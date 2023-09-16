from torch_geometric.datasets import WikiCS,Coauthor,Amazon,Planetoid
import  torch_geometric.transforms as T
import torch

def get_dataset(path='~/code/datasets',name='Coauthor-CS'):
    if name=='Cora':
        return Planetoid(root=path,name='Cora',transform=T.NormalizeFeatures())
    if name=='Coauthor-CS':
        return Coauthor(root=path,name='CS',transform=T.NormalizeFeatures())
    if name=='Coauthor-Phy':
        return Coauthor(root=path,name='physics',transform=T.NormalizeFeatures())
    if name=='Amazon-Computers':#767 1 13752
        return Amazon(root=path,name='computers',transform=T.NormalizeFeatures())
    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
    if name=='WikiCS':#wikiCS不需要name
        return WikiCS(root=path,transform=T.NormalizeFeatures())

def split_nodes(num_nodes,train_rate,val_rate):#划分训练节点和测试节点
    train_len=int(num_nodes*train_rate)
    val_len=int(num_nodes*val_rate)
    test_len=num_nodes-train_len-val_len

    train_mask=torch.zeros((num_nodes,)).to(torch.bool)
    val_mask=torch.zeros((num_nodes,)).to(torch.bool)
    test_mask=torch.zeros((num_nodes,)).to(torch.bool)

    idx=torch.randperm(num_nodes)
    train_mask[idx[:train_len]]=True
    val_mask[idx[train_len:train_len+val_len]]=True
    test_mask[idx[train_len+val_len:num_nodes]]=True
    return train_mask,val_mask,test_mask

if __name__=='__main__':
    dataset=get_dataset(name='Cora')
    print(dataset.num_features,dataset.len(),dataset[0].num_nodes)
    a,b,c=split_nodes(15,0.7,0.1)
    print(a)
    print(b)
    print(c)