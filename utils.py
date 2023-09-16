import torch
import torch.nn as nn
import torch.nn.functional as F
from model import logReg
from dataset import split_nodes
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GraphConv, GINConv

def log_regression(x,data,split,lr,epochs):
    test_device=x.device
    x=x.detach().to(test_device)#因为不是微调GNN的参数已经被固定住了，所以要把x从之前GNN的计算图拿下来，防止梯度回传到GNN造成多余计算
    y=data.y
    num_classes=y.max().item()+1
    model=logReg(x.size(1),num_classes).to(test_device)#模型里的参数注意也必须放进gpu
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    loss_fun=nn.CrossEntropyLoss()
    best_val_acc=0
    test_acc=0
    best_epoch=0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits=model(x[split['train']])
        loss=loss_fun(logits,y[split['train']])
        loss.backward()
        optimizer.step()

        train_acc=evaluate(logits,y[split['train']])
        #验证
        if (epoch+1)%20==0:#每20次验证一次
            model.eval()
            logits=model(x[split['val']])
            val_acc=evaluate(logits,y[split['val']])
            if best_val_acc<val_acc:
                best_val_acc=val_acc
                best_epoch=epoch
                logits=model(x[split['test']])
                test_acc=evaluate(logits,y[split['test']])
        #         print(f'Epoch: {epoch}, train: {train_acc:.4f}, val: {val_acc:.4f}, '
        #   f'test: {test_acc:.4f}')
    return test_acc

def evaluate(logits,y_true):
    y_pred=logits.argmax(-1)
    correct=(y_pred==y_true).sum().item()
    acc=correct/y_true.size(0)
    return acc

def get_base_model(name:str):
    def gat_wrapper(in_channels,out_channels):
        return GATConv(in_channels,out_channels//4,4)
    
    def gin_wrapper(in_channels,out_channels):
        mlp=nn.Sequential(nn.Linear(in_channels,out_channels),nn.ReLU(),nn.Linear(out_channels,out_channels))
        return GINConv(mlp)
    
    base_models={
        'GCNConv': GCNConv,
        'SGConv': SGConv,
        'SAGEConv': SAGEConv,
        'GATConv': gat_wrapper,
        'GraphConv': GraphConv,
        'GINConv': gin_wrapper
    }
    return base_models[name]

def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }

    return activations[name]
    