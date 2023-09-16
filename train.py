from dataset import get_dataset,split_nodes
from torch_geometric.nn.conv import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Encoder,GRACE
from utils import get_activation,get_base_model,log_regression
from augment import random_drop_attrs,random_drop_edges,random_drop_nodes
import random
import json
import argparse
import yaml
from yaml import SafeLoader
def train(model,data):
    model.train()
    optimizer.zero_grad()
    x,edge_index=data.x,data.edge_index
    #6.定义增强方式,删边，删点，删属性，但如何混合增强呢？？
    x1=random_drop_attrs(x,args.)
    x1,e1 = data.x,random_drop_edges(data.edge_index)
    x2,e2 = random_drop_attrs(data.x),data.edge_index

    z1 = model(x1, e1)
    z2 = model(x2, e2)

    loss = model.loss(z1, z2)
    loss.backward()
    optimizer.step()

    return loss.item()

def test(model,data,split,lr,epochs):
    model.eval()
    h = model(data.x, data.edge_index)
    acc = log_regression(h, data,split,lr,epochs)
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--param', type=str, default='local:wikics.json')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    parser.add_argument('--base_model', type=str, default='GCNConv')
    parser.add_argument('--num_layers', type=str, default=2)
    parser.add_argument('--weight_decay', type=str, default=1e-5)
    # args = parser.parse_args()

    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }
    # 读取 JSON 配置文件
    with open('config/Cora.json', "r") as f:
        default_param = json.load(f)

    # config=yaml.load(open(args.config),Loader=SafeLoader)[args.dataset]
    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), default=default_param[key])
    args = parser.parse_args()

    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)

    device = torch.device(args.device)
    #1.得到数据
    dataset=get_dataset()
    data=dataset[0].to(device)
    #2.划分训练数据，验证集和测试集节点
    train_mask,val_mask,test_mask=split_nodes(data.num_nodes,train_rate=0.8,val_rate=0.1)
    split={'train':train_mask,'val':val_mask,'test':test_mask}
    #3.批处理？分批放进节点的话就必须使用graphsage的邻居采样才能保证能够生成n阶表示？直接用全图？

    #4.定模型
    encoder = Encoder(dataset.num_features, args.num_hidden, get_activation(args.activation),
                      conv=get_base_model(args.base_model), k=args.num_layers).to(device)
    model = GRACE(encoder, args.num_hidden, args.num_proj_hidden, args.tau).to(device)
    #5.定训练步骤
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)


    for epoch in range(1, args.num_epochs + 1):
        loss = train(model,data)
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
        acc = test(model,data,split,args.learning_rate,args.num_epochs)
        print(f'(E) | Epoch={epoch:04d}, acc = {acc}')

    acc = test(model,data,split,args.learning_rate,args.num_epochs)
    print(f'{acc}')   
#0.5  0.5
#CS 0.7644492911668485
#Cora 0.4792802617230098