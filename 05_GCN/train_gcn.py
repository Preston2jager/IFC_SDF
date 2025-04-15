import os, yaml, time, torch

import torch.nn.functional as F
from torch.utils.data import random_split
from dgl.dataloading import GraphDataLoader

import numpy as np

import model.model_gcn as model_gcn
from utils.dataset_gcn import GCNDataset

import config_files
import graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = os.path.join(os.path.dirname(graph.__file__),"best_model.pth")

class Trainer:
    def __init__(self, config):
        self.cfg = config
        self.model = model_gcn.GCN(self.cfg).to(device)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.cfg['lr_model'],
            weight_decay=5e-4
        )

    def __call__(self):
        train_loader, val_loader = self.get_loaders()
        start_time = time.time()
        best_val_loss = float('inf')

        for epoch in range(self.cfg['epochs']):
            train_loss = self.train_epoch(train_loader)
            val_loss   = self.val_epoch(val_loader)

            print(f"Epoch [{epoch+1}/{self.cfg['epochs']}], "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), weight_path)
                print(f"Best model saved at epoch {epoch+1} with Val Loss {val_loss:.4f}")

        print(f"Training completed in {time.time() - start_time:.2f} s")

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        loss_fn = torch.nn.MSELoss() 
        for batched_graph in loader:
            batched_graph = batched_graph.to(self.device)
            x = batched_graph.ndata['feat']           
            target = batched_graph.ndata['target']   

            x_masked = x.clone()
            x_masked[:, 512:] = 0

            out = self.model(batched_graph, x_masked)       
            loss = loss_fn(out, target)*10
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def val_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        loss_fn = torch.nn.MSELoss()  # 节点特征回归
        with torch.no_grad():  # 验证阶段禁用梯度，省内存，速度更快
            for batched_graph in loader:
                batched_graph = batched_graph.to(self.device)
                x = batched_graph.ndata['feat']
                x_masked = x.clone()
                x_masked[:, 512:] = 0
                out = self.model(batched_graph, x_masked)
                target = batched_graph.ndata['target']
                loss = loss_fn(out, target)
                total_loss += loss.item()
        return total_loss / len(loader)

    def get_loaders(self):
        dataset = GCNDataset()
        train_size = int(len(dataset) * 0.7)
        val_size = len(dataset) - train_size
        train_data, val_data = random_split(dataset, [train_size, val_size])
        train_loader = GraphDataLoader(
            train_data,
            batch_size=self.cfg['batch_size'],
            shuffle=True
        )
        val_loader = GraphDataLoader(
            val_data,
            batch_size=self.cfg['batch_size'],
            shuffle=False
        )
        return train_loader, val_loader


if __name__=='__main__':
    train_cfg_path = os.path.join(os.path.dirname(config_files.__file__), 'train_gcn.yaml')
    with open(train_cfg_path, 'rb') as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)
    trainer = Trainer(train_cfg)
    trainer() 
