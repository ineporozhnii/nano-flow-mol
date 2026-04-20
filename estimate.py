import os
import time
import torch
import torch.nn.functional as F
from graph_utils import get_fully_connected_edge_index
from tqdm import tqdm

from data import get_dataloaders
from model import EGNNVectorField
from fm import OCFM

def estimate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data Setup
    train_loader, val_loader, _, _ = get_dataloaders(batch_size=128)
    print(f"Train batches: {len(train_loader)}")
    
    # 2. Model & Flow Matching
    model = EGNNVectorField(in_node_dim=5, hidden_dim=64, num_layers=6).to(device)
    fm = OCFM()
    
    epochs = 10
    
    model.train()
    
    print("Warming up...")
    batch_iter = iter(train_loader)
    
    # Warmup
    for _ in range(5):
        batch = next(batch_iter).to(device)
        x1, h1, y, batch_idx = batch.pos, batch.x, batch.y.view(-1, 1), batch.batch
        edge_index = get_fully_connected_edge_index(batch_idx)
        xt, ht, t, ux, uh = fm.sample_pt(x1, h1, batch_idx)
        vx, vh = model(xt, ht, t, y, batch_idx, edge_index)
        loss = F.mse_loss(vx, ux) + F.mse_loss(vh, uh)
        loss.backward()
    
    print("Measuring time for 50 batches...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(50):
        batch = next(batch_iter).to(device)
        x1, h1, y, batch_idx = batch.pos, batch.x, batch.y.view(-1, 1), batch.batch
        edge_index = get_fully_connected_edge_index(batch_idx)
        xt, ht, t, ux, uh = fm.sample_pt(x1, h1, batch_idx)
        vx, vh = model(xt, ht, t, y, batch_idx, edge_index)
        loss = F.mse_loss(vx, ux) + F.mse_loss(vh, uh)
        loss.backward()
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    time_per_batch = (end_time - start_time) / 50
    time_per_epoch = time_per_batch * len(train_loader)
    total_time = time_per_epoch * epochs
    
    print(f"Estimated time per batch: {time_per_batch:.4f} seconds")
    print(f"Estimated time per epoch: {time_per_epoch:.2f} seconds ({time_per_epoch/60:.2f} minutes)")
    print(f"Estimated total training time for {epochs} epochs: {total_time:.2f} seconds ({total_time/60:.2f} minutes or {total_time/3600:.2f} hours)")

if __name__ == "__main__":
    estimate()