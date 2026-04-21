import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.utils import scatter
from graph_utils import get_fully_connected_edge_index
from tqdm import tqdm

from data import get_dataloaders
from model import EGNNVectorField, SizePredictor
from fm import OCFM

# ==========================================
# HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 512    # Increased from 128
EPOCHS = 500
IN_NODE_DIM = 5
HIDDEN_DIM = 128
NUM_LAYERS = 6
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_NODES = 29
PRIOR_SCALE = 0.7
NUM_WORKERS = 4     # For multi-process data loading
# ==========================================

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data Setup (With multi-processing)
    train_loader, val_loader, y_mean, y_std = get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    # 2. Models
    model = EGNNVectorField(in_node_dim=IN_NODE_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
    size_predictor = SizePredictor(cond_dim=HIDDEN_DIM, max_nodes=MAX_NODES).to(device)
    fm = OCFM(prior_scale=PRIOR_SCALE)
    
    optimizer = AdamW(list(model.parameters()) + list(size_predictor.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    results_dir = f"results/run_{date_str}"
    os.makedirs(results_dir, exist_ok=True)
    
    history = {
        "train_flow": [], "train_size": [],
        "val_flow": [], "val_size": []
    }
    
    for epoch in range(1, EPOCHS + 1):
        model.train(); size_predictor.train()
        e_flow, e_size = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for batch in pbar:
            batch = batch.to(device)
            x1, h1, y, batch_idx = batch.pos, batch.x, batch.y.view(-1, 1), batch.batch
            
            edge_index = get_fully_connected_edge_index(batch_idx)

            # 1. Size loss (Regression NLL)
            mu, log_var = size_predictor(y)
            true_sizes = scatter(torch.ones_like(batch_idx), batch_idx, dim=0, reduce='sum').view(-1, 1).float()
            loss_size = 0.5 * (log_var + (true_sizes - mu)**2 / torch.exp(log_var)).mean()
            
            # 2. Flow loss
            xt, ht, t, ux, uh = fm.sample_pt(x1, h1, batch_idx)
            vx, vh = model(xt, ht, t, y, batch_idx, edge_index)
            
            loss_flow = F.mse_loss(vx, ux) + F.mse_loss(vh, uh)
            
            loss = loss_flow + loss_size
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            e_flow += loss_flow.item(); e_size += loss_size.item()
            pbar.set_postfix({'f': f"{loss_flow.item():.3f}", 's': f"{loss_size.item():.3f}"})
            
        history["train_flow"].append(e_flow/len(train_loader))
        history["train_size"].append(e_size/len(train_loader))
        
        # Validation
        model.eval(); size_predictor.eval()
        v_flow, v_size = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x1, h1, y, batch_idx = batch.pos, batch.x, batch.y.view(-1, 1), batch.batch
                edge_index = get_fully_connected_edge_index(batch_idx)
                
                mu, log_var = size_predictor(y)
                true_sizes = scatter(torch.ones_like(batch_idx), batch_idx, dim=0, reduce='sum').view(-1, 1).float()
                loss_size = 0.5 * (log_var + (true_sizes - mu)**2 / torch.exp(log_var)).mean()
                
                xt, ht, t, ux, uh = fm.sample_pt(x1, h1, batch_idx)
                vx, vh = model(xt, ht, t, y, batch_idx, edge_index)
                v_flow += (F.mse_loss(vx, ux) + F.mse_loss(vh, uh)).item()
                v_size += loss_size.item()
                
        history["val_flow"].append(v_flow/len(val_loader))
        history["val_size"].append(v_size/len(val_loader))
        
        print(f"Epoch {epoch} | Val Flow: {history['val_flow'][-1]:.4f} | Size: {history['val_size'][-1]:.4f}")

    torch.save({'model': model.state_dict(), 'size_predictor': size_predictor.state_dict()}, os.path.join(results_dir, "nano_flow_mol.pt"))
    
    config_data = {
        "in_node_dim": IN_NODE_DIM, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
        "batch_size": BATCH_SIZE, "epochs": EPOCHS, "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
        "max_nodes": MAX_NODES, "prior_scale": PRIOR_SCALE, "y_mean": y_mean.item(), "y_std": y_std.item()
    }
    json.dump(config_data, open(os.path.join(results_dir, "config.json"), "w"), indent=4)
    json.dump(history, open(os.path.join(results_dir, "losses.json"), "w"), indent=4)

if __name__ == "__main__":
    train()
