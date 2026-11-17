"""E(3)-invariant EGNN regressor for the HOMO-LUMO gap: the "proxy" used to score generated molecules.

Trained on the same train split as the generator and evaluated on the held-out test split. Ground-truth
validation of generated molecules is done separately with DFT (dft_eval.py).
"""
import os
import sys
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from tqdm import tqdm

from data import get_datasets, NUM_ATOM_TYPES
from graph_utils import get_fully_connected_edge_index
from model import EGNNLayer, GaussianExpansion


class GapRegressor(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=6, cutoff=15.0):
        super().__init__()
        self.node_emb = nn.Linear(NUM_ATOM_TYPES, hidden_dim)
        self.dist_expand = GaussianExpansion(0.0, cutoff, 50)
        # cond_dim=0: the layers are reused without any conditioning input
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim, hidden_dim, 0, 50) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x, h, batch, edge_index=None):
        if edge_index is None:
            edge_index = get_fully_connected_edge_index(batch)
        row, col = edge_index
        edge_attr = self.dist_expand(torch.norm(x[row] - x[col], dim=-1, keepdim=True))
        h = self.node_emb(h)
        cond = h.new_zeros(h.size(0), 0)
        for layer, norm in zip(self.layers, self.norms):
            _, dh = layer(x, norm(h), edge_index, edge_attr, cond)  # positions stay fixed
            h = h + dh
        return self.head(scatter(h, batch, dim=0, reduce='sum')).view(-1)


def load_regressor(path="checkpoints/regressor.pt", device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GapRegressor(**ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["y_mean"], ckpt["y_std"]


@torch.no_grad()
def predict_gaps(model, y_mean, y_std, mols, device, batch_size=512):
    """mols: list of (symbols, positions). Returns predicted gaps in eV (numpy)."""
    from chem import SYMBOLS
    idx = {s: i for i, s in enumerate(SYMBOLS)}
    preds = []
    for i in range(0, len(mols), batch_size):
        chunk = mols[i:i + batch_size]
        x = torch.cat([torch.as_tensor(p, dtype=torch.float32) for _, p in chunk]).to(device)
        types = torch.tensor([idx[s] for sym, _ in chunk for s in sym], device=device)
        h = F.one_hot(types, NUM_ATOM_TYPES).float()
        batch = torch.repeat_interleave(torch.arange(len(chunk), device=device),
                                        torch.tensor([len(s) for s, _ in chunk], device=device))
        preds.append(model(x, h, batch) * y_std + y_mean)
    return torch.cat(preds).cpu().numpy()


@torch.no_grad()
def evaluate_loader(model, loader, y_std, device):
    model.eval()
    err, n = 0.0, 0
    for b in loader:
        b = b.to(device)
        err += (model(b.pos, b.x, b.batch) - b.y).abs().sum().item() * y_std
        n += b.num_graphs
    return err / n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_batches", type=int, default=None, help="Limit batches per epoch (smoke tests)")
    p.add_argument("--out", type=str, default="checkpoints/regressor.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds, val_ds, test_ds, y_mean, y_std = get_datasets(seed=args.seed)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, 512, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, 512, num_workers=args.num_workers)

    hparams = {"hidden_dim": args.hidden_dim, "num_layers": args.num_layers}
    model = GapRegressor(**hparams).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for i, b in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", disable=not sys.stderr.isatty())):
            b = b.to(device)
            loss = F.l1_loss(model(b.pos, b.x, b.batch), b.y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            if args.max_batches and i + 1 >= args.max_batches:
                break
        sched.step()
        val_mae = evaluate_loader(model, val_loader, y_std, device)
        print(f"Epoch {epoch} | val MAE {val_mae:.4f} eV", flush=True)
        if val_mae < best:
            best = val_mae
            torch.save({"model": model.state_dict(), "hparams": hparams,
                        "y_mean": y_mean, "y_std": y_std}, args.out)

    model, _, _ = load_regressor(args.out, device)
    test_mae = evaluate_loader(model, test_loader, y_std, device)
    print(f"Best val MAE {best:.4f} eV | test MAE {test_mae:.4f} eV")
    with open(os.path.splitext(args.out)[0] + "_metrics.json", "w") as f:
        json.dump({"val_mae_ev": best, "test_mae_ev": test_mae}, f, indent=2)


if __name__ == "__main__":
    main()
