"""Train the gap-conditioned flow-matching model (EGNN vector field + size predictor) on QM9."""
import os
import sys
import json
import argparse
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch_geometric.utils import scatter
from tqdm import tqdm

from data import get_dataloaders, NUM_ATOM_TYPES
from fm import CondFlowMatcher
from graph_utils import get_fully_connected_edge_index
from model import EGNNVectorField, SizePredictor


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--prior_scale", type=float, default=0.5)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--max_nodes", type=int, default=29)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_batches", type=int, default=None, help="Limit batches per epoch (smoke tests)")
    p.add_argument("--out_dir", type=str, default="results")
    return p.parse_args()


def size_nll(size_predictor, y, batch_idx):
    mu, log_var = size_predictor(y)
    sizes = scatter(torch.ones_like(batch_idx), batch_idx, dim=0, reduce='sum').view(-1, 1).float()
    return 0.5 * (log_var + (sizes - mu) ** 2 / torch.exp(log_var)).mean()


def compute_losses(model, size_predictor, fm, batch):
    x1, h1, y, batch_idx = batch.pos, batch.x, batch.y.view(-1, 1), batch.batch
    edge_index = get_fully_connected_edge_index(batch_idx)
    xt, ht, t, ux, uh = fm.sample_pt(x1, h1, batch_idx)
    vx, vh = model(xt, ht, t, y, batch_idx, edge_index)
    return F.mse_loss(vx, ux) + F.mse_loss(vh, uh), size_nll(size_predictor, y, batch_idx)


@torch.no_grad()
def validate(model, size_predictor, fm, loader, device):
    model.eval(); size_predictor.eval()
    flow, size = 0.0, 0.0
    # fixed noise / times so validation losses are comparable across epochs
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        torch.manual_seed(0)
        for batch in loader:
            lf, ls = compute_losses(model, size_predictor, fm, batch.to(device))
            flow += lf.item(); size += ls.item()
    return flow / len(loader), size / len(loader)


def plot_losses(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key, title in [(axes[0], "flow", "Flow-matching loss"), (axes[1], "size", "Size NLL")]:
        ax.plot(history[f"train_{key}"], label="train")
        ax.plot(history[f"val_{key}"], label="val (EMA)")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def train():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, y_mean, y_std = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = EGNNVectorField(in_node_dim=NUM_ATOM_TYPES, hidden_dim=args.hidden_dim,
                            num_layers=args.num_layers).to(device)
    size_predictor = SizePredictor(cond_dim=args.hidden_dim, max_nodes=args.max_nodes).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay))
    fm = CondFlowMatcher(prior_scale=args.prior_scale, n_atom_types=NUM_ATOM_TYPES)
    params = list(model.parameters()) + list(size_predictor.parameters())
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    print(f"Parameters: {sum(p.numel() for p in params) / 1e6:.2f}M")

    run_dir = os.path.join(args.out_dir, f"run_{datetime.now().strftime('%Y-%m-%d_%H%M')}")
    os.makedirs(run_dir, exist_ok=True)
    config = {**{k: v for k, v in vars(args).items() if k not in ("out_dir", "max_batches")},
              "in_node_dim": NUM_ATOM_TYPES, "learning_rate": args.lr,
              "y_mean": y_mean, "y_std": y_std}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print(f"Run directory: {run_dir}")

    history = {"train_flow": [], "train_size": [], "val_flow": [], "val_size": []}
    best_val = float("inf")
    show_bar = sys.stderr.isatty()

    def save(name):
        torch.save({'model': model.state_dict(), 'model_ema': ema.module.state_dict(),
                    'size_predictor': size_predictor.state_dict()}, os.path.join(run_dir, name))

    for epoch in range(1, args.epochs + 1):
        model.train(); size_predictor.train()
        e_flow, e_size, n_batches = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not show_bar)
        for batch in pbar:
            loss_flow, loss_size = compute_losses(model, size_predictor, fm, batch.to(device))
            optimizer.zero_grad()
            (loss_flow + loss_size).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            ema.update_parameters(model)

            e_flow += loss_flow.item(); e_size += loss_size.item(); n_batches += 1
            pbar.set_postfix(f=f"{loss_flow.item():.3f}", s=f"{loss_size.item():.3f}")
            if args.max_batches and n_batches >= args.max_batches:
                break

        history["train_flow"].append(e_flow / n_batches)
        history["train_size"].append(e_size / n_batches)
        v_flow, v_size = validate(ema.module, size_predictor, fm, val_loader, device)
        history["val_flow"].append(v_flow)
        history["val_size"].append(v_size)
        print(f"Epoch {epoch} | train flow {history['train_flow'][-1]:.4f} | "
              f"val flow (EMA) {v_flow:.4f} | val size {v_size:.4f}", flush=True)

        if v_flow < best_val:
            best_val = v_flow
            save("best.pt")
        save("last.pt")
        with open(os.path.join(run_dir, "losses.json"), "w") as f:
            json.dump(history, f, indent=4)

    plot_losses(history, os.path.join(run_dir, "losses.png"))
    print(f"Done. Best val flow loss {best_val:.4f}. Checkpoints in {run_dir}")


if __name__ == "__main__":
    train()
