"""Generate 3D molecules with a target HOMO-LUMO gap from a trained run."""
import os
import glob
import json
import argparse

import torch

from chem import split_batch, mol_to_smiles, check_stability, write_xyz
from data import NUM_ATOM_TYPES
from fm import CondFlowMatcher
from model import EGNNVectorField, SizePredictor


def latest_run():
    runs = sorted(r for r in glob.glob("results/run_*") if
                  any(os.path.exists(os.path.join(r, c)) for c in ("best.pt", "nano_flow_mol.pt")))
    if runs:
        return runs[-1]
    if os.path.exists(os.path.join("checkpoints", "best.pt")):
        return "checkpoints"  # pretrained model shipped with the repo
    raise FileNotFoundError("No trained runs in results/ and no checkpoints/best.pt.")


def load_run(run_dir=None, device="cpu", use_ema=True, ckpt_name=None):
    """Load (model, size_predictor, flow_matcher, config) from a run directory.
    Supports new checkpoints (best.pt / last.pt with EMA weights) and the legacy nano_flow_mol.pt."""
    run_dir = run_dir or latest_run()
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    for name in ([ckpt_name] if ckpt_name else ["best.pt", "nano_flow_mol.pt"]):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            break
    ckpt = torch.load(path, map_location=device, weights_only=True)

    model = EGNNVectorField(in_node_dim=NUM_ATOM_TYPES, hidden_dim=config['hidden_dim'],
                            num_layers=config['num_layers']).to(device)
    size_predictor = SizePredictor(cond_dim=config['hidden_dim'], max_nodes=config['max_nodes']).to(device)
    model.load_state_dict(ckpt['model_ema'] if use_ema and 'model_ema' in ckpt else ckpt['model'])
    size_predictor.load_state_dict(ckpt['size_predictor'])
    model.eval(); size_predictor.eval()
    fm = CondFlowMatcher(prior_scale=config["prior_scale"], n_atom_types=NUM_ATOM_TYPES)
    config["run_dir"], config["checkpoint"] = run_dir, path
    return model, size_predictor, fm, config


@torch.no_grad()
def sample(model, size_predictor, fm, config, gaps_ev, device, steps=100, num_atoms=None,
           return_traj=False):
    """Generate one molecule per entry of gaps_ev (eV). Molecule sizes are drawn from the
    gap-conditioned size predictor unless num_atoms is given.
    Returns a list of (symbols, positions) and optionally the trajectory."""
    gaps = torch.as_tensor(gaps_ev, dtype=torch.float32, device=device).view(-1, 1)
    y = (gaps - config["y_mean"]) / config["y_std"]
    if num_atoms is None:
        mu, log_var = size_predictor(y)
        sizes = torch.normal(mu, torch.exp(0.5 * log_var)).round().long()
        num_atoms = sizes.clamp(3, config['max_nodes']).view(-1).tolist()
    elif isinstance(num_atoms, int):
        num_atoms = [num_atoms] * len(gaps)
    out = fm.generate(model, num_atoms, y, device, steps=steps, return_traj=return_traj)
    mols = split_batch(*out[:3])
    return (mols, out[3]) if return_traj else mols


def sample_many(model, size_predictor, fm, config, gaps_ev, device, batch_size=500, **kw):
    gaps_ev = list(gaps_ev)
    mols = []
    for i in range(0, len(gaps_ev), batch_size):
        mols += sample(model, size_predictor, fm, config, gaps_ev[i:i + batch_size], device, **kw)
    return mols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, default=None, help="Run directory (default: latest)")
    parser.add_argument("--gap", type=float, default=6.8, help="Target HOMO-LUMO gap in eV")
    parser.add_argument("--num_mols", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100, help="Euler integration steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="generated")
    parser.add_argument("--render", action="store_true", help="Render PNGs with xyzrender")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, size_predictor, fm, config = load_run(args.run_dir, device)
    print(f"Loaded {config['checkpoint']}")

    mols = sample(model, size_predictor, fm, config, [args.gap] * args.num_mols, device, steps=args.steps)
    os.makedirs(args.out_dir, exist_ok=True)
    n_valid = n_stable = 0
    for i, (symbols, pos) in enumerate(mols):
        smiles = mol_to_smiles(symbols, pos)
        stable = check_stability(symbols, pos)[0]
        n_valid += smiles is not None; n_stable += stable
        xyz_path = os.path.join(args.out_dir, f"mol_{i}.xyz")
        write_xyz(xyz_path, symbols, pos, comment=f"target gap {args.gap} eV | {smiles or 'invalid'}")
        print(f"mol_{i:<3d} atoms={len(symbols):<3d} stable={str(stable):<5s} smiles={smiles}")
        if args.render:
            from xyzrender import load, render
            render(load(xyz_path), output=os.path.join(args.out_dir, f"mol_{i}.png"))

    print(f"\nValid: {n_valid}/{len(mols)} | Stable: {n_stable}/{len(mols)} | saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
