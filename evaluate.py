"""Evaluate a trained run.

1. Sample quality (EDM protocol): atom/molecule stability, validity, uniqueness, novelty.
2. Distributions vs QM9: molecule size, atom types, bond lengths.
3. Conditioning accuracy with the proxy regressor: MAE between the target gap and the predicted gap of the
   generated molecule, compared with the EDM baselines (shuffled targets, #atoms-only, regressor error on QM9).
"""
import os
import json
import argparse
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

import plot_style
from plot_style import plt
from chem import SYMBOLS, check_stability, mol_to_smiles, canonical, bond_matrix
from data import get_datasets
from generate import load_run, sample_many


def train_smiles(train_ds, cache="data/qm9/train_smiles_seed42.txt"):
    if os.path.exists(cache):
        with open(cache) as f:
            return set(f.read().split())
    smiles = {canonical(s) for s in tqdm(train_ds.smiles, desc="Canonicalizing train SMILES")}
    smiles.discard(None)
    with open(cache, "w") as f:
        f.write("\n".join(sorted(smiles)))
    return smiles


def dataset_mols(ds, n, seed=0):
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n]
    out = []
    for i in idx.tolist():
        d = ds[i]
        out.append(([SYMBOLS[t] for t in d.x.argmax(-1).tolist()], d.pos.numpy()))
    return out


def quality_metrics(mols, known_smiles):
    stable_mols = stable_atoms = n_atoms = 0
    smiles = []
    for symbols, pos in mols:
        ms, sa, na = check_stability(symbols, pos)
        stable_mols += ms; stable_atoms += sa; n_atoms += na
        smiles.append(mol_to_smiles(symbols, pos))
    valid = [s for s in smiles if s is not None]
    unique = set(valid)
    novel = unique - known_smiles
    n = len(mols)
    return {
        "n_samples": n,
        "atom_stability": stable_atoms / n_atoms,
        "mol_stability": stable_mols / n,
        "validity": len(valid) / n,
        "uniqueness": len(unique) / max(len(valid), 1),          # among valid
        "novelty": len(novel) / max(len(unique), 1),             # among valid & unique
        "valid_x_unique": len(unique) / n,
    }, smiles


def bond_lengths(mols, pairs):
    out = {p: [] for p in pairs}
    for symbols, pos in mols:
        b = bond_matrix(symbols, pos)
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        for i, j in zip(*np.nonzero(np.triu(b))):
            key = tuple(sorted((symbols[i], symbols[j]), key="CNOHF".index))
            if key in out:
                out[key].append(d[i, j])
    return out


def plot_distributions(gen, ref, path):
    pairs = [("C", "C"), ("C", "H"), ("C", "O"), ("C", "N")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    gen_c, ref_c = plot_style.CATEGORICAL[0], plot_style.QM9_COLOR

    sizes = [np.array([len(s) for s, _ in m]) for m in (ref, gen)]
    bins = np.arange(2.5, 30.5, 1)
    ax = axes[0, 0]
    for x, c, lab in [(sizes[0], ref_c, "QM9"), (sizes[1], gen_c, "generated")]:
        ax.hist(x, bins=bins, density=True, histtype="step", color=c, lw=2, label=lab)
    ax.set_title("Atoms per molecule"); ax.legend()

    ax = axes[0, 1]
    freq = [Counter(a for s, _ in m for a in s) for m in (ref, gen)]
    tot = [sum(f.values()) for f in freq]
    xs = np.arange(len(SYMBOLS))
    ax.bar(xs - 0.2, [freq[0][a] / tot[0] for a in SYMBOLS], 0.38, color=ref_c, label="QM9")
    ax.bar(xs + 0.2, [freq[1][a] / tot[1] for a in SYMBOLS], 0.38, color=gen_c, label="generated")
    ax.set_yscale("log")  # F is ~0.1% of QM9 atoms; invisible on a linear axis
    ax.set_ylim(1e-4, 1)
    ax.set_xticks(xs, SYMBOLS); ax.set_title("Atom type frequency (log scale)"); ax.legend()

    bl = [bond_lengths(m, pairs) for m in (ref, gen)]
    for ax, p in zip([axes[0, 2], *axes[1]], pairs):
        bins = np.linspace(0.9, 1.75, 70)
        for d, c, lab in [(bl[0][p], ref_c, "QM9"), (bl[1][p], gen_c, "generated")]:
            if d:
                ax.hist(d, bins=bins, density=True, histtype="step", color=c, lw=2, label=lab)
        ax.set_title(f"{p[0]}–{p[1]} bond length (Å)")
    axes[0, 2].legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_sweep(targets, preds_by_target, ref_gaps, path):
    colors = plot_style.sequential_colors(len(targets))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    parts = ax.violinplot([preds_by_target[t] for t in targets], positions=targets,
                          widths=0.7, showmedians=True, showextrema=False)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c); body.set_edgecolor(c); body.set_alpha(0.55)
    parts["cmedians"].set_color(plot_style.TEXT)
    lim = [min(targets) - 1.5, max(targets) + 1.5]
    ax.plot(lim, lim, ls="--", lw=1, color=plot_style.TEXT_SECONDARY, label="y = x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Target gap (eV)"); ax.set_ylabel("Predicted gap of generated molecule (eV)")
    ax.set_title("Conditioning: target vs proxy-predicted gap"); ax.legend(loc="upper left")

    ax = axes[1]
    grid = np.linspace(1, 13, 400)
    ax.fill_between(grid, _kde(ref_gaps, grid), color=plot_style.QM9_COLOR, alpha=0.25, lw=0,
                    label="QM9 (all)")
    for t, c in zip(targets, colors):
        ax.plot(grid, _kde(preds_by_target[t], grid), color=c, label=f"target {t:g} eV")
        ax.axvline(t, color=c, lw=1, ls=":")
    ax.set_xlabel("Predicted gap (eV)"); ax.set_ylabel("density"); ax.set_yticks([])
    ax.set_title("Gap distribution of generated molecules"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _kde(x, grid, bw=None):
    x = np.asarray(x)
    bw = bw or 1.06 * x.std() * len(x) ** (-1 / 5) + 1e-3
    return np.exp(-0.5 * ((grid[:, None] - x[None]) / bw) ** 2).sum(1) / (len(x) * bw * np.sqrt(2 * np.pi))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", default=None)
    p.add_argument("--n_samples", type=int, default=10000, help="samples for quality metrics")
    p.add_argument("--n_cond", type=int, default=2000, help="samples for conditioning MAE")
    p.add_argument("--n_per_target", type=int, default=500)
    p.add_argument("--targets", type=float, nargs="+", default=[4, 5, 6, 7, 8, 9])
    p.add_argument("--regressor", default="checkpoints/regressor.pt")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None, help="default: <run_dir>/eval")
    args = p.parse_args()

    plot_style.apply()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, size_pred, fm, config = load_run(args.run_dir, device)
    out_dir = args.out_dir or os.path.join(config["run_dir"], "eval")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Evaluating {config['checkpoint']} -> {out_dir}")

    train_ds, _, test_ds, _, _ = get_datasets(normalize=False)
    train_gaps = train_ds.y.numpy()
    test_gaps = test_ds.y.numpy()
    known = train_smiles(train_ds)
    results = {"checkpoint": config["checkpoint"], "steps": args.steps}

    # 1. quality, with gaps drawn from the training distribution
    cond = np.random.choice(train_gaps, args.n_samples)
    gen = sample_many(model, size_pred, fm, config, cond, device, steps=args.steps)
    results["quality"], _ = quality_metrics(gen, known)
    ref = dataset_mols(test_ds, min(args.n_samples, 2000))
    results["quality_qm9_reference"], _ = quality_metrics(ref, known | {None})
    print(json.dumps(results["quality"], indent=2))

    # 2. distributions
    plot_distributions(gen[:5000], ref, os.path.join(out_dir, "distributions.png"))

    # 3. conditioning, scored by the proxy regressor
    if os.path.exists(args.regressor):
        from regressor import load_regressor, predict_gaps
        reg, r_mean, r_std = load_regressor(args.regressor, device)

        test_ref = dataset_mols(test_ds, len(test_ds))
        pred_test = predict_gaps(reg, r_mean, r_std, test_ref, device)
        true_test = test_gaps
        # #atoms baseline: mean train gap per molecule size
        n_train = np.diff(train_ds.slices["x"].numpy())[np.asarray(train_ds.indices())]
        size_mean = {k: train_gaps[n_train == k].mean() for k in np.unique(n_train)}
        n_test = np.array([len(s) for s, _ in test_ref])
        pred_size = np.array([size_mean.get(k, train_gaps.mean()) for k in n_test])

        targets = np.random.choice(test_gaps, args.n_cond)
        gen_c = sample_many(model, size_pred, fm, config, targets, device, steps=args.steps)
        pred_gen = predict_gaps(reg, r_mean, r_std, gen_c, device)
        valid = np.array([mol_to_smiles(s, p) is not None for s, p in gen_c])

        results["conditioning_mae_ev"] = {
            "nano-flow (all samples)": float(np.abs(pred_gen - targets).mean()),
            "nano-flow (valid only)": float(np.abs(pred_gen - targets)[valid].mean()),
            "baseline: shuffled targets": float(np.abs(pred_test - np.random.permutation(true_test)).mean()),
            "baseline: #atoms only": float(np.abs(pred_size - true_test).mean()),
            "lower bound: regressor on QM9 test": float(np.abs(pred_test - true_test).mean()),
        }
        print(json.dumps(results["conditioning_mae_ev"], indent=2))

        sweep = {}
        for t in args.targets:
            mols = sample_many(model, size_pred, fm, config, [t] * args.n_per_target, device, steps=args.steps)
            pg = predict_gaps(reg, r_mean, r_std, mols, device)
            sweep[t] = pg
            q, _ = quality_metrics(mols, known)
            results.setdefault("sweep", {})[f"{t:g}"] = {
                "pred_mean": float(pg.mean()), "pred_std": float(pg.std()),
                "mae": float(np.abs(pg - t).mean()), "validity": q["validity"],
                "mol_stability": q["mol_stability"],
                "mean_atoms": float(np.mean([len(s) for s, _ in mols]))}
        plot_sweep(args.targets, sweep, np.concatenate([train_gaps, test_gaps]),
                   os.path.join(out_dir, "conditioning.png"))
        print(json.dumps(results["sweep"], indent=2))
    else:
        print(f"Regressor {args.regressor} not found; skipping conditioning metrics (run regressor.py).")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics and figures to {out_dir}")


if __name__ == "__main__":
    main()
