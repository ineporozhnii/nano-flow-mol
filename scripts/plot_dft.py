"""DFT gaps of generated molecules (one colour per target gap), in the style of a joint plot with marginals.

Reads results of dft_eval.py. Two modes:
  * with proxy gaps (a regressor was used): y = proxy-predicted gap, gray cloud = QM9 test set (DFT vs proxy)
  * without (default when no regressor): y = target gap, top marginal also shows the QM9 gap
    distribution. In both modes, points on the dashed y = x line are molecules whose DFT gap hits the target.
"""
import os
import sys
import json
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import plot_style
from plot_style import plt

# marker categories, checked in this order (first match wins)
CATEGORIES = [
    ("oo", "O–O bond", dict(marker="o", filled=False)),
    ("conf", "graph changed during MMFF conformation", dict(marker="v", filled=False)),
    ("relax", "graph changed during DFT relaxation", dict(marker="v", filled=True)),
    ("ok", "unchanged graph, no O–O bond", dict(marker="o", filled=True)),
]


def category(r):
    if r["oo_bond"]:
        return "oo"
    if r["graph_changed_conformation"]:
        return "conf"
    if r["graph_changed_relaxation"]:
        return "relax"
    return "ok"


def kde(x, grid):
    x = np.asarray(x)
    if len(x) < 2:
        return np.zeros_like(grid)
    bw = 1.06 * x.std() * len(x) ** (-1 / 5) + 0.05
    return np.exp(-0.5 * ((grid[:, None] - x[None]) / bw) ** 2).sum(1) / (len(x) * bw * np.sqrt(2 * np.pi))


def qm9_background(regressor, cache):
    if os.path.exists(cache):
        d = np.load(cache)
        return d["true"], d["pred"]
    from data import get_datasets
    from chem import SYMBOLS
    from regressor import load_regressor, predict_gaps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg, m, s = load_regressor(regressor, device)
    _, _, test_ds, _, _ = get_datasets(normalize=False)
    mols = [([SYMBOLS[t] for t in d.x.argmax(-1).tolist()], d.pos.numpy()) for d in test_ds]
    pred = predict_gaps(reg, m, s, mols, device)
    true = test_ds.y.numpy()
    np.savez(cache, true=true, pred=pred)
    return true, pred


def qm9_gaps(cache):
    if os.path.exists(cache):
        return np.load(cache)["gaps"]
    from data import get_datasets
    tr, va, te, _, _ = get_datasets(normalize=False)
    gaps = np.concatenate([tr.y.numpy(), va.y.numpy(), te.y.numpy()])
    np.savez(cache, gaps=gaps)
    return gaps


def summarize(records, targets):
    rows = {}
    for t in targets:
        rs = [r for r in records if r["target_ev"] == t]
        if not rs:
            continue
        dft = np.array([r["dft_opt"]["gap_ev"] for r in rs])
        ok = np.array([category(r) == "ok" for r in rs])
        rows[f"{t:g}"] = {
            "n": len(rs),
            "dft_gap_mean": float(dft.mean()), "dft_gap_std": float(dft.std()),
            "dft_mae_to_target": float(np.abs(dft - t).mean()),
            "frac_within_0.5eV": float((np.abs(dft - t) < 0.5).mean()),
            "frac_within_1eV": float((np.abs(dft - t) < 1.0).mean()),
            "frac_unchanged_graph_no_OO": float(ok.mean()),
            "dft_mae_to_target_unchanged_only": float(np.abs(dft[ok] - t).mean()) if ok.any() else None,
        }
        if all(r.get("proxy_gap_ev") is not None for r in rs):
            proxy = np.array([r["proxy_gap_ev"] for r in rs])
            rows[f"{t:g}"]["proxy_vs_dft_mae"] = float(np.abs(proxy - dft).mean())
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dft_dir", default="results/dft")
    p.add_argument("--regressor", default="checkpoints/regressor.pt")
    p.add_argument("--x", default="dft_opt", choices=["dft_opt", "sp_raw"],
                   help="DFT gap at the relaxed geometry (default) or single point on the raw geometry")
    p.add_argument("--y", default="auto", choices=["auto", "target", "sp_raw", "proxy"],
                   help="y-axis: target gap (jittered), DFT single point on the raw generated geometry, or proxy "
                        "regressor gap; auto = proxy if available, else target")
    p.add_argument("--jitter", type=float, default=0.0,
                   help="--y target only: random vertical spread (eV) to separate overlapping points")
    p.add_argument("--markers", action="store_true",
                   help="marker shape per relaxation outcome (O-O bond / graph changed), with legend entries")
    p.add_argument("--out", default="assets/dft_eval.png")
    args = p.parse_args()
    plot_style.apply()

    with open(os.path.join(args.dft_dir, "candidates.json")) as f:
        targets = json.load(f)["targets"]
    records = []
    for path in sorted(glob.glob(os.path.join(args.dft_dir, "molecules", "*.json"))):
        with open(path) as f:
            r = json.load(f)
        if "error" not in r and r.get(args.x, {}).get("scf_converged", False):
            records.append(r)
    n_err = len(glob.glob(os.path.join(args.dft_dir, "molecules", "*.json"))) - len(records)
    print(f"{len(records)} molecules with converged DFT ({n_err} failed/unconverged)")

    has_proxy = bool(records) and all(r.get("proxy_gap_ev") is not None for r in records) \
        and os.path.exists(args.regressor)
    ymode = ("proxy" if has_proxy else "target") if args.y == "auto" else args.y
    if ymode == "proxy" and not has_proxy:
        sys.exit("--y proxy needs proxy gaps (run dft_eval.py with a regressor)")
    if ymode == "sp_raw":
        records = [r for r in records if r.get("sp_raw", {}).get("scf_converged", False)]
    yval = {"proxy": lambda r: r["proxy_gap_ev"], "sp_raw": lambda r: r["sp_raw"]["gap_ev"]}.get(ymode)
    has_right = ymode != "target"
    colors = dict(zip(targets, plot_style.CATEGORICAL))
    lo, hi = 0, 12
    grid = np.linspace(lo, hi, 500)
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(7.2, 7.2) if has_right else (6.4, 7.2))
    if has_right:
        gs = GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1.3, 5], hspace=0.04, wspace=0.04, figure=fig)
        ax = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    else:
        gs = GridSpec(2, 1, height_ratios=[1.3, 5], hspace=0.04, figure=fig)
        ax = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_right = None
    if ymode == "proxy":
        bg_true, bg_pred = qm9_background(args.regressor, os.path.join(args.dft_dir, "qm9_test_proxy.npz"))
        ax.scatter(bg_true, bg_pred, s=3, color=plot_style.QM9_COLOR, alpha=0.08, lw=0, rasterized=True, zorder=0)
    else:
        qm9 = qm9_gaps(os.path.join(args.dft_dir, "qm9_gaps.npz"))
        ax_top.fill_between(grid, kde(qm9[::5], grid), color=plot_style.QM9_COLOR, alpha=0.3, lw=0, zorder=0)

    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color=plot_style.TEXT, zorder=1)
    for t in targets:
        for a in (ax, ax_top):
            a.axvline(t, color=plot_style.QM9_COLOR, lw=1, zorder=1)
        for a in [ax] + ([ax_right] if ax_right else []):
            a.axhline(t, color=plot_style.QM9_COLOR, lw=1, zorder=1)

    for t in targets:
        rs = [r for r in records if r["target_ev"] == t]
        c = colors[t]
        groups = [(k, st) for k, _, st in CATEGORIES] if args.markers else [(None, dict(marker="o", filled=True))]
        for key, style in groups:
            sel = [r for r in rs if key is None or category(r) == key]
            if not sel:
                continue
            x = [r[args.x]["gap_ev"] for r in sel]
            y = [yval(r) for r in sel] if yval else t + rng.uniform(-args.jitter, args.jitter, len(sel))
            ax.scatter(x, y, s=46, marker=style["marker"], facecolors=c if style["filled"] else "none",
                       edgecolors=c if not style["filled"] else "white", linewidths=1.4 if not style["filled"] else 0.8,
                       alpha=0.9, zorder=3)
        if len(rs) > 1:
            dft = [r[args.x]["gap_ev"] for r in rs]
            ax_top.fill_between(grid, kde(dft, grid), color=c, alpha=0.25, lw=0)
            ax_top.plot(grid, kde(dft, grid), color=c, lw=1.6)
            if has_right:
                prox = [yval(r) for r in rs]
                ax_right.fill_betweenx(grid, kde(prox, grid), color=c, alpha=0.25, lw=0)
                ax_right.plot(kde(prox, grid), grid, color=c, lw=1.6)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    xlabel = "DFT gap, relaxed geometry (eV)" if args.x == "dft_opt" else "DFT gap, generated geometry (eV)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel({"proxy": "Proxy-predicted gap (eV)", "target": "Target gap (eV)",
                   "sp_raw": "DFT gap, generated geometry (eV)"}[ymode])
    if ymode == "target":
        ax.set_yticks(targets)
        # headroom so the legend sits above the data
        ax.set_ylim(min(targets) - 1.5, max(targets) + (4.3 if args.markers else 1.5))
    for a in [ax_top] + ([ax_right] if ax_right else []):
        a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
    ax_top.tick_params(labelbottom=False, left=False, labelleft=False, bottom=False)
    ax_top.set_ylim(bottom=0)
    if ax_right:
        ax_right.tick_params(labelleft=False, bottom=False, labelbottom=False, left=False)
        ax_right.set_xlim(left=0)

    handles = [Line2D([], [], ls="none", marker="s", markersize=9, color=colors[t], label=f"target {t:g} eV")
               for t in targets]
    if args.markers:
        handles += [Line2D([], [], ls="none", marker=s["marker"], markersize=8, color=plot_style.TEXT_SECONDARY,
                           markerfacecolor=plot_style.TEXT_SECONDARY if s["filled"] else "none", label=lab)
                    for _, lab, s in CATEGORIES]
        handles.append(Line2D([], [], ls="none", marker="o" if ymode == "proxy" else "s",
                              markersize=6 if ymode == "proxy" else 9, color=plot_style.QM9_COLOR, alpha=0.5,
                              label="QM9 test set" if ymode == "proxy" else "QM9 gap distribution"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=True, framealpha=0.95,
              edgecolor=plot_style.GRID)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    summary = summarize(records, targets)
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
