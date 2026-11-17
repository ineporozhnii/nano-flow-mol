"""Bar chart of sample-quality metrics (EDM protocol) from evaluate.py's metrics.json."""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plot_style
from plot_style import plt
from generate import latest_run

METRICS = [
    ("validity", "Validity"),
    ("uniqueness", "Uniqueness\n(of valid)"),
    ("novelty", "Novelty\n(of valid & unique)"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", default=None, help="default: <latest run>/eval/metrics.json")
    p.add_argument("--out", default="assets/metrics.png")
    args = p.parse_args()
    plot_style.apply()

    path = args.metrics or os.path.join(latest_run(), "eval", "metrics.json")
    with open(path) as f:
        m = json.load(f)
    gen = m["quality"]

    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    ys = list(range(len(METRICS)))[::-1]
    color = plot_style.CATEGORICAL[0]
    for y, (key, _) in zip(ys, METRICS):
        v = gen[key]
        ax.barh(y, 100 * v, height=0.56, color=color, zorder=2)
        ax.text(100 * v + 1.2, y, f"{100 * v:.1f}%", va="center", ha="left", fontsize=9.5,
                color=plot_style.TEXT, zorder=4)

    ax.set_yticks(ys, [lab for _, lab in METRICS])
    ax.set_xlim(0, 112)
    ax.set_xticks(range(0, 101, 20), [f"{t}%" for t in range(0, 101, 20)])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"Sample quality, {gen['n_samples']:,} generated molecules", loc="left", fontsize=11,
                 color=plot_style.TEXT)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
