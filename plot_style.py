"""Shared matplotlib style for all figures (colorblind-validated categorical slots, recessive axes)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# categorical slots, fixed order (blue, orange, aqua); validated all-pairs for scatter use
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]
QM9_COLOR = "#8a8984"       # reference data (QM9) is always neutral gray
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
# sequential: one hue, light -> dark, for ordered quantities such as target gap
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", ["#a9c9f0", "#2a78d6", "#0d3a73"])


def apply():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": TEXT_SECONDARY, "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "lines.linewidth": 2,
    })


def sequential_colors(n):
    return [SEQUENTIAL(i / max(n - 1, 1)) for i in range(n)]
