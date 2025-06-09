import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import ast

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
WIDTH = 0.8  # total width per group
BOX_WIDTH = 0.15  # individual box width

def load_file(path: Path) -> list[float]:
    """Load a Python-style list of numbers from a file."""
    with path.open() as f:
        return list(map(float, ast.literal_eval(f.read())))

def load_all_data(base_path: Path):
    """Load all datasets from subfolders."""
    datasets = {}  # folder name -> list of lists (group data)
    for folder in sorted(base_path.iterdir()):
        if folder.is_dir():
            group_files = sorted(folder.glob("*.txt"))
            if not group_files:
                continue
            data = [load_file(f) for f in group_files]
            datasets[folder.name] = data
    return datasets
# ─────────────────────────────────────────────────────────────────────────────

# Load all data sources (folders)
datasets = load_all_data(DATA_DIR)
n_sources = len(datasets)
source_names = list(datasets.keys())
n_groups = len(next(iter(datasets.values())))  # number of groups from first source

# Validate: all sources must have same number of groups
for name, groups in datasets.items():
    if len(groups) != n_groups:
        raise ValueError(f"Data source '{name}' has {len(groups)} groups; expected {n_groups}")

# Compute positions for each box
group_centers = np.arange(1, n_groups + 1)
total_width = min(WIDTH, n_sources * BOX_WIDTH * 1.2)
offsets = np.linspace(-total_width / 2, total_width / 2, n_sources)

# Generate distinct colors using colormap
cmap = cm.get_cmap("rainbow")
colors = [mcolors.to_hex(cmap(i / max(n_sources - 1, 1))) for i in range(n_sources)]

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

for source_idx, (source_name, group_data) in enumerate(datasets.items()):
    color = colors[source_idx]
    positions = group_centers + offsets[source_idx]

    for x, data in zip(positions, group_data):
        ax.boxplot(
            [data], positions=[x], widths=BOX_WIDTH,
            patch_artist=True, showmeans=True,
            boxprops     = dict(facecolor=color, color=color),
            medianprops  = dict(color='black'),
            whiskerprops = dict(color=color),
            capprops     = dict(color=color),
            flierprops   = dict(marker='o', color=color, markersize=4),
            meanprops    = dict(marker='D', markerfacecolor='white',
                                markeredgecolor=color, markersize=6),
        )

    # Add min/max/±1σ annotations
    for x, data in zip(positions, group_data):
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        dmin = np.min(data)
        dmax = np.max(data)

        ax.scatter(x, dmin, marker='v', color=color, s=20)
        ax.text(x + 0.05, dmin, f"{dmin:g}", va="center", fontsize=8)

        ax.scatter(x, dmax, marker='^', color=color, s=20)
        ax.text(x + 0.05, dmax, f"{dmax:g}", va="center", fontsize=8)

        ax.hlines([mean - std, mean + std], xmin=x - 0.1, xmax=x + 0.1,
                  linestyles="dashed", colors=color)
        ax.text(x - 0.1, mean + std, "+1σ", va="center", fontsize=8)
        ax.text(x - 0.1, mean - std, "−1σ", va="center", fontsize=8)



# ── Final Touches ────────────────────────────────────────────────────────────
ax.set_xticks(group_centers)
ax.set_xticklabels([f"Group {i+1}" for i in range(n_groups)])
ax.set_ylabel("Value")
ax.set_title("Comparison of Multiple Data Sources (Boxplot with Min/Max and ±1σ)")

legend_patches = [Patch(facecolor=colors[i], label=source_names[i])
                  for i in range(n_sources)]
ax.legend(handles=legend_patches, title="Data Sources",
          loc="center left", bbox_to_anchor=(1.0, 0.5))


plt.tight_layout()
plt.show()
