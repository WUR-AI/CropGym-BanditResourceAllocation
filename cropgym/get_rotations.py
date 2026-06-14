import os
import re
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.pyplot import ylabel

from cropgym import _SCENARIO_PATH

# Folder that contains `groningen`, `zeeland`, ... subfolders
BASE_DIR = _SCENARIO_PATH  # <- change this

def collect_rotation_rows(base_dir: str):
    rows = []

    for root, _, files in os.walk(base_dir):
        for fname in files:
            if not (fname.startswith("farmer_") and fname.endswith(".yaml")):
                continue

            fpath = os.path.join(root, fname)

            # Expect path: .../<region>/<year>/farmer_X.yaml
            parts = fpath.split(os.sep)
            region = parts[-3]
            year = int(parts[-2])

            m = re.search(r"farmer_(\d+)\.yaml", fname)
            if m is None:
                continue
            farmer_id = int(m.group(1))

            with open(fpath, "r") as f:
                data = yaml.safe_load(f)

            # Each key is like "field-1", "field-2", ...
            for field_key, field_info in data.items():
                if not field_key.startswith("field-"):
                    continue
                field_id = int(field_key.split("-")[1])

                crop = field_info.get("crop")
                area = field_info.get("area")
                soil_type = field_info.get("type")

                rows.append(
                    {
                        "region": region,
                        "year": year,
                        "farmer": farmer_id,
                        "field": field_id,
                        "crop": crop,
                        "area": area,
                        "soil_type": soil_type,
                    }
                )

    return rows



def plot_rotations(rotation_df: pd.DataFrame, save_path: str | None = None):
    """
    Plot all rotations as a colored table.

    x-axis: years
    y-axis: (region, farmer, field)
    colors: crop type
    """
    # Ensure deterministic order
    rotation_df = rotation_df.sort_index().sort_index(axis=1)

    # Map crops to integers and colors
    crops = ["sugarbeet", "potato", "winterwheat"]
    crop_to_color = {
        "sugarbeet": "#CCFFCC80",  # mint green
        "potato": "#865F3B80",     # brown
        "winterwheat": "#FFD96680" # light yellow
    }
    crop_to_int = {c: i for i, c in enumerate(crops)}

    # convert crop names to integer codes and TRANSPOSE
    # after transpose: rows = years, columns = (region, farmer, field)
    int_mat = rotation_df.replace(crop_to_int).values
    int_mat = int_mat.T

    cmap = ListedColormap([crop_to_color[c] for c in crops])

    n_rows, n_cols = int_mat.shape
    # size scales a bit with number of rows/cols (now wide instead of tall)
    fig, ax = plt.subplots(
        figsize=(max(6, 0.03 * n_cols), max(4, 0.4 * n_rows)),
        dpi=300,
    )
    im = ax.imshow(int_mat, aspect="auto", cmap=cmap, interpolation="none")

    # ---------------------------------------------------------------
    # Thin grid lines between every row and column (field-level grid)
    # ---------------------------------------------------------------
    ax.set_xticks([x - 0.5 for x in range(n_cols + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(n_rows + 1)], minor=True)

    ax.grid(which="minor", color="black", linewidth=0.2)

    # ------------------------------------------------------------------
    # Colored outlines per region (now along x-axis, since we transposed)
    # ------------------------------------------------------------------
    region_to_color = {
        "groningen": "green",
        "gelderland": "red",
        "zeeland": "blue",
    }

    # After transpose, fields are along the x-axis (columns of int_mat),
    # corresponding to the rows of rotation_df's MultiIndex
    regions = rotation_df.index.get_level_values("region")

    # Find contiguous column ranges for each region
    region_cols = {}
    for j, r in enumerate(regions):
        region_cols.setdefault(r, []).append(j)

    for region, cols_idx in region_cols.items():
        color = region_to_color.get(region, "black")
        start = cols_idx[0] - 0.5
        end = cols_idx[-1] + 0.5

        # Left & right borders for this region block
        ax.vlines(start, ymin=-0.5, ymax=n_rows - 0.5, color=color, linewidth=2, linestyle="--")
        ax.vlines(end,   ymin=-0.5, ymax=n_rows - 0.5, color=color, linewidth=2, linestyle="--")

        # Top & bottom horizontal borders for this region block
        ax.hlines(-0.5, xmin=start, xmax=end, color=color, linewidth=2, linestyle="--")
        ax.hlines(n_rows - 0.5, xmin=start, xmax=end, color=color, linewidth=2, linestyle="--")

    # ---- Region labels (centered along x-axis) ----
    region_centers = []
    region_labels = []

    for region, cols_idx in region_cols.items():
        start = cols_idx[0]
        end   = cols_idx[-1]
        center = (start + end) / 2.0

        region_centers.append(center)
        region_labels.append(region.capitalize())

    # ------------------------------------------------------------------
    # Tiny farm labels (centered within each farmer block)
    # ------------------------------------------------------------------
    farm_cols = {}
    for j, (reg, farm, field) in enumerate(rotation_df.index):
        farm_cols.setdefault((reg, farm), []).append(j)

    farm_centers = []
    farm_labels = []
    for (reg, farm), cols in farm_cols.items():
        center = (cols[0] + cols[-1]) / 2.0
        farm_centers.append(center)
        farm_labels.append(rf"$h_{{{int(farm)+1}}}$")

    # secondary x-axis for tiny farm labels
    ax_farm = ax.secondary_xaxis("bottom")
    ax_farm.set_xticks(farm_centers)
    ax_farm.set_xticklabels(farm_labels, fontsize=4)

    ax.set_xticks(region_centers)
    ax.set_xticklabels(region_labels, fontsize=12)
    ax.tick_params(pad=15, length=0)

    # ------------------------------------------------------------------
    # Existing farm-level separators (keep this if you still want them)
    # ------------------------------------------------------------------
    index_tuples = list(rotation_df.index)
    for i in range(1, len(index_tuples)):
        prev_region, prev_farmer, _ = index_tuples[i - 1]
        curr_region, curr_farmer, _ = index_tuples[i]

        if (prev_region != curr_region) or (prev_farmer != curr_farmer):
            ax.vlines(i - 0.5, ymin=-0.5, ymax=n_rows - 0.5, color="black", linewidth=1, linestyle=":")

    # y-axis: years (since we transposed, rows = years)
    years = list(rotation_df.columns)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(years)

    # x-axis: region/farmer/field are now summarized by region labels above
    ax.set_xlabel("Region / Farmer / Field")
    ax.set_ylabel("Year")

    ax.set_title("Crop rotations")

    # Legend
    legend_handles = [
        Patch(facecolor=crop_to_color[c], label=c) for c in crops
    ]
    ax.legend(
        handles=legend_handles,
        title="Crop",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    ax.grid(True, which="minor")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    rows = collect_rotation_rows(BASE_DIR)

    # 1) Tidy dataframe
    df = pd.DataFrame(rows)
    df = df.sort_values(["region", "farmer", "field", "year"]).reset_index(drop=True)
    print("Tidy dataframe (one row per region/year/farmer/field):")
    print(df.head())

    # 2) Rotation table: index = (region, farmer, field), columns = years, values = crop
    rotation_df = (
        df.pivot_table(
            index=["region", "farmer", "field"],
            columns="year",
            values="crop",
            aggfunc="first",
        )
        .sort_index(axis=1)
        .sort_index()
    )

    print("\nRotation dataframe (crops per year):")
    print(rotation_df.head())

    # 3) Optional: dictionary version
    rotation_dict = {}
    for _, row in df.iterrows():
        key = (row["region"], row["farmer"], row["field"])
        rotation_dict.setdefault(key, {})[row["year"]] = row["crop"]

    # Example of how one entry looks:
    example_key = next(iter(rotation_dict))
    print("\nExample dict entry for", example_key, ":", rotation_dict[example_key])

    plot_rotations(rotation_df)