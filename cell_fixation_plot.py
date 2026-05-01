from __future__ import annotations
import os
import re
from glob import glob
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import csv
import numpy as np
import matplotlib.pyplot as plt

#plot title and output names
PLOT_TITLE = "Average fixation by timestep"
OUT_PNG = "average_fixation_plot.png"
OUT_CSV = "average_fixation_plot.csv"

#list of runs to compare
#each run points to a simulation output directory
PLOT_SPECS = [
    {"label": "baseline", "sim_out": "baseline", "color": "tab:green"},
    {"label": "10% WT threshold", "sim_out": "10_threshold", "color": "tab:red"},
    {"label": "90% WT threshold", "sim_out": "90_threshold", "color": "tab:grey"},
    {"label": "95% WT threshold", "sim_out": "95_threshold", "color": "tab:purple"},
    #{"sim_out": "path/to/sim_out", "label": "experiment_X", "color": "tab:green"},
]

#if true, only compare timesteps present in every run
#if false, each run keeps its own available timesteps
REQUIRE_COMMON_TIMESTEPS = False

#regex for extracting timestep from snapshot filename
MITO_RE = re.compile(r"t(\d+)_mito\.txt$")


def parse_snapshot_homogeneity(path: str) -> Tuple[float, int]:
    #for one snapshot, measure how many cells are internally fixed
    #then ask which fixed haplotype is most common across cells
    total = 0
    current_cell_seqs: List[str] = []
    hap_to_count: Dict[str, int] = defaultdict(int)

    def finalize_cell() -> None:
        #called whenever a cell ends
        #only counts cells where every mtDNA sequence is identical
        nonlocal total, current_cell_seqs, hap_to_count

        if not current_cell_seqs:
            return

        total += 1

        #cell is counted as homogeneous only if all mtDNA copies match
        if len(set(current_cell_seqs)) == 1:
            hap_to_count[current_cell_seqs[0]] += 1

        current_cell_seqs = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith(">cell_"):
                #new cell starts, so finalize the previous cell
                finalize_cell()
                continue

            #store sequence for current cell
            current_cell_seqs.append(line.upper())

    #final cell has to be closed manually
    finalize_cell()

    if total == 0:
        return 0.0, 0

    #if no cells are internally homogeneous, homogeneity is zero
    if not hap_to_count:
        return 0.0, total

    #dominant fixed haplotype frequency across cells
    max_count = max(hap_to_count.values())
    homogeneity = max_count / total

    return homogeneity, total


def find_rep_snapshot_files(sim_out: str) -> Dict[str, List[str]]:
    #find all replicate folders and their mitotic snapshots
    #keeps files sorted by timestep within each replicate
    rep_dirs = sorted(
        d for d in glob(os.path.join(sim_out, "rep_*")) if os.path.isdir(d)
    )

    rep_to_files: Dict[str, List[str]] = {}

    for rep_dir in rep_dirs:
        files = glob(os.path.join(rep_dir, "t*_mito.txt"))
        parsed: List[Tuple[int, str]] = []

        for fp in files:
            m = MITO_RE.search(fp)
            if m:
                parsed.append((int(m.group(1)), fp))

        parsed.sort(key=lambda x: x[0])
        rep_to_files[rep_dir] = [fp for _t, fp in parsed]

    return rep_to_files


def compute_series(sim_out: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #computes mean homogeneity trajectory across replicates
    #each replicate contributes one value per timestep
    rep_to_files = find_rep_snapshot_files(sim_out)
    if not rep_to_files:
        raise FileNotFoundError(f"No rep_* folders found under: {sim_out}")

    timestep_to_values: Dict[int, List[float]] = defaultdict(list)

    for _rep_dir, files in rep_to_files.items():
        if not files:
            continue

        for fp in files:
            m = MITO_RE.search(fp)
            if not m:
                continue

            t = int(m.group(1))
            hom, total = parse_snapshot_homogeneity(fp)

            #skip empty/corrupt snapshots
            if total == 0:
                continue

            timestep_to_values[t].append(hom)

    if not timestep_to_values:
        raise FileNotFoundError(
            f"No mitotic snapshot files found under {sim_out} (expected t###_mito.txt)."
        )

    timesteps = np.array(sorted(timestep_to_values.keys()), dtype=int)

    #mean/std are across replicates at each timestep
    means = np.array([np.mean(timestep_to_values[t]) for t in timesteps], dtype=float)
    counts = np.array([len(timestep_to_values[t]) for t in timesteps], dtype=int)
    stds = np.array(
        [
            np.std(timestep_to_values[t], ddof=1)
            if len(timestep_to_values[t]) > 1
            else 0.0
            for t in timesteps
        ],
        dtype=float
    )

    #convert to percentages for plotting/output
    means_pct = 100.0 * means
    stds_pct = 100.0 * stds

    return timesteps, means_pct, stds_pct, counts


def main() -> None:
    #driver for computing and plotting homogeneity trajectories
    #compares multiple simulation output folders
    series = []

    for spec in PLOT_SPECS:
        label = spec["label"]
        sim_out = spec["sim_out"]
        color: Optional[str] = spec.get("color", None)

        ts, mp, sp, counts = compute_series(sim_out)

        series.append({
            "label": label,
            "sim_out": sim_out,
            "color": color,
            "ts": ts,
            "mp": mp,
            "sp": sp,
            "counts": counts,
        })

    if REQUIRE_COMMON_TIMESTEPS and len(series) > 1:
        #optionally restrict to timesteps shared by every run
        common = set(series[0]["ts"].tolist())

        for s in series[1:]:
            common &= set(s["ts"].tolist())

        common_ts = np.array(sorted(common), dtype=int)

        for s in series:
            idx = np.isin(s["ts"], common_ts)
            s["ts"] = s["ts"][idx]
            s["mp"] = s["mp"][idx]
            s["sp"] = s["sp"][idx]
            s["counts"] = s["counts"][idx]

    #write all run trajectories into one long-format csv
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "label",
            "timestep",
            "mean_pct_homogeneity",
            "std_pct_homogeneity",
            "n_reps",
        ])

        for s in series:
            for t, m, sd, n in zip(s["ts"], s["mp"], s["sp"], s["counts"]):
                w.writerow([s["label"], int(t), float(m), float(sd), int(n)])

    #plot mean homogeneity trajectories
    plt.figure()

    for s in series:
        plt.plot(s["ts"], s["mp"], label=s["label"], color=s["color"])

    plt.xlabel("Mitotic timestep")
    plt.ylabel("Average % homogeneity across cells")
    plt.title(PLOT_TITLE)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)

    print(f"Wrote plot: {OUT_PNG}")
    print(f"Wrote CSV: {OUT_CSV}")

    #small run summary for sanity checking
    for s in series:
        print(f"  - {s['label']}: {len(s['ts'])} timesteps from {s['sim_out']}")


if __name__ == "__main__":
    #entry point
    main()
