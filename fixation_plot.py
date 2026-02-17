from __future__ import annotations
import os
import re
from glob import glob
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import csv
import numpy as np
import matplotlib.pyplot as plt

#plot title, plot name
PLOT_TITLE = "Average fixation by timestep"
OUT_PNG = "age_bias_fixation_plot.png"
OUT_CSV = "age_bias_fixation_plot.csv"

#each item here becomes one line on the plot
#label denotes what to call the particular run
#sim_out denotes the outdirectory of simulation snapshots
#set color to what it is on line
PLOT_SPECS = [
    {"label": "basic_test", "sim_out": "transfer_base", "color": "tab:green"},
    {"label": "transfer_test", "sim_out": "transfer_test", "color": "tab:olive"}, 
    #{"sim_out": "path/to/sim_out3", "label": "experiment_X", "color": "tab:green"},
]

#if true, we only plot timesteps that exist in all series (common/shared)
#if false, each series plots what timesteps it has
REQUIRE_COMMON_TIMESTEPS = False

#regex for timestep extraction
MITO_RE = re.compile(r"t(\d+)_mito\.txt$")

"""
given one snapshot file, compute the total amount of cells and fixed cells
"""
def parse_snapshot_fixation(path: str) -> Tuple[int, int]:
    #initialize total, fixed cell counts
    fixed = 0
    total = 0
    #store sequences in current cell
    current_cell_seqs: List[str] = []

    #check if cell fixed
    def finalize_cell():
        nonlocal fixed, total, current_cell_seqs
        #if no sequences
        if not current_cell_seqs:
            return

        #increment total cells
        total += 1
        #if all sequences in this cell are identical, fixed
        if len(set(current_cell_seqs)) == 1:
            fixed += 1
        #renew current_cell_seqs
        current_cell_seqs = []

    #read file line by line
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            #hits end of cell
            if line.startswith("## Cell"):
                finalize_cell() #check if fixed
                continue

            #if sequence line
            if line.startswith(">"):
                parts = line.split()
                #if no sequence there
                if len(parts) < 2:
                    continue
                #collect and append sequence
                seq = parts[-1].upper()
                current_cell_seqs.append(seq)

    #check if fixed for last cell
    finalize_cell()
    #return
    return fixed, total

"""
function to find all snapshot files
output is a dictionary mapping rep to timestep txt files (filepaths)
"""
def find_rep_snapshot_files(sim_out: str) -> Dict[str, List[str]]:
    #find all rep directories
    rep_dirs = sorted(
        d for d in glob(os.path.join(sim_out, "rep_*")) if os.path.isdir(d)
    )

    #creates dictionary
    rep_to_files: Dict[str, List[str]] = {}

    #loop over each rep
    for rep_dir in rep_dirs:
        #find snapshot files in this rep
        files = glob(os.path.join(rep_dir, "t*_mito.txt"))
        #extract timestep and pair with file
        parsed: List[Tuple[int, str]] = []
        #loop over snapshot files
        for fp in files:
            m = MITO_RE.search(fp)
            #if regex matches, append
            if m:
                parsed.append((int(m.group(1)), fp))

        #sort by timestep
        parsed.sort(key=lambda x: x[0])
        #store only file paths
        rep_to_files[rep_dir] = [fp for _t, fp in parsed]

    #return filepaths dictionary
    return rep_to_files

"""
function to compute timesteps found for simulation (one output folder),
mean % fixed cells across reps at a given timestep,
and how many replications contributed at a timestep
"""
def compute_series(sim_out: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #find all snapshot files in sim
    rep_to_files = find_rep_snapshot_files(sim_out)
    if not rep_to_files:
        raise FileNotFoundError(f"No rep_* folders found under: {sim_out}")

    #prepare container to store fixation from each rep at a given timestep
    timestep_to_values: Dict[int, List[float]] = defaultdict(list)

    #loop over replications
    for rep_dir, files in rep_to_files.items():
        #skip empty reps
        if not files:
            continue
        #loop over snapshot files
        for fp in files:
            #extract timestep from filename
            m = MITO_RE.search(fp)
            if not m:
                continue
            t = int(m.group(1))

            #compute fixation, total from snapshot
            fixed, total = parse_snapshot_fixation(fp)
            #if no cells
            if total == 0:
                continue
            #compute fixed fraction
            frac = fixed / total
            #store for this timestep
            timestep_to_values[t].append(frac)

    #if no snapshot files found
    if not timestep_to_values:
        raise FileNotFoundError(f"No mitotic snapshot files found under {sim_out} (expected t###_mito.txt).")
    #sort
    timesteps = np.array(sorted(timestep_to_values.keys()), dtype=int)
    #compute mean fixation per timestep
    means = np.array([np.mean(timestep_to_values[t]) for t in timesteps], dtype=float)
    #compute rep counts
    counts = np.array([len(timestep_to_values[t]) for t in timesteps], dtype=int)
    stds = np.array(
        [np.std(timestep_to_values[t], ddof=1) if len(timestep_to_values[t]) > 1 else 0.0 for t in timesteps],
        dtype=float
    )
    #convert fraction to percent
    means_pct = 100.0 * means
    stds_pct = 100.0 * stds

    #return
    return timesteps, means_pct, stds_pct, counts

"""
main function for plotting
"""
def main() -> None:
    #go through series individually
    series = []
    for spec in PLOT_SPECS:
        #take in label, sim_out. color
        label = spec["label"]
        sim_out = spec["sim_out"]
        color: Optional[str] = spec.get("color", None)

        #compute fixation per simulation
        ts, mp, sp, counts = compute_series(sim_out)
        #append results to series, with label and color
        series.append({"label": label, "sim_out": sim_out, "color": color, "ts": ts, "mp": mp, "sp": sp, "counts": counts})

    #timestep alignment
    if REQUIRE_COMMON_TIMESTEPS and len(series) > 1:
        common = set(series[0]["ts"].tolist())
        for s in series[1:]:
            common &= set(s["ts"].tolist())
        common_ts = np.array(sorted(common), dtype=int)

        #here, keep only shared/common timesteps
        for s in series:
            #keep only common timesteps
            idx = np.isin(s["ts"], common_ts)
            s["ts"] = s["ts"][idx]
            s["mp"] = s["mp"][idx]
            s["sp"] = s["sp"][idx]
            s["counts"] = s["counts"][idx]

    #write to csv
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "timestep", "mean_pct_fixed", "std_pct_fixed", "n_reps"])
        for s in series:
            for t, m, sd, n in zip(s["ts"], s["mp"], s["sp"], s["counts"]):
                w.writerow([s["label"], int(t), float(m), float(sd), int(n)])

    #plot
    plt.figure()
    for s in series:
        plt.plot(s["ts"], s["mp"], label=s["label"], color=s["color"])

    plt.xlabel("Mitotic timestep")
    plt.ylabel("Average % fixed cells")
    plt.title(PLOT_TITLE)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)

    #summary output
    print(f"Wrote plot: {OUT_PNG}")
    print(f"Wrote CSV: {OUT_CSV}")
    for s in series:
        print(f"  - {s['label']}: {len(s['ts'])} timesteps from {s['sim_out']}")

#run main
if __name__ == "__main__":
    main()

