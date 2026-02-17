from __future__ import annotations
import argparse
import os
import re
from glob import glob
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import csv

#1-indexed mtDNA position to analyze (eg. 1 and onwards)
POS1_DEFAULT = 1 

#output file name
OUT_PNG_DEFAULT = "basefreq_by_timestep_multi.png"

#list of runs (label: name, sim_out: directory, color: color you want the line)
RUNS = [
    {"label": "basic_test", "sim_out": "transfer_base", "color": "tab:green"},
    {"label": "transfer_test", "sim_out": "transfer_test", "color": "tab:olive"},
    #{"sim_out": "path/to/sim_out3", "label": "experiment_X", "color": "tab:green"},
]

#regex to extract timestep from snapshots
MITO_RE = re.compile(r"t(\d+)_mito\.txt$")

"""
function to find snapshot files and extract timesteps
returns them sorted by time
"""
def iter_mito_files(sim_out: str) -> List[Tuple[int, str]]:
    #find all snapshot files
    files = glob(os.path.join(sim_out, "rep_*", "t*_mito.txt"))
    #create container to hold timestep, filename
    parsed: List[Tuple[int, str]] = []
    #extract timestep from filename with REGEX
    for fp in files:
        m = MITO_RE.search(fp)
        if m:
            parsed.append((int(m.group(1)), fp))
    #sort by timestep
    parsed.sort(key=lambda x: x[0])
    #return result
    return parsed

#try to infer a cell identifier from the header line
def extract_cell_id_from_header(parts: List[str]) -> str:
    #look for token like cell=12, cell:12, cell12, cell_12
    for tok in parts:
        t = tok.strip()
        if not t:
            continue
        low = t.lower()
        if low.startswith("cell=") or low.startswith("cell:"):
            return t.split("=", 1)[-1].split(":", 1)[-1]
        if low.startswith("cell_") or low.startswith("cell"):
            #cell_12 or cell12
            m = re.search(r"cell[_:]?(\d+)", low)
            if m:
                return m.group(1)

    #fallback: treat first token (after '>') as cell id
    #this matches many formats like: >0  ...   or >cell0 ...
    return parts[0]

"""
opens snapshot file and counts which base appears at specific mtdna position across all copies
"""
def count_bases_at_pos_in_snapshot_by_cell(path: str, pos0: int) -> Dict[str, Dict[str, int]]:
    #initialize counters
    per_cell: Dict[str, Dict[str, int]] = defaultdict(lambda: {"A": 0, "T": 0, "G": 0, "C": 0})

    #open snapshot file and read line by line
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or not line.startswith(">"):
                continue

            #split tokens
            parts = line.split()
            if len(parts) < 2:
                continue

            #acquire exact sequence and check if position is valid
            seq = parts[-1].upper()
            if pos0 < 0 or pos0 >= len(seq):
                continue

            #get base
            base = seq[pos0]
            #count bases
            if base in ("A", "T", "G", "C"):
                #infer cell id
                cell_id = extract_cell_id_from_header(parts)
                per_cell[cell_id][base] += 1

    #return
    return dict(per_cell)

"""
takes one simulation output directory and produces:
for each timestep % of base pair at chosen position, aggregated across all cells and reps
"""
def compute_basefreq_series(sim_out: str, pos1: int) -> Tuple[List[int], Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    #convert to python indexing
    pos0 = pos1 - 1
    #get all snapshot files
    mito_files = iter_mito_files(sim_out)
    #if no files found
    if not mito_files:
        raise FileNotFoundError(f"No files found matching {sim_out}/rep_*/t*_mito.txt")

    #prepare accumulators
    #store per-cell percentages (not pooled counts) so we can mean/stdev later
    timestep_cell_pcts: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {"A": [], "T": [], "G": [], "C": []})

    #loop through all snapshot files
    for t, fp in mito_files:
        #count bases inside snapshot (by cell)
        per_cell_counts = count_bases_at_pos_in_snapshot_by_cell(fp, pos0)

        #convert each cell's counts into percentages and collect them
        for _cell_id, c in per_cell_counts.items():
            tot = float(c["A"] + c["T"] + c["G"] + c["C"])
            if tot <= 0:
                continue
            for b in ("A", "T", "G", "C"):
                timestep_cell_pcts[t][b].append(100.0 * float(c[b]) / tot)

    #sort timesteps
    timesteps = sorted(timestep_cell_pcts.keys())

    #compute mean/stdev across cells for each timestep
    mean_pct: Dict[str, np.ndarray] = {}
    std_pct: Dict[str, np.ndarray] = {}
    n_cells = np.array([len(timestep_cell_pcts[t]["A"]) for t in timesteps], dtype=int)

    for b in ("A", "T", "G", "C"):
        arrs: List[float] = []
        #build per-timestep arrays (ragged -> compute per timestep)
        means: List[float] = []
        stds: List[float] = []
        for t in timesteps:
            vals = np.array(timestep_cell_pcts[t][b], dtype=float)
            if vals.size == 0:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(float(np.mean(vals)))
                #sample stdev (ddof=1) when possible, else 0
                stds.append(float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0)
        mean_pct[b] = np.array(means, dtype=float)
        std_pct[b] = np.array(stds, dtype=float)

    #return results
    return timesteps, mean_pct, std_pct, n_cells

def safe_slug(s: str) -> str:
    s2 = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
    return s2 if s2 else "run"

def write_cellavg_csv(path: str, timesteps: List[int], mean_pct: Dict[str, np.ndarray], std_pct: Dict[str, np.ndarray], n_cells: np.ndarray, pos1: int) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "timestep",
            f"A_mean_pos{pos1}", f"A_std_pos{pos1}",
            f"T_mean_pos{pos1}", f"T_std_pos{pos1}",
            f"G_mean_pos{pos1}", f"G_std_pos{pos1}",
            f"C_mean_pos{pos1}", f"C_std_pos{pos1}",
            "n_cells",
        ])
        for i, t in enumerate(timesteps):
            w.writerow([
                t,
                mean_pct["A"][i], std_pct["A"][i],
                mean_pct["T"][i], std_pct["T"][i],
                mean_pct["G"][i], std_pct["G"][i],
                mean_pct["C"][i], std_pct["C"][i],
                int(n_cells[i]),
            ])

"""
main plotting function
"""
def main():
    #argument inputs
    pos1 = POS1_DEFAULT
    out_png = OUT_PNG_DEFAULT

    #for each run, draw 4 lines in that run's color but different linestyles
    base_linestyle = {"A": "-", "T": "--", "G": "-.", "C": ":"}

    #make figure
    plt.figure()

    #flag for plotting
    any_plotted = False
    #loop over runs
    for run in RUNS:
        sim_out = run["sim_out"]
        label = run.get("label", sim_out)
        color = run.get("color", None)

        #compute base frequency time series
        try:
            timesteps, mean_pct, std_pct, n_cells = compute_basefreq_series(sim_out, pos1)
        except FileNotFoundError as e:
            print(f"[WARN] Skipping '{label}' ({sim_out}): {e}")
            continue

        #write csv for this run
        csv_path = f"{safe_slug(label)}_cellavg_pos{pos1}.csv"
        write_cellavg_csv(csv_path, timesteps, mean_pct, std_pct, n_cells, pos1)
        print(f"Done. Wrote CSV: {csv_path}")

        #plot the four lines
        for b in ("A", "T", "G", "C"):
            # Legend shows both run label and base
            line_label = f"{label} — {b}"
            plt.plot(
                timesteps,
                mean_pct[b],
                label=line_label,
                color=color,
                linestyle=base_linestyle[b],
            )
            any_plotted = True

    #if nothing plotted
    if not any_plotted:
        raise SystemExit("No runs were plotted (all RUNS missing files?). Check RUNS paths.")

    #labels
    plt.xlabel("Mitotic timestep")
    plt.ylabel(f"Mean base proportion per cell at position {pos1} (%)")
    plt.title(f"Base composition at mtDNA position {pos1} across timesteps (per-cell mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Done. Wrote plot: {out_png}")

#run main
if __name__ == "__main__":
    main()

