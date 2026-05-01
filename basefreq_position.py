from __future__ import annotations
import os
import re
from glob import glob
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import csv

#1-indexed mtDNA position to analyze (converted to 0-index later)
POS1_DEFAULT = 2

#output file name for plot
OUT_PNG_DEFAULT = "basefreq_by_timestep_multi.png"

#list of runs to compare (each is a separate simulation output directory)
RUNS = [
    {"label": "baseline", "sim_out": "baseline", "color": "tab:green"},
    {"label": "10% WT threshold", "sim_out": "10_threshold", "color": "tab:red"},
    {"label": "90% WT threshold", "sim_out": "90_threshold", "color": "tab:grey"},
    {"label": "95% WT threshold", "sim_out": "95_threshold", "color": "tab:purple"},
]

#regex for parsing timestep from filenames
MITO_RE = re.compile(r"t(\d+)_mito\.txt$")
#regex for parsing cell headers
CELL_RE = re.compile(r"^>cell_(\d+)$", re.IGNORECASE)


def iter_mito_files(sim_out: str) -> List[Tuple[int, str]]:
    #collect all snapshot files and extract timestep from filename
    #returns sorted list so downstream assumes temporal order
    files = glob(os.path.join(sim_out, "rep_*", "t*_mito.txt"))
    parsed: List[Tuple[int, str]] = []

    for fp in files:
        m = MITO_RE.search(fp)
        if m:
            parsed.append((int(m.group(1)), fp))

    parsed.sort(key=lambda x: x[0])
    return parsed


def count_bases_at_pos_in_snapshot_by_cell(path: str, pos0: int) -> Dict[str, Dict[str, int]]:
    #for a single snapshot, count base composition at one position per cell
    #returns per-cell counts (not pooled across cells)
    per_cell: Dict[str, Dict[str, int]] = defaultdict(lambda: {"A": 0, "T": 0, "G": 0, "C": 0})
    current_cell_id: str | None = None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith(">"):
                #switch context to a new cell
                m = CELL_RE.match(line)
                current_cell_id = m.group(1) if m else None
                continue

            if current_cell_id is None:
                continue

            #skip if position is invalid for this sequence
            if pos0 < 0 or pos0 >= len(line):
                continue

            base = line[pos0].upper()

            #only count canonical bases
            if base in ("A", "T", "G", "C"):
                per_cell[current_cell_id][base] += 1

    return dict(per_cell)


def compute_basefreq_series(sim_out: str, pos1: int) -> Tuple[List[int], Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    #aggregates per-cell base frequencies across timesteps
    #key idea: compute per-cell proportions first, then average across cells
    pos0 = pos1 - 1
    mito_files = iter_mito_files(sim_out)

    if not mito_files:
        raise FileNotFoundError(f"No files found matching {sim_out}/rep_*/t*_mito.txt")

    #stores per-cell percentages for each timestep
    timestep_cell_pcts: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {"A": [], "T": [], "G": [], "C": []})

    for t, fp in mito_files:
        per_cell_counts = count_bases_at_pos_in_snapshot_by_cell(fp, pos0)

        for _cell_id, c in per_cell_counts.items():
            tot = float(c["A"] + c["T"] + c["G"] + c["C"])
            if tot <= 0:
                continue

            #convert counts to percentages per cell
            for b in ("A", "T", "G", "C"):
                timestep_cell_pcts[t][b].append(100.0 * float(c[b]) / tot)

    timesteps = sorted(timestep_cell_pcts.keys())

    #compute mean/std across cells at each timestep
    mean_pct: Dict[str, np.ndarray] = {}
    std_pct: Dict[str, np.ndarray] = {}
    n_cells = np.array([len(timestep_cell_pcts[t]["A"]) for t in timesteps], dtype=int)

    for b in ("A", "T", "G", "C"):
        means: List[float] = []
        stds: List[float] = []

        for t in timesteps:
            vals = np.array(timestep_cell_pcts[t][b], dtype=float)

            #handle empty or single-cell cases
            if vals.size == 0:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0)

        mean_pct[b] = np.array(means, dtype=float)
        std_pct[b] = np.array(stds, dtype=float)

    return timesteps, mean_pct, std_pct, n_cells


def safe_slug(s: str) -> str:
    #sanitize label for use as filename
    #removes spaces/special chars
    s2 = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
    return s2 if s2 else "run"


def write_cellavg_csv(path: str, timesteps: List[int], mean_pct: Dict[str, np.ndarray], std_pct: Dict[str, np.ndarray], n_cells: np.ndarray, pos1: int) -> None:
    #writes summary statistics (mean/std per base) to csv
    #one row per timestep
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


def main() -> None:
    #driver for comparing multiple simulation runs
    #plots mean base composition over time
    pos1 = POS1_DEFAULT
    out_png = OUT_PNG_DEFAULT

    #different bases use different linestyles for clarity
    base_linestyle = {"A": "-", "T": "--", "G": "-.", "C": ":"}

    plt.figure()
    any_plotted = False

    for run in RUNS:
        sim_out = run["sim_out"]
        label = run.get("label", sim_out)
        color = run.get("color", None)

        try:
            timesteps, mean_pct, std_pct, n_cells = compute_basefreq_series(sim_out, pos1)
        except FileNotFoundError as e:
            #skip runs that don't have valid outputs
            print(f"[WARN] Skipping '{label}' ({sim_out}): {e}")
            continue

        #write csv summary per run
        csv_path = f"{safe_slug(label)}_cellavg_pos{pos1}.csv"
        write_cellavg_csv(csv_path, timesteps, mean_pct, std_pct, n_cells, pos1)
        print(f"Done. Wrote CSV: {csv_path}")

        #plot each base as separate line
        for b in ("A", "T", "G", "C"):
            line_label = f"{label} — {b}"
            plt.plot(
                timesteps,
                mean_pct[b],
                label=line_label,
                color=color,
                linestyle=base_linestyle[b],
            )
            any_plotted = True

    #fail early if nothing was plotted
    if not any_plotted:
        raise SystemExit("No runs were plotted (all RUNS missing files?). Check RUNS paths.")

    plt.xlabel("Mitotic timestep")
    plt.ylabel(f"Mean base proportion per cell at position {pos1} (%)")
    plt.title(f"Base composition at mtDNA position {pos1} across timesteps (per-cell mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

    print(f"Done. Wrote plot: {out_png}")


if __name__ == "__main__":
    #entry point
    main()
