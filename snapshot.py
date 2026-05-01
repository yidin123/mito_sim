#snapshot.py
from __future__ import annotations
import os
import numpy as np
from sim_core import Population

def write_snapshot(path: str, pop: Population, alphabet: str = "ACGT") -> None:
    #writes snapshot to output directory
    os.makedirs(os.path.dirname(path), exist_ok=True)
    amap = np.array(list(alphabet), dtype="<U1")

    with open(path, "w", encoding="utf-8") as f:
        n = pop.n_cells
        for ci in range(n):
            f.write(f"## Cell {ci} age={pop.c_age[ci]:.6f}\n")
            M = int(pop.m_count[ci])
            for mi in range(M):
                seq_idx = pop.genomes[ci, mi, :].astype(int)
                seq = "".join(amap[seq_idx].tolist())
                f.write(f">Cell{ci}_mtDNA{mi} age={pop.m_ages[ci, mi]:.6f} {seq}\n")
            f.write("\n")
