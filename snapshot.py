from __future__ import annotations
import os
from typing import List
from sim_core import Cell

"""
writes a snapshot of simulated cell population to txt file
exports every cell, every mtDNA copy per cell, ages, and sequences
"""
def write_snapshot(path: str, population: List[Cell]) -> None:
    #gets path and creates it if non-existent
    os.makedirs(os.path.dirname(path), exist_ok=True)
    #open file for writing
    with open(path, "w", encoding="utf-8") as f:
        #loop through cells
        for ci, cell in enumerate(population):
            f.write(f"## Cell {ci} age={cell.age:.6f}\n")
            #loop through mtdna copies
            for mi, m in enumerate(cell.mtdna):
                f.write(f">Cell{ci}_mtDNA{mi} age={m.age:.6f} {m.seq}\n")
            #next line
            f.write("\n")
