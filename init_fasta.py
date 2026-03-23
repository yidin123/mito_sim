#init_fasta.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass(frozen=True)
class InitSpec:
    length: int
    mutant_fraction: float
    alphabet: str
    hap_seqs: np.ndarray   
    hap_probs: np.ndarray  

@dataclass(frozen=True)
class InitCellType:
    name: str
    proportion: float
    spec: InitSpec

def read_init_fasta_multi(path: str) -> List[InitCellType]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"init_fasta_path not found: {path}")

    data = np.load(path, allow_pickle=False)
    alphabet = str(data["alphabet"])
    length = int(data["length"])
    ct_names = np.asarray(data["celltype_names"], dtype=object)
    ct_props = np.asarray(data["celltype_proportions"], dtype=float)
    mutant_fracs = np.asarray(data["mutant_fractions"], dtype=float)

    seqs = np.asarray(data["seqs"], dtype=np.int8)          
    hap_probs = np.asarray(data["hap_probs"], dtype=float)  
    hap_counts = np.asarray(data["hap_counts"], dtype=int)  

    C = int(ct_names.shape[0])
    out: List[InitCellType] = []

    for i in range(C):
        H = int(hap_counts[i])
        spec = InitSpec(
            length=length,
            mutant_fraction=float(mutant_fracs[i]),
            alphabet=alphabet,
            hap_seqs=seqs[i, :H, :],
            hap_probs=(hap_probs[i, :H] / float(hap_probs[i, :H].sum())),
        )
        out.append(InitCellType(name=str(ct_names[i]), proportion=float(ct_props[i]), spec=spec))

    s = sum(ct.proportion for ct in out)
    if s <= 0:
        raise ValueError("sum of cell type proportions must be > 0")
    out = [InitCellType(ct.name, ct.proportion / s, ct.spec) for ct in out]
    return out
