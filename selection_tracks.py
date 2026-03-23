#selection_tracks.py
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SelectionModel:
    length: int
    alphabet: str
    effects: np.ndarray    
    thresholds: np.ndarray 
    positions: np.ndarray   

def read_selection_tracks(path: str) -> SelectionModel:
    if not os.path.exists(path):
        raise FileNotFoundError(f"selection track file not found: {path}")

    d = np.load(path, allow_pickle=False)
    length = int(d["length"])
    alphabet = str(d["alphabet"])
    effects = np.asarray(d["effects"], dtype=float)
    thresholds = np.asarray(d["thresholds"], dtype=float)
    positions = np.asarray(d["positions"], dtype=int)

    A = len(alphabet)
    if effects.shape != (length, A):
        raise ValueError(f"effects shape {effects.shape} != ({length},{A})")
    if thresholds.shape != (length, A):
        raise ValueError(f"thresholds shape {thresholds.shape} != ({length},{A})")
    if positions.ndim != 1:
        raise ValueError("positions must be 1D")
    if np.any((positions < 0) | (positions >= length)):
        raise ValueError("positions out of range")

    return SelectionModel(length=length, alphabet=alphabet, effects=effects, thresholds=thresholds, positions=positions)
