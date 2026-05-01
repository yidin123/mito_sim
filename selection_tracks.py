#selection_tracks.py
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SelectionModel:
    #container for selection landscape
    #effects are additive contributions, thresholds enable nonlinear behavior
    length: int
    alphabet: str
    effects: np.ndarray
    thresholds: np.ndarray
    positions: np.ndarray

def read_selection_tracks(path: str) -> SelectionModel:
    #loads selection model from npz file
    #assumes consistency with init alphabet/length
    if not os.path.exists(path):
        raise FileNotFoundError(f"selection track file not found: {path}")

    d = np.load(path, allow_pickle=False)

    #global metadata
    length = int(d["length"])
    alphabet = str(d["alphabet"])

    #effects: additive fitness contributions per position/base
    effects = np.asarray(d["effects"], dtype=float)

    #thresholds: optional nonlinear cutoffs (nan if unused)
    thresholds = np.asarray(d["thresholds"], dtype=float)

    #positions: subset of sites that actually have nontrivial effects
    #used to avoid scanning entire genome in some workflows
    positions = np.asarray(d["positions"], dtype=int)

    A = len(alphabet)

    #shape checks enforce alignment with mutation/init models
    if effects.shape != (length, A):
        raise ValueError(f"effects shape {effects.shape} != ({length},{A})")

    if thresholds.shape != (length, A):
        raise ValueError(f"thresholds shape {thresholds.shape} != ({length},{A})")

    #positions should be a simple 1D index list
    if positions.ndim != 1:
        raise ValueError("positions must be 1D")

    #ensure all positions are within valid sequence bounds
    if np.any((positions < 0) | (positions >= length)):
        raise ValueError("positions out of range")

    #no normalization here; interpretation of effects happens downstream
    return SelectionModel(
        length=length,
        alphabet=alphabet,
        effects=effects,
        thresholds=thresholds,
        positions=positions
    )
