#mutation_tracks.py
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class MutationModel:
    #container for mutation dynamics
    #mu controls mutation rates, to_probs controls destination distribution
    length: int
    alphabet: str
    mu: np.ndarray
    to_probs: np.ndarray

def read_mutation_tracks(path: str) -> MutationModel:
    #loads mutation model from npz file
    #assumes alignment with init alphabet and sequence length
    if not os.path.exists(path):
        raise FileNotFoundError(f"mutation_track_path not found: {path}")

    d = np.load(path, allow_pickle=False)

    #global metadata
    length = int(d["length"])
    alphabet = str(d["alphabet"])

    #mu: per-position, per-base mutation rate
    mu = np.asarray(d["mu"], dtype=float)

    #to_probs: per-position, per-base transition probabilities
    to_probs = np.asarray(d["to_probs"], dtype=float)

    A = len(alphabet)

    #shape checks ensure consistency with simulation expectations
    if mu.shape != (length, A):
        raise ValueError(f"mu shape {mu.shape} != ({length},{A})")

    if to_probs.shape != (length, A, A):
        raise ValueError(f"to_probs shape {to_probs.shape} != ({length},{A},{A})")

    #no further normalization here; assumes preprocessing already handled it
    return MutationModel(
        length=length,
        alphabet=alphabet,
        mu=mu,
        to_probs=to_probs
    )
