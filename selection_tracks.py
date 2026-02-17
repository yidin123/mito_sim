from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Tuple

#dict to store selection effects of each base [[pos0, base], effect]
EffectMap = Dict[Tuple[int, str], float]

#data class to store selection model
@dataclass(frozen=True)
class SelectionModel:
    length: int
    alphabet: str
    effects: EffectMap

#helper function to normalize alphabet
def _normalize_alphabet(meta_val: str) -> str:
    #can accept '#alphabet: A C G T" or '#alphabet: ACGT"
    s = meta_val.replace(",", " ").replace(" ", "").upper()
    if not s:
        raise ValueError("alphabet empty")
    return s

"""
reads in the various selection files and stores content in SelectionModel objects
blank lines skipped; processes lines starting with "#" as metadata when in key:value form
"""
def read_selection_tracks(path: str) -> SelectionModel:
    #if file not found
    if not os.path.exists(path):
        raise FileNotFoundError(f"selection track file not found: {path}")

    #dict to store metadata on attributes (eg. length, alphabet)
    meta: Dict[str, str] = {}
    #empty EffectMap
    effects: EffectMap = {}

    #txt file parser function
    with open(path, "r", encoding="utf-8") as f:
        for raw in f: #opens txt file and reads one line at a time
            #normalize line to remove blank spaces and skip empty lines
            line = raw.strip()
            if not line:
                continue

            #read in metadata line (eg. #length: 10)
            if line.startswith("#"):
                line2 = line[1:].strip() #normalize
                if ":" in line2:
                    k, v = line2.split(":", 1)
                    meta[k.strip().lower()] = v.strip() #stores in meta
                continue

            #check required metadata
            if "length" not in meta or "alphabet" not in meta:
                raise ValueError("selection file must include '#length:' and '#alphabet:' before rules.")

            #read metadata into variables
            length = int(meta["length"])
            alphabet = _normalize_alphabet(meta["alphabet"])
            allowed = set(alphabet)

            #splits line into tokens (for position, effects)
            parts = line.split()
            #if line is not pos + alphabet bases
            if len(parts) != 1 + len(alphabet):
                raise ValueError(
                    f"Expected {1+len(alphabet)} fields (pos + {len(alphabet)} effects) but got {len(parts)}: {line}"
                )

            #input position
            pos1 = int(parts[0])
            #if position is out of bounds
            if not (1 <= pos1 <= length):
                raise ValueError(f"pos {pos1} out of range 1..{length}")

            #change indexing
            pos0 = pos1 - 1
            #selection values
            vals = [float(x) for x in parts[1:]]

            #map effects in order of alphabet string bases
            for base, eff in zip(alphabet, vals):
                if base not in allowed:
                    continue
                effects[(pos0, base)] = eff

    #check metadata
    if "length" not in meta:
        raise ValueError("missing '#length: N'")
    if "alphabet" not in meta:
        raise ValueError("missing '#alphabet: ...'")

    #return final SelectionModel object
    return SelectionModel(length=int(meta["length"]), alphabet=_normalize_alphabet(meta["alphabet"]), effects=effects)
