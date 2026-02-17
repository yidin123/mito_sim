from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

#stores a rule key (position, from_base) and to (mutation rate of original, [to_base, probability of this])
RuleMap = Dict[Tuple[int, str], Tuple[float, List[Tuple[str, float]]]]

#dataclass to store mutation model
@dataclass(frozen=True)
class MutationModel:
    length: int
    alphabet: str
    rules: RuleMap

#helper function to normalize alphabet
def _normalize_alphabet(meta_val: str) -> str:
    #can accept '#alphabet: A C G T" or '#alphabet: ACGT"    
    s = meta_val.replace(",", " ").replace(" ", "").upper()
    if not s:
        raise ValueError("alphabet empty")
    return s

"""
reads in mutation_tracks.txt file and stores contents in MutationModel object
works for any file of that format
skips blank lines, processes lines starting with "#" as metadata when in key:value form 
"""
def read_mutation_tracks(path: str) -> MutationModel:
    #if file not found
    if not os.path.exists(path):
        raise FileNotFoundError(f"mutation_track_path not found: {path}")

    #dict to store metadata on attributes (eg. length, alphabet)
    meta: Dict[str, str] = {}
    #rulemap
    rules: RuleMap = {}

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

            #splits line (eg. position, from_base, mutation rate, targets)
            parts = line.split()
            #check if mutation rule has fewer than needed parts
            if len(parts) < 4:
                raise ValueError(f"Bad mutation rule line (need >= 4 fields): {line}")

            #assign line splits
            pos1 = int(parts[0]) #position
            from_base = parts[1].upper() #original base
            mu = float(parts[2]) #mutation rate
            to_specs = parts[3:] #targets list
            to_list: List[Tuple[str, float]] = []
            #check individual targets for ':' token
            for spec in to_specs:
                if ":" not in spec:
                    raise ValueError(f"Bad to:prob token '{spec}' in line: {line}")
                #splitting spec into base and probability
                b, p = spec.split(":", 1)
                to_list.append((b.upper(), float(p)))

            #check required metadata
            if "length" not in meta or "alphabet" not in meta:
                raise ValueError("mutation_tracks file must include '#length:' and '#alphabet:' metadata.")

            #read metadata into variables
            length = int(meta["length"])
            alphabet = _normalize_alphabet(meta["alphabet"])
            allowed = set(alphabet)

            #check metadata
            if not (1 <= pos1 <= length):
                raise ValueError(f"pos {pos1} out of range 1..{length}")
            if from_base not in allowed:
                raise ValueError(f"from_base '{from_base}' not in alphabet '{alphabet}'")
            if mu < 0:
                raise ValueError(f"mu must be >= 0, got {mu}")

            #validating individual bases and probabilities of mutation
            for tb, pr in to_list:
                if tb not in allowed:
                    raise ValueError(f"to_base '{tb}' not in alphabet '{alphabet}'")
                if pr < 0:
                    raise ValueError(f"prob must be >= 0, got {pr} in line: {line}")

            #check that probabilities of individual base mutations are valid and sum to > 0
            s = sum(pr for _tb, pr in to_list)
            if s <= 0:
                raise ValueError(f"sum of probs must be > 0 in line: {line}")

            #normalize probabilities of base change
            to_list = [(tb, pr / s) for tb, pr in to_list]

            pos0 = pos1 - 1 #convert indexing
            #add entry to rules
            rules[(pos0, from_base)] = (mu, to_list)

    #check finalized metadata
    if "length" not in meta:
        raise ValueError("missing '#length: N'")
    if "alphabet" not in meta:
        raise ValueError("missing '#alphabet: ...'")

    #variables for metadata
    length = int(meta["length"])
    alphabet = _normalize_alphabet(meta["alphabet"])

    #create and return MutationModel object
    return MutationModel(length=length, alphabet=alphabet, rules=rules)
