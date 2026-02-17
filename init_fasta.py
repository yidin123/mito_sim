from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

#class to store possible variant haplotype data from input
@dataclass(frozen=True)
class InitHaplotype:
    name: str
    sequence: str
    proportion_within_mutants: float

#class to store full specifications of input haplotype variants
@dataclass(frozen=True)
class InitSpec:
    length: int
    mutant_fraction: float
    wt_sequence: str #wild-type haplotype
    haplotypes: List[InitHaplotype] #list of individual variant haplotypes
    alphabet: str #which codons are allowed (eg. ACGT or A C G T)

#helper function to parse key-value attributes from FASTA header
def _parse_header_kv(header_tail: str) -> Dict[str, str]:
    #initialize container
    kv: Dict[str, str] = {}
    for tok in header_tail.strip().split(): #split header into tokens
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k.strip()] = v.strip() #split key and value
    return kv

#helper function to normalize alphabet
def _normalize_alphabet(meta_val: str) -> str:
    #can accept '#alphabet: A C G T" or '#alphabet: ACGT"
    s = meta_val.replace(",", " ").replace(" ", "").upper()
    if not s:
        raise ValueError("alphabet empty")
    return s

"""
reads in mtDNA_init.fasta file and stores contents in InitSpec object
works for any file of that format, with the following specifications:
    - blank lines skipped
    - processes lines starting with "#" as metadata when in key:value form (else ignored) 
    - concatenates multi-line sequences
    - requires one wild-type record sequence "WT"
    - requires proportion_within_mutants for all non-WT records
"""
def read_init_fasta(path: str) -> InitSpec:
    #if file not found
    if not os.path.exists(path):
        raise FileNotFoundError(f"init_fasta_path not found: {path}")

    #dict to store metadata on attributes (eg. length, mutant_fraction)
    meta: Dict[str, str] = {}
    #list to store individual sequence records (name, proportion attribute, sequence)
    records: List[Tuple[str, Dict[str, str], str]] = []

    #stores id of sequence parsed
    current_id: str | None = None
    #stores attributes/metadata
    current_attrs: Dict[str, str] = {}
    #stores sequence lines if broken up
    current_seq_parts: List[str] = []

    #helper subfunction to append a particular record after parsing
    def flush_record() -> None:
        nonlocal current_id, current_attrs, current_seq_parts
        if current_id is None:
            return #no sequence parsed/open
        seq = "".join(current_seq_parts).replace(" ", "").strip().upper() #combines sequences, normalizes
        if not seq:
            #no sequence found
            raise ValueError(f"empty sequence for record {current_id} in {path}")

        #append parsed sequence to records tuple list
        records.append((current_id, current_attrs, seq))
        #refresh storage varables for next round
        current_id = None
        current_attrs = {}
        current_seq_parts = []

    #FASTA parser function
    with open(path, "r", encoding="utf-8") as f:
        for raw in f: #opens fasta and reads one at a time
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

            #read in sequence header information
            if line.startswith(">"):
                flush_record() #append previous record (parsing done)
                header = line[1:].strip() #normalize
                parts = header.split(None, 1) #split into name and proportion in mutants
                #save in storage variables
                current_id = parts[0].strip()
                tail = parts[1] if len(parts) > 1 else ""
                current_attrs = _parse_header_kv(tail)
                continue

            #if just sequence
            current_seq_parts.append(line)

    #last FASTA sequence not followed by header
    flush_record()

    #if parts of metadata missing
    if "length" not in meta:
        raise ValueError("missing '#length: N' in init fasta metadata.")
    if "mutant_fraction" not in meta:
        raise ValueError("missing '#mutant_fraction: X' in init fasta metadata.")
    if "alphabet" not in meta:
        raise ValueError("missing '#alphabet: ...' in init fasta metadata.")

    #read in metadata information
    length = int(meta["length"])
    mutant_fraction = float(meta["mutant_fraction"])
    alphabet = _normalize_alphabet(meta["alphabet"])

    #errors in metadata information
    if length <= 0:
       raise ValueError(f"length must be > 0, got {length}")
    if not (0.0 <= mutant_fraction <= 1.0):
       raise ValueError(f"mutant_fraction must be in [0, 1], got {mutant_fraction}")

    #wild-type sequence
    #read in
    wt = [r for r in records if r[0] == "WT"]
    if len(wt) != 1: #too many WT or too few
        raise ValueError("expected exactly one WT record with header '>WT'")
    #get sequence 
    wt_sequence = wt[0][2]
    #check if WT length matches specified meta 
    if len(wt_sequence) != length:
        raise ValueError(f"WT length {len(wt_sequence)} != meta length {length}")
    
    #check if base pairs used are in alphabet
    allowed = set(alphabet)
    for rid, _attrs, seq in records:
        bad = {ch for ch in seq if ch not in allowed}
        #if not found
        if bad:
            raise ValueError(f"record {rid} has invalid bases {sorted(bad)} not in alphabet {alphabet}")

    #haplotype sequence processing
    haps: List[InitHaplotype] = [] #to store them
    for rid, attrs, seq in records:
        if rid == "WT":
            continue
        #if length not matching meta 
        if len(seq) != length:
            raise ValueError(f"record {rid} length {len(seq)} != meta length {length}")
        #if missing proportion in mutants
        if "proportion_within_mutants" not in attrs:
            raise ValueError(f"record {rid} missing proportion_within_mutants=... in header")
        #read in proportion within mutants
        p = float(attrs["proportion_within_mutants"])
        #attach to haps 
        haps.append(InitHaplotype(name=rid, sequence=seq, proportion_within_mutants=p))

    #if mutant fraction > 0 but no mutant haplotypes
    if mutant_fraction > 0 and not haps:
        raise ValueError("mutant_fraction > 0 but no mutant haplotypes provided.")

    #normalize haplotype proportions (in mutants)
    if haps:
        s = sum(h.proportion_within_mutants for h in haps)
        # negative or 0 total
        if s <= 0:
            raise ValueError("sum of proportion_within_mutants must be > 0")
        #normalize each entry in haps 
        haps = [
                InitHaplotype(h.name, h.sequence, h.proportion_within_mutants / s)
                for h in haps
            ]

    #construct and return InitSpec object with sequence and specifications
    return InitSpec(
            length=length,
            mutant_fraction=mutant_fraction,
            wt_sequence=wt_sequence,
            haplotypes=haps,
            alphabet=alphabet,
        )

