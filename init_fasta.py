# init_fasta.py:
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

#class to store possible starting cell type data from input
@dataclass(frozen=True)
class InitCellType:
    name: str
    proportion: float
    spec: InitSpec

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

"""
reads in mtDNA_init.fasta file and stores contents in InitCellType list
works for any file of that format, with the following specifications:
    - supports legacy single-type format (no #celltype lines) and returns one InitCellType
    - supports multi-type format using '#celltype: NAME proportion=... mutant_fraction=...'
    - all cell types share one global alphabet and length
    - each cell type requires one WT and proportion_within_mutants for all non-WT records
"""
def read_init_fasta_multi(path: str) -> List[InitCellType]:
    #if file not found
    if not os.path.exists(path):
        raise FileNotFoundError(f"init_fasta_path not found: {path}")

    #first pass: check if file has any #celltype lines
    has_celltype = False
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#celltype:"):
                has_celltype = True
                break

    #legacy behavior: no cell types specified
    if not has_celltype:
        spec = read_init_fasta(path)
        return [InitCellType(name="DEFAULT", proportion=1.0, spec=spec)]

    #dict to store metadata on attributes (eg. length, alphabet)
    meta_global: Dict[str, str] = {}

    #container for output cell types
    celltypes: List[InitCellType] = []

    #current celltype state
    current_ct_name: str | None = None
    current_ct_attrs: Dict[str, str] = {}
    current_records: List[Tuple[str, Dict[str, str], str]] = []

    #stores id of sequence parsed
    current_id: str | None = None
    #stores attributes/metadata
    current_attrs: Dict[str, str] = {}
    #stores sequence lines if broken up
    current_seq_parts: List[str] = []

    #helper subfunction to append a particular record after parsing
    def flush_record() -> None:
        nonlocal current_id, current_attrs, current_seq_parts, current_records
        if current_id is None:
            return #no sequence parsed/open
        seq = "".join(current_seq_parts).replace(" ", "").strip().upper() #combines sequences, normalizes
        if not seq:
            #no sequence found
            raise ValueError(f"empty sequence for record {current_id} in {path}")

        #append parsed sequence to records tuple list
        current_records.append((current_id, current_attrs, seq))
        #refresh storage varables for next round
        current_id = None
        current_attrs = {}
        current_seq_parts = []

    #helper subfunction to finalize a celltype block after parsing
    def flush_celltype() -> None:
        nonlocal current_ct_name, current_ct_attrs, current_records

        if current_ct_name is None:
            return

        #if global metadata missing
        if "length" not in meta_global:
            raise ValueError("missing '#length: N' in init fasta metadata.")
        if "alphabet" not in meta_global:
            raise ValueError("missing '#alphabet: ...' in init fasta metadata.")

        #read in metadata information
        length = int(meta_global["length"])
        alphabet = _normalize_alphabet(meta_global["alphabet"])

        #errors in metadata information
        if length <= 0:
            raise ValueError(f"length must be > 0, got {length}")

        #celltype metadata
        if "proportion" not in current_ct_attrs:
            raise ValueError(f"celltype {current_ct_name} missing proportion=... in header")
        if "mutant_fraction" not in current_ct_attrs:
            raise ValueError(f"celltype {current_ct_name} missing mutant_fraction=... in header")

        proportion = float(current_ct_attrs["proportion"])
        mutant_fraction = float(current_ct_attrs["mutant_fraction"])

        if not (0.0 <= mutant_fraction <= 1.0):
            raise ValueError(f"mutant_fraction must be in [0, 1], got {mutant_fraction} in celltype {current_ct_name}")

        #wild-type sequence
        #read in
        wt = [r for r in current_records if r[0] == "WT"]
        if len(wt) != 1: #too many WT or too few
            raise ValueError(f"expected exactly one WT record with header '>WT' in celltype {current_ct_name}")
        #get sequence
        wt_sequence = wt[0][2]
        #check if WT length matches specified meta
        if len(wt_sequence) != length:
            raise ValueError(f"WT length {len(wt_sequence)} != meta length {length} in celltype {current_ct_name}")

        #check if base pairs used are in alphabet
        allowed = set(alphabet)
        for rid, _attrs, seq in current_records:
            bad = {ch for ch in seq if ch not in allowed}
            #if not found
            if bad:
                raise ValueError(f"record {rid} has invalid bases {sorted(bad)} not in alphabet {alphabet} in celltype {current_ct_name}")

        #haplotype sequence processing
        haps: List[InitHaplotype] = [] #to store them
        for rid, attrs, seq in current_records:
            if rid == "WT":
                continue
            #if length not matching meta
            if len(seq) != length:
                raise ValueError(f"record {rid} length {len(seq)} != meta length {length} in celltype {current_ct_name}")
            #if missing proportion in mutants
            if "proportion_within_mutants" not in attrs:
                raise ValueError(f"record {rid} missing proportion_within_mutants=... in header in celltype {current_ct_name}")
            #read in proportion within mutants
            p = float(attrs["proportion_within_mutants"])
            #attach to haps
            haps.append(InitHaplotype(name=rid, sequence=seq, proportion_within_mutants=p))

        #if mutant fraction > 0 but no mutant haplotypes
        if mutant_fraction > 0 and not haps:
            raise ValueError(f"mutant_fraction > 0 but no mutant haplotypes provided in celltype {current_ct_name}.")

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

        #construct InitSpec
        spec = InitSpec(
            length=length,
            mutant_fraction=mutant_fraction,
            wt_sequence=wt_sequence,
            haplotypes=haps,
            alphabet=alphabet,
        )

        #append celltype to output
        celltypes.append(InitCellType(name=current_ct_name, proportion=proportion, spec=spec))

        #reset for next block
        current_ct_name = None
        current_ct_attrs = {}
        current_records = []

    #FASTA parser function
    with open(path, "r", encoding="utf-8") as f:
        for raw in f: #opens fasta and reads one at a time
            #normalize line to remove blank spaces and skip empty lines
            line = raw.strip()
            if not line:
                continue

            #read in celltype line
            if line.startswith("#celltype:"):
                flush_record()
                flush_celltype()

                tail = line[len("#celltype:"):].strip()
                if not tail:
                    raise ValueError("empty celltype header after '#celltype:'")

                parts = tail.split(None, 1)
                current_ct_name = parts[0].strip()
                rest = parts[1] if len(parts) > 1 else ""
                current_ct_attrs = _parse_header_kv(rest)

                continue

            #read in metadata line (eg. #length: 10)
            if line.startswith("#"):
                line2 = line[1:].strip() #normalize
                if ":" in line2:
                    k, v = line2.split(":", 1)
                    meta_global[k.strip().lower()] = v.strip() #stores in meta
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
    flush_celltype()

    #if no celltypes were read
    if not celltypes:
        raise ValueError("no cell types found; expected at least one '#celltype: ...' block")

    #normalize cell type proportions
    s = sum(ct.proportion for ct in celltypes)
    if s <= 0:
        raise ValueError("sum of cell type proportions must be > 0")

    celltypes = [
        InitCellType(name=ct.name, proportion=ct.proportion / s, spec=ct.spec)
        for ct in celltypes
    ]

    return celltypes
