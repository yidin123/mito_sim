#convert_inputs_to_npz.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

def _normalize_alphabet(meta_val: str) -> str:
    #normalizes alphabet metadata into compact uppercase form
    #lets input use spaces/commas without changing downstream logic
    s = meta_val.replace(",", " ").replace(" ", "").upper()
    if not s:
        raise ValueError("alphabet empty")
    return s

def _parse_header_kv(header_tail: str) -> Dict[str, str]:
    #parses key=value fields from FASTA-style headers
    #used for both celltype metadata and haplotype metadata
    kv: Dict[str, str] = {}
    for tok in header_tail.strip().split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv

def _seq_to_idx(seq: str, alphabet: str) -> np.ndarray:
    #converts sequence letters into integer-coded bases
    #this is the compact representation used in npz files
    amap = {ch: i for i, ch in enumerate(alphabet)}
    out = np.fromiter((amap[ch] for ch in seq.strip().upper()), dtype=np.int8)
    return out

@dataclass(frozen=True)
class InitHaplotype:
    #one mutant haplotype within a cell type
    #proportion is conditional on being in the mutant fraction
    name: str
    seq: str
    prop_within_mutants: float

@dataclass(frozen=True)
class InitCellTypeParsed:
    #parsed cell type block from init fasta
    #contains WT plus optional mutant haplotypes
    name: str
    proportion: float
    mutant_fraction: float
    wt: str
    haps: List[InitHaplotype]

def parse_init_fasta_multi(path: str) -> Tuple[str, int, List[InitCellTypeParsed]]:
    #parses init fasta into structured cell types
    #this keeps biological setup separate from simulation internals
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    meta_global: Dict[str, str] = {}
    celltypes: List[InitCellTypeParsed] = []

    current_ct_name: Optional[str] = None
    current_ct_attrs: Dict[str, str] = {}
    current_records: List[Tuple[str, Dict[str, str], str]] = []

    current_id: Optional[str] = None
    current_attrs: Dict[str, str] = {}
    current_seq_parts: List[str] = []

    def flush_record() -> None:
        #closes the current FASTA record
        #sequence lines are accumulated until the next header/celltype
        nonlocal current_id, current_attrs, current_seq_parts, current_records
        if current_id is None:
            return
        seq = "".join(current_seq_parts).replace(" ", "").strip().upper()
        if not seq:
            raise ValueError(f"empty sequence for record {current_id}")
        current_records.append((current_id, current_attrs, seq))
        current_id = None
        current_attrs = {}
        current_seq_parts = []

    def flush_celltype() -> None:
        #closes the current celltype block
        #validates metadata and normalizes mutant haplotype proportions
        nonlocal current_ct_name, current_ct_attrs, current_records

        if current_ct_name is None:
            return

        if "length" not in meta_global:
            raise ValueError("missing #length:")
        if "alphabet" not in meta_global:
            raise ValueError("missing #alphabet:")

        length = int(meta_global["length"])
        alphabet = _normalize_alphabet(meta_global["alphabet"])

        if "proportion" not in current_ct_attrs:
            raise ValueError(f"celltype {current_ct_name} missing proportion=")
        if "mutant_fraction" not in current_ct_attrs:
            raise ValueError(f"celltype {current_ct_name} missing mutant_fraction=")

        proportion = float(current_ct_attrs["proportion"])
        mutant_fraction = float(current_ct_attrs["mutant_fraction"])

        #each cell type needs exactly one WT reference haplotype
        wt = [r for r in current_records if r[0] == "WT"]
        if len(wt) != 1:
            raise ValueError(f"expected exactly one WT in {current_ct_name}")
        wt_seq = wt[0][2]
        if len(wt_seq) != length:
            raise ValueError("WT length mismatch")

        #validate alphabet before converting to arrays
        allowed = set(alphabet)
        for rid, _attrs, seq in current_records:
            bad = {ch for ch in seq if ch not in allowed}
            if bad:
                raise ValueError(f"{rid} has invalid bases {sorted(bad)}")

        haps: List[InitHaplotype] = []
        for rid, attrs, seq in current_records:
            if rid == "WT":
                continue
            if "proportion_within_mutants" not in attrs:
                raise ValueError(f"{rid} missing proportion_within_mutants")
            p = float(attrs["proportion_within_mutants"])
            haps.append(InitHaplotype(rid, seq, p))

        #if mutant fraction exists, there must be mutant haplotypes to allocate it to
        if mutant_fraction > 0 and not haps:
            raise ValueError(f"mutant_fraction>0 but no mutant haps in {current_ct_name}")

        #normalize mutant haplotypes within the mutant fraction
        if haps:
            s = sum(h.prop_within_mutants for h in haps)
            if s <= 0:
                raise ValueError("sum proportion_within_mutants <= 0")
            haps = [InitHaplotype(h.name, h.seq, h.prop_within_mutants / s) for h in haps]

        celltypes.append(
            InitCellTypeParsed(
                name=current_ct_name,
                proportion=proportion,
                mutant_fraction=mutant_fraction,
                wt=wt_seq,
                haps=haps,
            )
        )

        #reset state for next celltype block
        current_ct_name = None
        current_ct_attrs = {}
        current_records = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("#celltype:"):
                #new celltype starts, so close previous record/block first
                flush_record()
                flush_celltype()

                tail = line[len("#celltype:"):].strip()
                parts = tail.split(None, 1)
                current_ct_name = parts[0].strip()
                rest = parts[1] if len(parts) > 1 else ""
                current_ct_attrs = _parse_header_kv(rest)
                continue

            if line.startswith("#"):
                #global metadata such as length/alphabet
                line2 = line[1:].strip()
                if ":" in line2:
                    k, v = line2.split(":", 1)
                    meta_global[k.strip().lower()] = v.strip()
                continue

            if line.startswith(">"):
                #new haplotype record inside current celltype
                flush_record()
                header = line[1:].strip()
                parts = header.split(None, 1)
                current_id = parts[0].strip()
                tail = parts[1] if len(parts) > 1 else ""
                current_attrs = _parse_header_kv(tail)
                continue

            #sequence lines can be multi-line, so accumulate them
            current_seq_parts.append(line)

    #close final record/celltype after file ends
    flush_record()
    flush_celltype()

    if not celltypes:
        raise ValueError("no celltypes found")

    alphabet = _normalize_alphabet(meta_global["alphabet"])
    length = int(meta_global["length"])

    #normalize celltype proportions across all celltypes
    s = sum(ct.proportion for ct in celltypes)
    if s <= 0:
        raise ValueError("sum of celltype proportions <= 0")
    celltypes = [
        InitCellTypeParsed(ct.name, ct.proportion / s, ct.mutant_fraction, ct.wt, ct.haps)
        for ct in celltypes
    ]

    return alphabet, length, celltypes

def write_init_npz(init_fasta_path: str, out_path: str) -> None:
    #converts parsed init fasta into compressed arrays
    #this is basically the simulation-ready representation of starting state
    alphabet, length, celltypes = parse_init_fasta_multi(init_fasta_path)
    A = len(alphabet)
    C = len(celltypes)

    #Hmax pads celltypes to a common haplotype dimension
    hap_counts = np.array([1 + len(ct.haps) for ct in celltypes], dtype=int)
    Hmax = int(hap_counts.max())

    seqs = np.zeros((C, Hmax, length), dtype=np.int8)
    probs = np.zeros((C, Hmax), dtype=float)

    ct_names = np.array([ct.name for ct in celltypes], dtype="U")
    ct_props = np.array([ct.proportion for ct in celltypes], dtype=float)
    mutant_fracs = np.array([ct.mutant_fraction for ct in celltypes], dtype=float)

    for i, ct in enumerate(celltypes):
        #hap 0 is always WT
        seqs[i, 0, :] = _seq_to_idx(ct.wt, alphabet)
        probs[i, 0] = 1.0 - ct.mutant_fraction

        #mutant haplotypes occupy later slots
        for j, h in enumerate(ct.haps, start=1):
            seqs[i, j, :] = _seq_to_idx(h.seq, alphabet)
            probs[i, j] = ct.mutant_fraction * float(h.prop_within_mutants)

        #renormalize to guard against imperfect input proportions
        s = probs[i, :hap_counts[i]].sum()
        if s <= 0:
            raise ValueError("init hap probs sum <= 0")
        probs[i, :hap_counts[i]] /= s

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        alphabet=alphabet,
        length=length,
        celltype_names=ct_names,
        celltype_proportions=ct_props,
        mutant_fractions=mutant_fracs,
        seqs=seqs,
        hap_probs=probs,
        hap_counts=hap_counts,
    )
    print("Wrote:", out_path)

def parse_mutation_tracks_dense(path: str, alphabet: str, length: int) -> Tuple[np.ndarray, np.ndarray]:
    #parses mutation track text into dense mutation-rate arrays
    #mu is mutation probability/rate, to_probs is destination-base distribution
    A = len(alphabet)
    amap = {ch: i for i, ch in enumerate(alphabet)}
    mu = np.zeros((length, A), dtype=float)
    to_probs = np.zeros((length, A, A), dtype=float)

    #default transition is identity unless overwritten
    for p in range(length):
        for b in range(A):
            to_probs[p, b, b] = 1.0

    meta: Dict[str, str] = {}

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("#"):
                #metadata can cross-check length/alphabet against init
                line2 = line[1:].strip()
                if ":" in line2:
                    k, v = line2.split(":", 1)
                    meta[k.strip().lower()] = v.strip()
                continue

            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"bad mutation line: {line}")

            pos1 = int(parts[0])
            fb = parts[1].upper()
            mu_val = float(parts[2])
            specs = parts[3:]

            if not (1 <= pos1 <= length):
                raise ValueError("pos out of range")
            if fb not in amap:
                raise ValueError(f"from_base {fb} not in alphabet")
            if mu_val < 0:
                raise ValueError("mu < 0")

            p0 = pos1 - 1
            fbi = amap[fb]
            mu[p0, fbi] = mu_val

            #parse destination base probabilities like A:0.5 G:0.5
            probs = np.zeros((A,), dtype=float)
            for sp in specs:
                if ":" not in sp:
                    raise ValueError(f"bad to:prob {sp}")
                tb, pr = sp.split(":", 1)
                tb = tb.upper()
                if tb not in amap:
                    raise ValueError(f"to_base {tb} not in alphabet")
                probs[amap[tb]] += float(pr)

            s = probs.sum()
            if s <= 0:
                raise ValueError("to_probs sum <= 0")
            to_probs[p0, fbi, :] = probs / s

    #metadata consistency checks
    if "length" in meta and int(meta["length"]) != length:
        raise ValueError("mutation file length != init length")
    if "alphabet" in meta and _normalize_alphabet(meta["alphabet"]) != alphabet:
        raise ValueError("mutation file alphabet != init alphabet")

    return mu, to_probs

def write_mutation_npz(mutation_txt: str, init_npz: str, out_path: str) -> None:
    #loads init metadata and writes mutation model as npz
    #keeps mutation tracks aligned to init alphabet/length
    init = np.load(init_npz, allow_pickle=False)
    alphabet = str(init["alphabet"])
    length = int(init["length"])
    mu, to_probs = parse_mutation_tracks_dense(mutation_txt, alphabet, length)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, alphabet=alphabet, length=length, mu=mu, to_probs=to_probs)
    print("Wrote:", out_path)

def parse_selection_dense(path: str, alphabet: str, length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #parses selection file into dense effect/threshold arrays
    #effects are additive, thresholds are optional per base/site
    A = len(alphabet)
    effects = np.zeros((length, A), dtype=float)
    thresholds = np.full((length, A), np.nan, dtype=float)

    meta: Dict[str, str] = {}

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("#"):
                #metadata again mainly serves as a consistency check
                line2 = line[1:].strip()
                if ":" in line2:
                    k, v = line2.split(":", 1)
                    meta[k.strip().lower()] = v.strip()
                continue

            parts = line.split()
            if len(parts) != 1 + A:
                raise ValueError(f"Expected {1+A} fields but got {len(parts)}: {line}")

            pos1 = int(parts[0])
            if not (1 <= pos1 <= length):
                raise ValueError("pos out of range")
            pos0 = pos1 - 1

            for bi, tok in enumerate(parts[1:]):
                t = tok.strip()

                #token can be effect only, or effect,threshold
                if "," in t:
                    a, b = t.split(",", 1)
                    eff = float(a)
                    thr = float(b)
                    effects[pos0, bi] = eff
                    thresholds[pos0, bi] = thr
                else:
                    effects[pos0, bi] = float(t)

    #positions stores only sites with nonzero effects or finite thresholds
    pos_mask = (effects != 0.0).any(axis=1) | np.isfinite(thresholds).any(axis=1)
    positions = np.where(pos_mask)[0].astype(int)

    if "length" in meta and int(meta["length"]) != length:
        raise ValueError(f"{path}: length != init length")
    if "alphabet" in meta:
        if _normalize_alphabet(meta["alphabet"]) != alphabet:
            raise ValueError(f"{path}: alphabet != init alphabet")

    return effects, thresholds, positions

def write_selection_npz(selection_txt: str, init_npz: str, out_path: str) -> None:
    #loads init metadata and writes one selection track as npz
    #selection arrays stay aligned to the same alphabet/length convention
    init = np.load(init_npz, allow_pickle=False)
    alphabet = str(init["alphabet"])
    length = int(init["length"])

    effects, thresholds, positions = parse_selection_dense(selection_txt, alphabet, length)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        alphabet=alphabet,
        length=length,
        effects=effects,
        thresholds=thresholds,
        positions=positions,
    )
    print("Wrote:", out_path)

#PLEASE DEFINE INPUT FILES HERE
def main() -> None:
    #main conversion pipeline
    init_fasta = "input_files/mtDNA_init.fasta"
    init_npz = "input_files/init.npz"

    #init has to be converted first because other files use its alphabet/length
    write_init_npz(init_fasta, init_npz)

    mut_txt = "input_files/mutation_tracks.txt"
    write_mutation_npz(mut_txt, init_npz, "input_files/mutation_tracks.npz")

    #selection files are optional and converted only if present
    sels = [
        ("input_files/mito_dup_selection.txt", "input_files/mito_dup_selection.npz"),
        ("input_files/mito_decay_selection.txt", "input_files/mito_decay_selection.npz"),
        ("input_files/cell_dup_selection.txt", "input_files/cell_dup_selection.npz"),
        ("input_files/cell_decay_selection.txt", "input_files/cell_decay_selection.npz"),
    ]

    for src, dst in sels:
        if os.path.exists(src):
            write_selection_npz(src, init_npz, dst)

if __name__ == "__main__":
    #entry point
    main()
