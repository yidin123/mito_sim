# sim_core.py:
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from config import SimConfig
from init_fasta import InitSpec, InitCellType
from mutation_tracks import MutationModel
from selection_tracks import SelectionModel

#set up non-frozen (adjustable mid-run) dataclasses for mtDNA and cells
@dataclass
class MtDNA:
    seq: str #sequence
    age: float = 0.0 #"age"

@dataclass
class Cell:
    mtdna: List[MtDNA] #list of mtdna
    age: float = 0.0 #"age"

#population of cells under study
Population = List[Cell]

"""
helper function to assign mtdna indices to specified haplotype distributions
sums up to exactly mtcn and leftover copies (since not all haplotype proportions will adhere to certain mtdna counts well) assigned to biggest fractions
outputs array of counts
"""
def deterministic_counts(mtcn: int, probs: np.ndarray) -> np.ndarray:
    #convert input probs to np float array
    probs = np.asarray(probs, dtype=float)
    #normalize probabilities
    probs = probs / probs.sum()

    #compute ideal allocations (exact fractional target)
    raw = mtcn * probs
    #convert to floor (base integer counts)
    counts = np.floor(raw).astype(int)

    #compute how many copies are still not allocated
    remainder = mtcn - counts.sum()
    #if some are not allocated yet
    if remainder > 0:
        #computes fractional parts (eg. largest fraction culled by flooring)
        frac = raw - counts
        #give the leftover copies to the biggest fractional parts
        idx = np.argsort(frac)[::-1]
        counts[idx[:remainder]] += 1

    #return final counts
    return counts

"""
helper function to do weighted random sampling from weights
picks index (int) with probability proportional to w[i]
"""
def _pick_index_by_weights(w: np.ndarray, rng: np.random.Generator) -> int:
    #take in array of weights and check if empty or not
    n = int(len(w))
    if n <= 0:
        raise ValueError("Cannot pick from empty log-weight array.")

    #convert to float, replace outliers with 0
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, 0.0)
    #compute total weight
    wsum = float(w.sum())

    #if weights are broken, pick uniform random index
    if not np.isfinite(wsum) or wsum <= 0.0:
        return int(rng.integers(0, n))

    #normalize weights to probabilities
    p = w / wsum
    #random draw from n based on p probabilities
    return int(rng.choice(n, p=p))

"""
returns relative selection factor based on ordering
if above mean, then higher weight
if below mean, then lower weight
scaled to be nonnegative
"""
def _relative_factor_from_scores(x: np.ndarray, strength: float, eps: float = 1e-12) -> np.ndarray:
    #specify baseline 
    baseline = 1.0 
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)

    #mean-center
    x = x - float(x.mean())

    #linear scoring with baseline (prevents zero collapse)
    s = baseline + float(strength) * x

    #ensure non-negativity
    return np.maximum(s, eps)

"""
helper function to take in a DNA sequence and randomly mutate positions according to mutation model
occurs for ONE duplication event
for each position i, IF there is a mutation rule, with probability mutation rate specified, change
base to one of the specified possible to_bases/after states
multiple mutations can occur per duplication
"""
def mutate_on_duplication(seq: str, model: Optional[MutationModel], rng: np.random.Generator) -> str:
    #if no mutation model specified
    if model is None:
        return seq

    #if sequence length different from metadata specified in mutation model
    if len(seq) != model.length:
        raise ValueError(f"Sequence length {len(seq)} != mutation model length {model.length}")

    #convert input sequence into a list
    s = list(seq)
    for i, base in enumerate(s): #loop through every pos/base
        #check if relevant rule exists in model
        rule = model.rules.get((i, base))
        if rule is None:
            continue #skip here if no rules exist for pos/base

        #decompose rule into mutation rate and "to" bases
        mu, to_list = rule
        if mu <= 0.0:
            continue #no mutation possible

        #if mutation occurs
        if rng.random() < mu:
            #list possible to_bases and corresponding probabilities
            to_bases = [tb for tb, _ in to_list]
            probs = np.array([p for _, p in to_list], dtype=float)
            #select to_base based on probability
            new_base = str(rng.choice(to_bases, p=probs))
            s[i] = new_base

    #converts sequence back to string and returns
    return "".join(s)

"""
helper function for additive effects of mtdna base pairs on selection
sums base pair effects across sequence
unspecified sites contribute 0
"""
def _mtdna_effect_sum(seq: str, sel: Optional[SelectionModel]) -> float:
    #no selection model provided
    if sel is None:
        return 0.0
    #if metadata provided differs from sequence length
    if len(seq) != sel.length:
        raise ValueError(f"seq length {len(seq)} != selection length {sel.length}")

    s = 0.0 #initial selection sum
    #sum across base pairs
    for i, base in enumerate(seq):
        #get selection effect; if not provided then 0.0
        s += sel.effects.get((i, base), 0.0)

    #return sum / scaling factor
    return float(s) / 1000000.0

"""
helper function for additive effects of mtdna base pairs on cell-level
inputs _mtdna_effect_sum (WITH CELL-LEVEL BP SELECTION MODEL) and sums up between mtdna in a cell
"""
def _cell_effect_sum(cell: Cell, sel: Optional[SelectionModel]) -> float:
    #no selection model provided
    if sel is None:
        return 0.0

    #sum up mtdna bp cell-level effects for all mtdna in cell, return
    return float(sum(_mtdna_effect_sum(m.seq, sel) for m in cell.mtdna))

"""
helper function for incrementing ages by some delta value input
"""
def increment_ages(pop: Population, delta: float) -> None:
    #no age increase
    if delta == 0.0:
        return
    #loops across cells and mtdna
    for cell in pop:
        cell.age += delta
        for m in cell.mtdna:
            m.age += delta

"""
at a specified (in config) transfer_event_prob each pre-mitotic timestep, we do:
    1. iterate through all cells and check if they engage in a transfer event. If so, pair them up
    with another randomly selected cell
    2. swap specific (in config) transfer_number random mtdna copies between each pair
swapping here is uniform random and preserves mtdna objects (sequence, age)
"""
def maybe_transfer_event(pop: Population, cfg: SimConfig, rng: np.random.Generator) -> None:
    #values for transfer_event_prob and transfer_number
    #interpret transfer_event_prob as per-cell probability of attempting to pair
    p = float(getattr(cfg, "transfer_rate", getattr(cfg, "transfer_event_prob", 0.0)))
    k = int(getattr(cfg, "transfer_number", 0))

    #if either one is 0 or under, end
    if p <= 0.0 or k <= 0:
        return
    #if we have less than 2 cells left:
    if len(pop) < 2:
        return

    #otherwise, we continue with the transfer event:
    #each cell independently has probability p to attempt pairing; each cell can pair at most once
    available = list(range(len(pop)))
    rng.shuffle(available)

    #walk through available cells; when a cell "fires", it pairs with a random remaining cell
    i = 0
    while i < len(available) - 1:
        a = available[i]

        #cell a does not attempt pairing this round
        if rng.random() >= p:
            i += 1
            continue

        #pick a random partner from the remaining unpaired cells
        j = int(rng.integers(i + 1, len(available)))
        b = available[j]

        #remove partner b from availability (swap-remove)
        available[j] = available[i + 1]
        available[i + 1] = b

        #the pair is (a, b); both are now considered paired for this round
        ca = pop[int(a)] #pair set 1
        cb = pop[int(b)] #pair set 2

        #acquire mtdna counts
        na = len(ca.mtdna)
        nb = len(cb.mtdna)
        #if cell in pair has 0 mtdna, stop here
        if na != 0 and nb != 0:
            #transfer/swap minimum of transfer_number/k or the # of mtdna in cell with fewer
            kk = min(k, na, nb)
            #if this is 0 or fewer, stop here
            if kk > 0:
                #randomly choosing kk # mtdna from each cell in pair for swapping
                sel_a = rng.choice(na, size=kk, replace=False)
                sel_b = rng.choice(nb, size=kk, replace=False)

                #swapping
                for ia, ib in zip(sel_a, sel_b):
                    ia = int(ia)
                    ib = int(ib)
                    ca.mtdna[ia], cb.mtdna[ib] = cb.mtdna[ib], ca.mtdna[ia]

        #advance past both paired cells
        i += 2

"""
initializes cell/mtdna population to specified ncells and mtcn
assigns haplotypes to mtdna at designated proportions
shuffles mtdna list order
assigns starting age to 0.0 for cells and mtdna
"""
def initialize_population(spec: InitSpec, ncells: int, mtcn: int, rng: np.random.Generator) -> Population:
    #constructs probability vector for drawing 1) wild-type and 2) individual mutant proportions
    probs = np.array(
        [1.0 - spec.mutant_fraction] +
        [spec.mutant_fraction * h.proportion_within_mutants for h in spec.haplotypes],
        dtype=float
    )

    #normalizes probabilities
    probs = probs / probs.sum()

    #constructs array of sequences (wt first, then haplotypes in order, same as probs)
    seqs = [spec.wt_sequence] + [h.sequence for h in spec.haplotypes]

    #creates cell population
    pop: Population = []
    #assign mtdna fractions to distribution in prob
    counts0 = deterministic_counts(mtcn, probs)

    #iterate through range of cells
    for _ in range(ncells):
        counts = counts0.copy() #all identical in composition

        #produce list of mtdna sequences based on haplotypes specified and counts
        mtdna_list: List[MtDNA] = []
        for seq, k in zip(seqs, counts):
            mtdna_list.extend(MtDNA(seq=seq, age=0.0) for _ in range(int(k)))

        #append completed cell to population
        pop.append(Cell(mtdna=mtdna_list, age=0.0))

    #return initialized population
    return pop

"""
initializes cell/mtdna population to specified ncells and mtcn
assigns starting "cell types" based on proportions in init fasta
assigns haplotypes to mtdna at designated proportions per cell type
shuffles cell order
assigns starting age to 0.0 for cells and mtdna
"""
def initialize_population_multi(celltypes: List[InitCellType], ncells: int, mtcn: int, rng: np.random.Generator) -> Population:
    #construct probability vector for drawing cell type assignments
    probs_ct = np.array([ct.proportion for ct in celltypes], dtype=float)
    probs_ct = probs_ct / probs_ct.sum()

    #assign cell type fractions to distribution in prob
    counts_ct = deterministic_counts(ncells, probs_ct)

    #creates cell population
    pop: Population = []

    #iterate through cell types
    for ct, n_ct in zip(celltypes, counts_ct):
        spec = ct.spec

        #constructs probability vector for drawing 1) wild-type and 2) individual mutant proportions
        probs = np.array(
            [1.0 - spec.mutant_fraction] +
            [spec.mutant_fraction * h.proportion_within_mutants for h in spec.haplotypes],
            dtype=float
        )

        #normalizes probabilities
        probs = probs / probs.sum()

        #constructs array of sequences (wt first, then haplotypes in order, same as probs)
        seqs = [spec.wt_sequence] + [h.sequence for h in spec.haplotypes]

        #assign mtdna fractions to distribution in prob
        counts0 = deterministic_counts(mtcn, probs)

        #iterate through range of cells
        for _ in range(int(n_ct)):
            counts = counts0.copy() #all identical in composition

            #produce list of mtdna sequences based on haplotypes specified and counts
            mtdna_list: List[MtDNA] = []
            for seq, k in zip(seqs, counts):
                mtdna_list.extend(MtDNA(seq=seq, age=0.0) for _ in range(int(k)))

            #append completed cell to population
            pop.append(Cell(mtdna=mtdna_list, age=0.0))

    #shuffles cell order
    rng.shuffle(pop)

    #return initialized population
    return pop

"""
runs moran process for mtDNA within a particular cell
first, one mtdna is picked to die (with age bias and decay selection)
then, another mtdna is picked to duplicate (with dup selection)
duplication produces a new copy (here, mutation may occur)
then, ages are refreshed using factors from config
"""
def moran_step_cell(
    cell: Cell,
    cfg: SimConfig,
    rng: np.random.Generator,
    mut_model: Optional[MutationModel],
    mtdna_dup_sel: Optional[SelectionModel] = None,
    mtdna_decay_sel: Optional[SelectionModel] = None,
) -> None:
    #store number of mtdna in cell
    n = len(cell.mtdna)
    if n <= 1: #1 or fewer mtdna left
        return

    #initialize death weights to 1/uniform
    w_death = np.ones(n, dtype=float)

    #if age bias is present
    if cfg.mtdna_death_age_bias != 1.0:
        #build array of mtdna ages
        ages = np.array([m.age for m in cell.mtdna], dtype=float)
        #produce weighted death by age using bias * age
        w_death *= np.power(float(cfg.mtdna_death_age_bias), ages)

    #if we have a decay selection model and strength is nonzero
    if mtdna_decay_sel is not None and float(getattr(cfg, "mtdna_decay_sel_strength", 0.0)) != 0.0:
        #store strength
        strength = float(cfg.mtdna_decay_sel_strength)
        #produce array of additive effect sums per mtdna
        eff = np.array([_mtdna_effect_sum(m.seq, mtdna_decay_sel) for m in cell.mtdna], dtype=float)
        #apply multiplicative linear fitness factor
        w_death *= _relative_factor_from_scores(eff, strength)

    #pick which mtdna dies by logweights
    dead = _pick_index_by_weights(w_death, rng)

    #indices for survivors
    idxs = [i for i in range(n) if i != dead]
    #initialize duplication weights to 1/uniform
    w_dup = np.ones(len(idxs), dtype=float)

    #if we have a duplication selection model and strength is nonzero
    if mtdna_dup_sel is not None and float(getattr(cfg, "mtdna_dup_sel_strength", 0.0)) != 0.0:
        #runs similarly to death selection
        strength = float(cfg.mtdna_dup_sel_strength)
        eff = np.array([_mtdna_effect_sum(cell.mtdna[i].seq, mtdna_dup_sel) for i in idxs], dtype=float)
        w_dup *= _relative_factor_from_scores(eff, strength)

    #pick which surviving mtdna is duplicated by logweights
    dup_local = _pick_index_by_weights(w_dup, rng)
    dup = idxs[dup_local]

    #acquire mtdna itself
    parent = cell.mtdna[dup]
    #set prior age to current parent age
    parent_age_before = parent.age

    #refresh parent's age after duplication
    parent.age = parent_age_before * cfg.mtdna_dup_refresh_parent

    #producing new copy
    new_seq = mutate_on_duplication(parent.seq, mut_model, rng) #mutation
    #creating new copy, with age refreshing
    new_copy = MtDNA(seq=new_seq, age=parent_age_before * cfg.mtdna_dup_refresh_child)
    #replacing dead copy with new
    cell.mtdna[dead] = new_copy

"""
runs one mtdna moran step per cell for the whole population
"""
def moran_step_population(
    pop: Population,
    cfg: SimConfig,
    rng: np.random.Generator,
    mut_model: Optional[MutationModel],
    mtdna_dup_sel: Optional[SelectionModel] = None,
    mtdna_decay_sel: Optional[SelectionModel] = None,
) -> None:
    for cell in pop:
        moran_step_cell(cell, cfg, rng, mut_model, mtdna_dup_sel, mtdna_decay_sel)

"""
remove specificied #mtdna copies per cell prior to mitosis
chosen using same mtdna decay weighting as normal mtdna death
leaves at least 1 mtdna per cell
"""
def apply_mtdna_loss(
    pop: Population,
    cfg: SimConfig,
    rng: np.random.Generator,
    k_loss: int,
    mtdna_decay_sel: Optional[SelectionModel] = None,
) -> None:
    #if no loss specified
    if k_loss <= 0:
        return

    #iterating through cells in population
    for cell in pop:
        #if only one copy left, then k = 0 and no loss occurs (does not delete all copies)
        k = min(k_loss, max(0, len(cell.mtdna) - 1))
        #iterates through k turns
        for _ in range(k):
            #initialize weights to 1 across mtdna
            n = len(cell.mtdna)
            w = np.ones(n, dtype=float)

            #if age bias is present
            if cfg.mtdna_death_age_bias != 1.0:
                #build array of mtdna ages
                ages = np.array([m.age for m in cell.mtdna], dtype=float)
                #produce weighted death by age using bias ** age
                w *= np.power(float(cfg.mtdna_death_age_bias), ages)

            #if we have a decay selection model and strength is nonzero
            if mtdna_decay_sel is not None and float(getattr(cfg, "mtdna_decay_sel_strength", 0.0)) != 0.0:
                #store strength
                strength = float(cfg.mtdna_decay_sel_strength)
                #produce array of additive effect sums per mtdna
                eff = np.array([_mtdna_effect_sum(m.seq, mtdna_decay_sel) for m in cell.mtdna], dtype=float)
                #apply multiplicative linear fitness factor
                w *= _relative_factor_from_scores(eff, strength)

            #pick index for death
            dead = _pick_index_by_weights(w, rng)
            #kill the mtdna index for this cell
            cell.mtdna.pop(dead)

"""
mitosis (mtdna replication + segregation, and cell age refresh)
mtdna are doubled by N sequential duplication events (N -> 2N)
each event picks one parent mtdna to duplicate using selection biases
remaining logic is similar to other mtdna duplication (mutation can occur, age refreshing)
then, 2N mtdna are shuffled and split into 2 daughter cells
daughter cell ages are refreshed randomly via A, B factors relative to parent
"""
def mitotic_division(
    parent: Cell,
    cfg: SimConfig,
    rng: np.random.Generator,
    mut_model: Optional[MutationModel],
    mtdna_dup_sel: Optional[SelectionModel] = None,
) -> Tuple[Cell, Cell]:
    #take existing pool of mtdna and append to list (half of 2N goal)
    replicated: List[MtDNA] = list(parent.mtdna)
    #store length
    n0 = len(replicated)
    #target 2N
    target = 2 * n0

    #run N rounds of duplication until we reach 2N total mtdna
    while len(replicated) < target:
        #initialize log weights of n length
        n = len(replicated)
        w_dup = np.ones(n, dtype=float)

        #if we have a duplication selection model and strength is nonzero
        if mtdna_dup_sel is not None and float(getattr(cfg, "mtdna_dup_sel_strength", 0.0)) != 0.0:
            #store strength
            strength = float(cfg.mtdna_dup_sel_strength)
            #produce array of additive effect sums per mtdna
            eff = np.array([_mtdna_effect_sum(m.seq, mtdna_dup_sel) for m in replicated], dtype=float)
            #apply multiplicative linear fitness factor
            w_dup *= _relative_factor_from_scores(eff, strength)

        #pick index for duplication
        dup_idx = _pick_index_by_weights(w_dup, rng)
        parent_m = replicated[dup_idx]

        #refresh chosen parent's age, create new copy, append to list
        age_before = parent_m.age
        parent_m.age = age_before * cfg.mtdna_dup_refresh_parent
        new_seq = mutate_on_duplication(parent_m.seq, mut_model, rng)
        replicated.append(MtDNA(seq=new_seq, age=age_before * cfg.mtdna_dup_refresh_child))

    #random segregation into daughter cells (by equal split)
    rng.shuffle(replicated)
    half = len(replicated) // 2
    d1_mtdna = replicated[:half]
    d2_mtdna = replicated[half:]

    #cell age refreshing
    a = cfg.cell_div_refresh_a
    b = cfg.cell_div_refresh_b
    #assign randomly
    f1, f2 = (a, b) if rng.random() < 0.5 else (b, a)

    #refresh ages of daughters
    parent_age = parent.age
    d1 = Cell(mtdna=d1_mtdna, age=parent_age * f1)
    d2 = Cell(mtdna=d2_mtdna, age=parent_age * f2)

    #return tuple
    return d1, d2

"""
permanently remove specified # cells (before mitosis)
chosen using same cell death weighting as in mitotic death (age bias and decay selection)
leaves at least 1 cell left
"""
def apply_cell_loss(
    pop: Population,
    cfg: SimConfig,
    rng: np.random.Generator,
    k_loss: int,
    cell_decay_sel: Optional[SelectionModel] = None,
) -> Population:
    #if no loss specified
    if k_loss <= 0:
        return pop

    #prevent loss if only one cell remaining
    k_loss = min(k_loss, max(0, len(pop) - 1))
    if k_loss <= 0:
        return pop

    #iterate k_loss times
    for _ in range(k_loss):
        #create array of weights
        n = len(pop)
        w_death = np.ones(n, dtype=float)

        #if age bias is present
        if cfg.cell_death_age_bias != 1.0:
            #build array of cell ages
            ages = np.array([c.age for c in pop], dtype=float)
            #produce weighted death by age using bias ** age
            w_death *= np.power(float(cfg.cell_death_age_bias), ages)

        #if we have a decay selection model and strength is nonzero
        if cell_decay_sel is not None and float(getattr(cfg, "cell_decay_sel_strength", 0.0)) != 0.0:
            #store strength
            strength = float(cfg.cell_decay_sel_strength)
            #produce array of additive effect sums per cell
            scores = np.array([_cell_effect_sum(c, cell_decay_sel) for c in pop], dtype=float)
            #apply multiplicative linear fitness factor
            w_death *= _relative_factor_from_scores(scores, strength)

        #pick index for death
        dead_idx = _pick_index_by_weights(w_death, rng)
        #kill cell
        pop.pop(dead_idx)

    #return new cell population
    return pop

"""
moran-style cell turnover during mitosis
repeats len(pop) (# cells) times
increments ages for all cells and mtdna
chooses a cell to die, and a different cell to divide (with relevant biases/selection)
dividing cell produces 2 daughters
replace parent and dead slot with daughters
"""
def mitotic_generation(
    pop: Population,
    cfg: SimConfig,
    rng: np.random.Generator,
    mut_model: Optional[MutationModel],
    dup_sel: Optional[SelectionModel] = None,
    decay_sel: Optional[SelectionModel] = None,
    cell_dup_sel: Optional[SelectionModel] = None,
    cell_decay_sel: Optional[SelectionModel] = None,
) -> Population:
    #read in number of cells
    ncells = len(pop)
    #if one or fewer cells left, dont do anything
    if ncells <= 1:
        return pop

    #iterate for all cells
    for _ in range(ncells):
        #increment ages
        increment_ages(pop, cfg.mitotic_age_step)

        #create array of weights
        n = len(pop)
        w_death = np.ones(n, dtype=float)

        #if age bias is present
        if cfg.cell_death_age_bias != 1.0:
            #build array of cell ages
            ages = np.array([c.age for c in pop], dtype=float)
            #produce weighted death by age using bias ** age
            w_death *= np.power(float(cfg.cell_death_age_bias), ages)

        #if we have a decay selection model and strength is nonzero
        if cell_decay_sel is not None and float(getattr(cfg, "cell_decay_sel_strength", 0.0)) != 0.0:
            #store strength
            strength = float(cfg.cell_decay_sel_strength)
            #produce array of additive effect sums per cell
            scores = np.array([_cell_effect_sum(c, cell_decay_sel) for c in pop], dtype=float)
            #apply multiplicative linear fitness factor
            w_death *= _relative_factor_from_scores(scores, strength)

        #pick index for death
        dead_idx = _pick_index_by_weights(w_death, rng)

        #restrict to survivors
        idxs = [i for i in range(n) if i != dead_idx]
        #create array of weights
        w_dup = np.ones(len(idxs), dtype=float)

        #if selection model is present and strength nonzero
        if cell_dup_sel is not None and float(getattr(cfg, "cell_dup_sel_strength", 0.0)) != 0.0:
            #logic similar to decay selection model
            strength = float(cfg.cell_dup_sel_strength)
            scores = np.array([_cell_effect_sum(pop[i], cell_dup_sel) for i in idxs], dtype=float)
            w_dup *= _relative_factor_from_scores(scores, strength)

        #pick index for duplication
        parent_local = _pick_index_by_weights(w_dup, rng)
        parent_idx = idxs[parent_local]

        #acquire specific cell
        parent_cell = pop[parent_idx]

        #mitotic division
        d1, d2 = mitotic_division(parent_cell, cfg, rng, mut_model, mtdna_dup_sel=dup_sel)

        #assign daughters to parent and dead spots
        pop[dead_idx] = d1
        pop[parent_idx] = d2

    #return population
    return pop
