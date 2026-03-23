from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np

from config import SimConfig
from init_fasta import InitCellType, InitSpec
from mutation_tracks import MutationModel
from selection_tracks import SelectionModel


try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco


@njit(cache=True)
def _fill_ones_nb(out: np.ndarray, n: int) -> None:
    for i in range(n):
        out[i] = 1.0


@njit(cache=True)
def _fill_mt_age_weights_nb(
    out: np.ndarray,
    mt_birth_row: np.ndarray,
    M: int,
    sim_time: float,
    age_bias: float,
) -> None:
    if age_bias == 1.0:
        for i in range(M):
            out[i] = 1.0
    else:
        for i in range(M):
            out[i] = age_bias ** (sim_time - mt_birth_row[i])


@njit(cache=True)
def _fill_cell_age_weights_nb(
    out: np.ndarray,
    cell_birth: np.ndarray,
    n: int,
    sim_time: float,
    age_bias: float,
) -> None:
    if age_bias == 1.0:
        for i in range(n):
            out[i] = 1.0
    else:
        for i in range(n):
            out[i] = age_bias ** (sim_time - cell_birth[i])


@njit(cache=True)
def _apply_relative_scores_inplace_nb(
    w: np.ndarray,
    scores: np.ndarray,
    n: int,
    strength: float,
    eps: float = 1e-12,
) -> None:
    ssum = 0.0
    cnt = 0
    for i in range(n):
        x = scores[i]
        if np.isfinite(x):
            ssum += x
            cnt += 1
    mean = ssum / cnt if cnt > 0 else 0.0

    for i in range(n):
        x = scores[i]
        if not np.isfinite(x):
            x = 0.0
        fac = 1.0 + strength * (x - mean)
        if fac < eps:
            fac = eps
        w[i] *= fac


@njit(cache=True)
def _pick_index_by_weights_nb(w: np.ndarray, n: int, u: float) -> int:
    total = 0.0
    for i in range(n):
        wi = w[i]
        if not np.isfinite(wi) or wi < 0.0:
            wi = 0.0
        total += wi

    if (not np.isfinite(total)) or (total <= 0.0):
        return min(int(u * n), n - 1)

    target = u * total
    c = 0.0
    for i in range(n):
        wi = w[i]
        if not np.isfinite(wi) or wi < 0.0:
            wi = 0.0
        c += wi
        if target < c:
            return i
    return n - 1


@njit(cache=True)
def _current_bases_for_positions_nb(
    reference: np.ndarray,
    parent_pos: np.ndarray,
    parent_alt: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    n = positions.shape[0]
    out = np.empty(n, dtype=np.int64)

    i = 0
    j = 0
    P = parent_pos.shape[0]

    while i < n:
        p = positions[i]
        while j < P and parent_pos[j] < p:
            j += 1
        if j < P and parent_pos[j] == p:
            out[i] = int(parent_alt[j])
        else:
            out[i] = int(reference[p])
        i += 1

    return out


@njit(cache=True)
def _merge_sparse_child_nb(
    reference: np.ndarray,
    parent_pos: np.ndarray,
    parent_alt: np.ndarray,
    mut_pos: np.ndarray,
    new_bases: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    P = parent_pos.shape[0]
    M = mut_pos.shape[0]

    out_pos = np.empty(P + M, dtype=np.int32)
    out_alt = np.empty(P + M, dtype=np.int8)
    n_out = 0

    i = 0
    j = 0

    while i < P and j < M:
        pp = parent_pos[i]
        mp = mut_pos[j]

        if pp < mp:
            out_pos[n_out] = pp
            out_alt[n_out] = parent_alt[i]
            n_out += 1
            i += 1
        elif mp < pp:
            nb = new_bases[j]
            if nb != reference[mp]:
                out_pos[n_out] = mp
                out_alt[n_out] = nb
                n_out += 1
            j += 1
        else:
            nb = new_bases[j]
            if nb != reference[mp]:
                out_pos[n_out] = mp
                out_alt[n_out] = nb
                n_out += 1
            i += 1
            j += 1

    while i < P:
        out_pos[n_out] = parent_pos[i]
        out_alt[n_out] = parent_alt[i]
        n_out += 1
        i += 1

    while j < M:
        mp = mut_pos[j]
        nb = new_bases[j]
        if nb != reference[mp]:
            out_pos[n_out] = mp
            out_alt[n_out] = nb
            n_out += 1
        j += 1

    return out_pos[:n_out], out_alt[:n_out]


@dataclass
class SparseHaplotype:
    pos: np.ndarray
    alt: np.ndarray
    score_cache: Dict[str, float] = field(default_factory=dict)


class HaplotypeBank:
    def __init__(self, reference: np.ndarray, alphabet: str):
        self.reference = np.asarray(reference, dtype=np.int8).copy()
        self.alphabet = alphabet
        self.length = int(self.reference.shape[0])

        self.haps: List[SparseHaplotype] = [
            SparseHaplotype(
                pos=np.empty(0, dtype=np.int32),
                alt=np.empty(0, dtype=np.int8),
                score_cache={}
            )
        ]
        self.key_to_id: Dict[Tuple[Tuple[int, int], ...], int] = {(): 0}

        self._ref_score_cache: Dict[str, float] = {}
        self._mut_model_cache: Dict[int, Dict[str, np.ndarray]] = {}

    def _key(self, pos: np.ndarray, alt: np.ndarray) -> Tuple[Tuple[int, int], ...]:
        return tuple((int(p), int(a)) for p, a in zip(pos.tolist(), alt.tolist()))

    def register_dense(self, seq: np.ndarray) -> int:
        seq = np.asarray(seq, dtype=np.int8)
        diff = np.flatnonzero(seq != self.reference).astype(np.int32)
        if diff.size == 0:
            return 0
        alt = seq[diff].astype(np.int8)
        return self._register_sparse(diff, alt)

    def _register_sparse(self, pos: np.ndarray, alt: np.ndarray) -> int:
        pos = np.asarray(pos, dtype=np.int32)
        alt = np.asarray(alt, dtype=np.int8)

        if pos.size == 0:
            return 0

        key = self._key(pos, alt)
        hid = self.key_to_id.get(key)
        if hid is not None:
            return hid

        hid = len(self.haps)
        self.haps.append(SparseHaplotype(pos=pos, alt=alt, score_cache={}))
        self.key_to_id[key] = hid
        return hid

    def materialize(self, hid: int) -> np.ndarray:
        out = self.reference.copy()
        hap = self.haps[hid]
        if hap.pos.size:
            out[hap.pos] = hap.alt
        return out

    def base_at(self, hid: int, pos: int) -> int:
        hap = self.haps[hid]
        idx = np.searchsorted(hap.pos, pos)
        if idx < hap.pos.size and int(hap.pos[idx]) == pos:
            return int(hap.alt[idx])
        return int(self.reference[pos])

    def _get_mut_cache(self, model: MutationModel) -> Dict[str, np.ndarray]:
        key = id(model)
        cached = self._mut_model_cache.get(key)
        if cached is not None:
            return cached

        ref_int = self.reference.astype(np.int64, copy=False)
        ref_mu = model.mu[np.arange(model.length, dtype=np.int64), ref_int]

        cached = {"ref_mu": ref_mu}
        self._mut_model_cache[key] = cached
        return cached

    def _current_bases_for_positions(self, hid: int, positions: np.ndarray) -> np.ndarray:
        if positions.size == 0:
            return np.empty(0, dtype=np.int64)
        hap = self.haps[hid]
        if hap.pos.size == 0:
            return self.reference[positions].astype(np.int64, copy=True)
        return _current_bases_for_positions_nb(self.reference, hap.pos, hap.alt, positions)

    def _child_from_parent_with_mutations(
        self,
        hid: int,
        mut_pos: np.ndarray,
        new_bases: np.ndarray
    ) -> int:
        if mut_pos.size == 0:
            return hid

        hap = self.haps[hid]
        out_pos, out_alt = _merge_sparse_child_nb(
            self.reference, hap.pos, hap.alt, mut_pos, new_bases
        )

        if out_pos.size == 0:
            return 0

        return self._register_sparse(out_pos, out_alt)

    def mutate_child(
        self,
        hid: int,
        model: Optional[MutationModel],
        rng: np.random.Generator
    ) -> int:
        if model is None:
            return hid

        ref_mu = self._get_mut_cache(model)["ref_mu"]
        hap = self.haps[hid]

        u = rng.random(model.length)
        hit = (u < ref_mu)

        if hap.pos.size:
            pos64 = hap.pos.astype(np.int64, copy=False)
            alt64 = hap.alt.astype(np.int64, copy=False)
            hit[hap.pos] = (u[hap.pos] < model.mu[pos64, alt64])

        if not np.any(hit):
            return hid

        mut_pos = np.flatnonzero(hit).astype(np.int32)
        cur_bases = self._current_bases_for_positions(hid, mut_pos)

        A = len(model.alphabet)
        new_bases = cur_bases.astype(np.int8, copy=True)

        for i, p in enumerate(mut_pos):
            cb = int(cur_bases[i])
            probs = model.to_probs[int(p), cb, :]
            s = float(probs.sum())
            if not np.isfinite(s) or s <= 0.0:
                nb = int(rng.integers(0, A))
            else:
                nb = int(rng.choice(A, p=(probs / s)))
            new_bases[i] = np.int8(nb)

        return self._child_from_parent_with_mutations(hid, mut_pos, new_bases)

    def _ref_score(self, sel: SelectionModel, cache_key: str) -> float:
        cached = self._ref_score_cache.get(cache_key)
        if cached is not None:
            return cached

        ref_bases = self.reference.astype(np.int64, copy=False)
        score = float(sel.effects[np.arange(sel.length), ref_bases].sum()) / 100.0
        self._ref_score_cache[cache_key] = score
        return score

    def hap_additive_score(self, hid: int, sel: SelectionModel, cache_key: str) -> float:
        hap = self.haps[hid]
        cached = hap.score_cache.get(cache_key)
        if cached is not None:
            return cached

        ref_score = self._ref_score(sel, cache_key)

        if hap.pos.size == 0:
            hap.score_cache[cache_key] = ref_score
            return ref_score

        delta = (
            sel.effects[hap.pos, hap.alt.astype(np.int64, copy=False)] -
            sel.effects[hap.pos, self.reference[hap.pos].astype(np.int64, copy=False)]
        ).sum() / 100.0

        score = ref_score + float(delta)
        hap.score_cache[cache_key] = score
        return score

    def score_many_into(
        self,
        hids: np.ndarray,
        sel: SelectionModel,
        cache_key: str,
        out: np.ndarray,
    ) -> np.ndarray:
        n = int(hids.size)
        if n == 0:
            return out[:0]

        get_score = self.hap_additive_score

        h0 = int(hids[0])
        same = True
        for i in range(1, n):
            if int(hids[i]) != h0:
                same = False
                break
        if same:
            s0 = get_score(h0, sel, cache_key)
            out[:n] = s0
            return out[:n]

        s0 = get_score(h0, sel, cache_key)
        h1 = -1
        s1 = 0.0
        i = 0
        while i < n:
            hid = int(hids[i])
            if hid == h0:
                out[i] = s0
            elif h1 != -1 and hid == h1:
                out[i] = s1
            elif h1 == -1:
                h1 = hid
                s1 = get_score(h1, sel, cache_key)
                out[i] = s1
            else:
                break
            i += 1

        if i == n:
            return out[:n]

        local_cache: Dict[int, float] = {h0: s0}
        if h1 != -1:
            local_cache[h1] = s1

        for j in range(i, n):
            hid = int(hids[j])
            s = local_cache.get(hid)
            if s is None:
                s = get_score(hid, sel, cache_key)
                local_cache[hid] = s
            out[j] = s
        return out[:n]

    def score_many(
        self,
        hids: np.ndarray,
        sel: SelectionModel,
        cache_key: str
    ) -> np.ndarray:
        out = np.empty(int(hids.size), dtype=float)
        return self.score_many_into(hids, sel, cache_key, out)
        

@dataclass
class Population:
    copy_hid: np.ndarray
    mt_birth: np.ndarray
    m_count: np.ndarray
    cell_birth: np.ndarray
    n_cells: int
    sim_time: float

    cell_dup_sum: Optional[np.ndarray] = None
    cell_decay_sum: Optional[np.ndarray] = None

    tmp_mt_w1: Optional[np.ndarray] = None
    tmp_mt_w2: Optional[np.ndarray] = None
    tmp_mt_scores: Optional[np.ndarray] = None
    tmp_cell_w1: Optional[np.ndarray] = None
    tmp_cell_w2: Optional[np.ndarray] = None


def deterministic_counts(n: int, probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    raw = n * probs
    counts = np.floor(raw).astype(np.int32)
    rem = n - int(counts.sum())
    if rem > 0:
        frac = raw - counts
        idx = np.argsort(frac)[::-1]
        counts[idx[:rem]] += 1
    return counts


def _pick_index_by_weights_prefix(w: np.ndarray, n: int, rng: np.random.Generator) -> int:
    return _pick_index_by_weights_nb(w, n, float(rng.random()))


def mt_ages(pop: Population, ci: int) -> np.ndarray:
    M = int(pop.m_count[ci])
    return pop.sim_time - pop.mt_birth[ci, :M]


def cell_ages(pop: Population) -> np.ndarray:
    return pop.sim_time - pop.cell_birth[:pop.n_cells]


def _fill_scores_for_cell(
    pop: Population,
    ci: int,
    bank: HaplotypeBank,
    sel: SelectionModel,
    cache_key: str
) -> np.ndarray:
    M = int(pop.m_count[ci])
    hids = pop.copy_hid[ci, :M]
    bank.score_many_into(hids, sel, cache_key, pop.tmp_mt_scores)
    return pop.tmp_mt_scores[:M]


def _recompute_cell_sum(
    pop: Population,
    ci: int,
    bank: HaplotypeBank,
    sel: Optional[SelectionModel],
    cache_key: str
) -> float:
    if sel is None:
        return 0.0
    M = int(pop.m_count[ci])
    hids = pop.copy_hid[ci, :M]
    bank.score_many_into(hids, sel, cache_key, pop.tmp_mt_scores)
    return float(pop.tmp_mt_scores[:M].sum())


def initialize_population_multi_fast(
    celltypes: List[InitCellType],
    ncells: int,
    mtcn: int,
    rng: np.random.Generator,
    birth_dtype=np.float32
) -> Tuple[Population, HaplotypeBank]:
    ref = np.asarray(celltypes[0].spec.hap_seqs[0], dtype=np.int8)
    alphabet = celltypes[0].spec.alphabet
    bank = HaplotypeBank(reference=ref, alphabet=alphabet)

    probs_ct = np.array([ct.proportion for ct in celltypes], dtype=float)
    counts_ct = deterministic_counts(ncells, probs_ct)

    copy_hid = np.zeros((ncells, mtcn), dtype=np.int32)
    mt_birth = np.zeros((ncells, mtcn), dtype=birth_dtype)
    m_count = np.full((ncells,), mtcn, dtype=np.int32)
    cell_birth = np.zeros((ncells,), dtype=birth_dtype)

    cursor = 0
    for ct, n_ct in zip(celltypes, counts_ct):
        spec: InitSpec = ct.spec
        hap_ids = np.array([bank.register_dense(hseq) for hseq in spec.hap_seqs], dtype=np.int32)
        hap_counts = deterministic_counts(mtcn, spec.hap_probs)

        template = np.empty((mtcn,), dtype=np.int32)
        idx = 0
        for hid, k in zip(hap_ids, hap_counts):
            k = int(k)
            template[idx:idx + k] = int(hid)
            idx += k

        if idx != mtcn:
            raise RuntimeError("init mtcn mismatch")

        for _ in range(int(n_ct)):
            copy_hid[cursor, :] = template
            cursor += 1

    perm = rng.permutation(ncells)

    tmp_mt_cap = 2 * mtcn
    pop = Population(
        copy_hid=copy_hid[perm],
        mt_birth=mt_birth[perm],
        m_count=m_count[perm],
        cell_birth=cell_birth[perm],
        n_cells=ncells,
        sim_time=0.0,
        cell_dup_sum=None,
        cell_decay_sum=None,
        tmp_mt_w1=np.empty(tmp_mt_cap, dtype=np.float64),
        tmp_mt_w2=np.empty(tmp_mt_cap, dtype=np.float64),
        tmp_mt_scores=np.empty(tmp_mt_cap, dtype=np.float64),
        tmp_cell_w1=np.empty(ncells, dtype=np.float64),
        tmp_cell_w2=np.empty(ncells, dtype=np.float64),
    )
    return pop, bank


def maybe_transfer_event(pop: Population, cfg: SimConfig, rng: np.random.Generator) -> None:
    p = float(getattr(cfg, "transfer_event_prob", 0.0))
    k = int(getattr(cfg, "transfer_number", 0))
    n = pop.n_cells
    if p <= 0.0 or k <= 0 or n < 2:
        return

    available = np.arange(n, dtype=np.int32)
    rng.shuffle(available)

    i = 0
    while i < n - 1:
        a = int(available[i])
        if rng.random() >= p:
            i += 1
            continue

        j = int(rng.integers(i + 1, n))
        b = int(available[j])

        available[j] = available[i + 1]
        available[i + 1] = b

        na = int(pop.m_count[a])
        nb = int(pop.m_count[b])
        kk = min(k, na, nb)
        if kk > 0:
            sel_a = rng.choice(na, size=kk, replace=False)
            sel_b = rng.choice(nb, size=kk, replace=False)

            tmp_h = pop.copy_hid[a, sel_a].copy()
            tmp_t = pop.mt_birth[a, sel_a].copy()

            pop.copy_hid[a, sel_a] = pop.copy_hid[b, sel_b]
            pop.mt_birth[a, sel_a] = pop.mt_birth[b, sel_b]

            pop.copy_hid[b, sel_b] = tmp_h
            pop.mt_birth[b, sel_b] = tmp_t

        i += 2


def moran_step_cell(
    pop: Population,
    ci: int,
    cfg: SimConfig,
    rng: np.random.Generator,
    bank: HaplotypeBank,
    mut_model: Optional[MutationModel],
    mtdna_dup_sel: Optional[SelectionModel] = None,
    mtdna_decay_sel: Optional[SelectionModel] = None,
    mito_dup_key: str = "mito_dup",
    mito_decay_key: str = "mito_decay",
) -> None:
    M = int(pop.m_count[ci])
    if M <= 1:
        return

    age_bias = float(cfg.mtdna_death_age_bias)
    use_decay_sel = (
        mtdna_decay_sel is not None and
        float(getattr(cfg, "mtdna_decay_sel_strength", 0.0)) != 0.0
    )
    use_dup_sel = (
        mtdna_dup_sel is not None and
        float(getattr(cfg, "mtdna_dup_sel_strength", 0.0)) != 0.0
    )

    if age_bias == 1.0 and not use_decay_sel:
        dead = int(rng.integers(0, M))
    else:
        w_death = pop.tmp_mt_w1[:M]
        _fill_mt_age_weights_nb(w_death, pop.mt_birth[ci], M, pop.sim_time, age_bias)

        if use_decay_sel:
            scores = _fill_scores_for_cell(pop, ci, bank, mtdna_decay_sel, mito_decay_key)
            _apply_relative_scores_inplace_nb(
                w_death, scores, M, float(cfg.mtdna_decay_sel_strength)
            )

        dead = _pick_index_by_weights_prefix(w_death, M, rng)

    if not use_dup_sel:
        u = int(rng.integers(0, M - 1))
        parent = u if u < dead else (u + 1)
    else:
        w_dup = pop.tmp_mt_w2[:M]
        _fill_ones_nb(w_dup, M)
        scores = _fill_scores_for_cell(pop, ci, bank, mtdna_dup_sel, mito_dup_key)
        _apply_relative_scores_inplace_nb(
            w_dup, scores, M, float(cfg.mtdna_dup_sel_strength)
        )
        w_dup[dead] = 0.0
        parent = _pick_index_by_weights_prefix(w_dup, M, rng)

    parent_hid = int(pop.copy_hid[ci, parent])
    parent_birth = float(pop.mt_birth[ci, parent])
    parent_age = pop.sim_time - parent_birth

    new_parent_age = parent_age * float(cfg.mtdna_dup_refresh_parent)
    pop.mt_birth[ci, parent] = pop.sim_time - new_parent_age

    child_hid = bank.mutate_child(parent_hid, mut_model, rng)
    child_age = parent_age * float(cfg.mtdna_dup_refresh_child)

    pop.copy_hid[ci, dead] = child_hid
    pop.mt_birth[ci, dead] = pop.sim_time - child_age


def moran_step_population(
    pop: Population,
    cfg: SimConfig,
    rng: np.random.Generator,
    bank: HaplotypeBank,
    mut_model: Optional[MutationModel],
    mtdna_dup_sel: Optional[SelectionModel] = None,
    mtdna_decay_sel: Optional[SelectionModel] = None,
) -> None:
    for ci in range(pop.n_cells):
        moran_step_cell(
            pop, ci, cfg, rng, bank, mut_model,
            mtdna_dup_sel=mtdna_dup_sel,
            mtdna_decay_sel=mtdna_decay_sel
        )


def apply_mtdna_loss(
    pop: Population,
    cfg: SimConfig,
    rng: np.random.Generator,
    bank: HaplotypeBank,
    k_loss: int,
    mtdna_decay_sel: Optional[SelectionModel] = None,
    mito_decay_key: str = "mito_decay",
) -> None:
    if k_loss <= 0:
        return

    age_bias = float(cfg.mtdna_death_age_bias)
    use_decay_sel = (
        mtdna_decay_sel is not None and
        float(getattr(cfg, "mtdna_decay_sel_strength", 0.0)) != 0.0
    )

    for ci in range(pop.n_cells):
        M = int(pop.m_count[ci])
        k = min(k_loss, max(0, M - 1))
        for _ in range(k):
            M = int(pop.m_count[ci])
            if M <= 1:
                break

            if age_bias == 1.0 and not use_decay_sel:
                dead = int(rng.integers(0, M))
            else:
                w = pop.tmp_mt_w1[:M]
                _fill_mt_age_weights_nb(w, pop.mt_birth[ci], M, pop.sim_time, age_bias)

                if use_decay_sel:
                    scores = _fill_scores_for_cell(pop, ci, bank, mtdna_decay_sel, mito_decay_key)
                    _apply_relative_scores_inplace_nb(
                        w, scores, M, float(cfg.mtdna_decay_sel_strength)
                    )

                dead = _pick_index_by_weights_prefix(w, M, rng)

            last = M - 1
            if dead != last:
                pop.copy_hid[ci, dead] = pop.copy_hid[ci, last]
                pop.mt_birth[ci, dead] = pop.mt_birth[ci, last]
            pop.m_count[ci] = last


def apply_cell_loss(
    pop: Population,
    cfg: SimConfig,
    rng: np.random.Generator,
    bank: HaplotypeBank,
    k_loss: int,
    cell_decay_sel: Optional[SelectionModel] = None,
    cell_decay_key: str = "cell_decay",
) -> Population:
    if k_loss <= 0:
        return pop

    k_loss = min(k_loss, max(0, pop.n_cells - 1))
    age_bias = float(cfg.cell_death_age_bias)
    use_cell_decay = (
        cell_decay_sel is not None and
        float(getattr(cfg, "cell_decay_sel_strength", 0.0)) != 0.0
    )

    for _ in range(k_loss):
        n = pop.n_cells
        if n <= 1:
            break

        if age_bias == 1.0 and not use_cell_decay:
            dead = int(rng.integers(0, n))
        else:
            w = pop.tmp_cell_w1[:n]
            _fill_cell_age_weights_nb(w, pop.cell_birth, n, pop.sim_time, age_bias)

            if use_cell_decay:
                if cfg.thresholding:
                    raise NotImplementedError("Fast path cell-loss selection assumes thresholding=False")
                if pop.cell_decay_sum is None:
                    pop.cell_decay_sum = np.array([
                        _recompute_cell_sum(pop, i, bank, cell_decay_sel, cell_decay_key)
                        for i in range(n)
                    ], dtype=float)
                _apply_relative_scores_inplace_nb(
                    w, pop.cell_decay_sum, n, float(cfg.cell_decay_sel_strength)
                )

            dead = _pick_index_by_weights_prefix(w, n, rng)

        last = n - 1
        if dead != last:
            pop.copy_hid[dead, :] = pop.copy_hid[last, :]
            pop.mt_birth[dead, :] = pop.mt_birth[last, :]
            pop.m_count[dead] = pop.m_count[last]
            pop.cell_birth[dead] = pop.cell_birth[last]
            if pop.cell_dup_sum is not None:
                pop.cell_dup_sum[dead] = pop.cell_dup_sum[last]
            if pop.cell_decay_sum is not None:
                pop.cell_decay_sum[dead] = pop.cell_decay_sum[last]

        pop.n_cells = last

    return pop


def mitotic_division(
    parent_h: np.ndarray,
    parent_birth: np.ndarray,
    parent_cell_birth: float,
    pop_time: float,
    cfg: SimConfig,
    rng: np.random.Generator,
    bank: HaplotypeBank,
    mut_model: Optional[MutationModel],
    scratch_w: np.ndarray,
    scratch_scores: np.ndarray,
    mtdna_dup_sel: Optional[SelectionModel] = None,
    mito_dup_key: str = "mito_dup",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    M = int(parent_h.shape[0])
    target = 2 * M

    work_h = np.empty((target,), dtype=np.int32)
    work_t = np.empty((target,), dtype=parent_birth.dtype)
    work_h[:M] = parent_h
    work_t[:M] = parent_birth
    cur = M

    use_dup_sel = (
        mtdna_dup_sel is not None and
        float(getattr(cfg, "mtdna_dup_sel_strength", 0.0)) != 0.0
    )
    dup_strength = float(cfg.mtdna_dup_sel_strength)

    if use_dup_sel:
        work_score = scratch_scores[:target]
        bank.score_many_into(parent_h, mtdna_dup_sel, mito_dup_key, work_score[:M])

    while cur < target:
        if use_dup_sel:
            w = scratch_w[:cur]
            _fill_ones_nb(w, cur)
            _apply_relative_scores_inplace_nb(w, work_score, cur, dup_strength)
            pidx = _pick_index_by_weights_prefix(w, cur, rng)
        else:
            pidx = int(rng.integers(0, cur))

        parent_hid = int(work_h[pidx])
        parent_birth_time = float(work_t[pidx])
        parent_age = pop_time - parent_birth_time

        new_parent_age = parent_age * float(cfg.mtdna_dup_refresh_parent)
        work_t[pidx] = pop_time - new_parent_age

        child_hid = bank.mutate_child(parent_hid, mut_model, rng)
        child_age = parent_age * float(cfg.mtdna_dup_refresh_child)

        work_h[cur] = child_hid
        work_t[cur] = pop_time - child_age

        if use_dup_sel:
            work_score[cur] = bank.hap_additive_score(int(child_hid), mtdna_dup_sel, mito_dup_key)

        cur += 1

    perm = rng.permutation(target)
    work_h = work_h[perm]
    work_t = work_t[perm]

    half = target // 2
    d1h, d1t = work_h[:half], work_t[:half]
    d2h, d2t = work_h[half:], work_t[half:]

    parent_cell_age = pop_time - parent_cell_birth
    a = float(cfg.cell_div_refresh_a)
    b = float(cfg.cell_div_refresh_b)
    f1, f2 = (a, b) if rng.random() < 0.5 else (b, a)

    d1_cell_birth = pop_time - parent_cell_age * f1
    d2_cell_birth = pop_time - parent_cell_age * f2

    return d1h, d1t, d2h, d2t, d1_cell_birth, d2_cell_birth


def mitotic_generation(
    pop: Population,
    cfg: SimConfig,
    rng: np.random.Generator,
    bank: HaplotypeBank,
    mut_model: Optional[MutationModel],
    dup_sel: Optional[SelectionModel] = None,
    decay_sel: Optional[SelectionModel] = None,
    cell_dup_sel: Optional[SelectionModel] = None,
    cell_decay_sel: Optional[SelectionModel] = None,
    mito_dup_key: str = "mito_dup",
    cell_dup_key: str = "cell_dup",
    cell_decay_key: str = "cell_decay",
) -> Population:
    ncells = pop.n_cells
    if ncells <= 1:
        return pop

    if cfg.thresholding and (cell_dup_sel is not None or cell_decay_sel is not None):
        raise NotImplementedError("Fast path mitotic cell selection assumes thresholding=False")

    if cell_dup_sel is not None and pop.cell_dup_sum is None:
        pop.cell_dup_sum = np.array([
            _recompute_cell_sum(pop, i, bank, cell_dup_sel, cell_dup_key)
            for i in range(pop.n_cells)
        ], dtype=float)

    if cell_decay_sel is not None and pop.cell_decay_sum is None:
        pop.cell_decay_sum = np.array([
            _recompute_cell_sum(pop, i, bank, cell_decay_sel, cell_decay_key)
            for i in range(pop.n_cells)
        ], dtype=float)

    use_cell_decay = (
        cell_decay_sel is not None and
        float(getattr(cfg, "cell_decay_sel_strength", 0.0)) != 0.0
    )
    use_cell_dup = (
        cell_dup_sel is not None and
        float(getattr(cfg, "cell_dup_sel_strength", 0.0)) != 0.0
    )
    cell_decay_strength = float(cfg.cell_decay_sel_strength)
    cell_dup_strength = float(cfg.cell_dup_sel_strength)
    age_bias = float(cfg.cell_death_age_bias)

    for _ in range(ncells):
        pop.sim_time += float(cfg.mitotic_age_step)
        n = pop.n_cells

        if age_bias == 1.0 and not use_cell_decay:
            dead_idx = int(rng.integers(0, n))
        else:
            w_death = pop.tmp_cell_w1[:n]
            _fill_cell_age_weights_nb(w_death, pop.cell_birth, n, pop.sim_time, age_bias)

            if use_cell_decay:
                _apply_relative_scores_inplace_nb(
                    w_death, pop.cell_decay_sum, n, cell_decay_strength
                )

            dead_idx = _pick_index_by_weights_prefix(w_death, n, rng)

        if not use_cell_dup:
            if n == 2:
                parent_idx = 1 - dead_idx
            else:
                u = int(rng.integers(0, n - 1))
                parent_idx = u if u < dead_idx else (u + 1)
        else:
            w_dup = pop.tmp_cell_w2[:n]
            _fill_ones_nb(w_dup, n)
            _apply_relative_scores_inplace_nb(
                w_dup, pop.cell_dup_sum, n, cell_dup_strength
            )
            w_dup[dead_idx] = 0.0
            parent_idx = _pick_index_by_weights_prefix(w_dup, n, rng)

        M = int(pop.m_count[parent_idx])
        parent_h = pop.copy_hid[parent_idx, :M].copy()
        parent_t = pop.mt_birth[parent_idx, :M].copy()
        parent_cell_birth = float(pop.cell_birth[parent_idx])

        d1h, d1t, d2h, d2t, d1cb, d2cb = mitotic_division(
            parent_h, parent_t, parent_cell_birth, pop.sim_time,
            cfg, rng, bank, mut_model,
            scratch_w=pop.tmp_mt_w1,
            scratch_scores=pop.tmp_mt_scores,
            mtdna_dup_sel=dup_sel,
            mito_dup_key=mito_dup_key
        )

        pop.copy_hid[dead_idx, :M] = d1h
        pop.mt_birth[dead_idx, :M] = d1t
        pop.m_count[dead_idx] = M
        pop.cell_birth[dead_idx] = d1cb

        pop.copy_hid[parent_idx, :M] = d2h
        pop.mt_birth[parent_idx, :M] = d2t
        pop.m_count[parent_idx] = M
        pop.cell_birth[parent_idx] = d2cb

        if pop.cell_dup_sum is not None and cell_dup_sel is not None:
            pop.cell_dup_sum[dead_idx] = _recompute_cell_sum(pop, dead_idx, bank, cell_dup_sel, cell_dup_key)
            pop.cell_dup_sum[parent_idx] = _recompute_cell_sum(pop, parent_idx, bank, cell_dup_sel, cell_dup_key)

        if pop.cell_decay_sum is not None and cell_decay_sel is not None:
            pop.cell_decay_sum[dead_idx] = _recompute_cell_sum(pop, dead_idx, bank, cell_decay_sel, cell_decay_key)
            pop.cell_decay_sum[parent_idx] = _recompute_cell_sum(pop, parent_idx, bank, cell_decay_sel, cell_decay_key)

    return pop
