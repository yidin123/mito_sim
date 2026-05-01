from __future__ import annotations
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from config import SimConfig
from init_fasta import read_init_fasta_multi
from mutation_tracks import read_mutation_tracks
from selection_tracks import read_selection_tracks

from sim_core import (
    initialize_population_multi_fast,
    moran_step_population,
    mitotic_generation,
    apply_mtdna_loss,
    apply_cell_loss,
    maybe_transfer_event,
)


def write_snapshot_fast(path: str, pop, bank, alphabet: str) -> None:
    #writes out current population state as sequences per cell
    #materializes sparse haplotypes into full sequences only at output time
    alpha_bytes = np.frombuffer(alphabet.encode("ascii"), dtype="S1")
    materialize = bank.materialize
    copy_hid = pop.copy_hid
    m_count = pop.m_count
    n_cells = pop.n_cells

    with open(path, "w", encoding="ascii") as f:
        for ci in range(n_cells):
            f.write(f">cell_{ci}\n")
            M = int(m_count[ci])
            row_hids = copy_hid[ci, :M]

            #each mtDNA copy is written as a full sequence
            for hid in row_hids:
                seq = materialize(int(hid))
                f.write(alpha_bytes[seq].tobytes().decode("ascii"))
                f.write("\n")


def _run_one_rep(cfg: SimConfig, rep: int) -> str:
    #runs a single replicate of the full simulation
    #everything (rng, output dir, population) is scoped to this replicate
    celltypes = read_init_fasta_multi(cfg.init_fasta_path)
    spec = celltypes[0].spec
    alphabet = spec.alphabet

    #selection/mutation models are optional and loaded if provided
    mito_dup_sel = None
    mito_decay_sel = None
    cell_dup_sel = None
    cell_decay_sel = None
    mut_model = None

    if cfg.mito_dup_selection_path:
        mito_dup_sel = read_selection_tracks(cfg.mito_dup_selection_path)
    if cfg.mito_decay_selection_path:
        mito_decay_sel = read_selection_tracks(cfg.mito_decay_selection_path)
    if cfg.cell_dup_selection_path:
        cell_dup_sel = read_selection_tracks(cfg.cell_dup_selection_path)
    if cfg.cell_decay_selection_path:
        cell_decay_sel = read_selection_tracks(cfg.cell_decay_selection_path)
    if cfg.mutation_track_path:
        mut_model = read_mutation_tracks(cfg.mutation_track_path)

    #replicate-specific RNG seed so runs are reproducible but independent
    rep_seed = None if cfg.rng_seed is None else (cfg.rng_seed + rep)
    rng = np.random.default_rng(rep_seed)

    #each replicate writes to its own directory
    rep_dir = os.path.join(cfg.out_dir, f"rep_{rep:03d}")
    os.makedirs(rep_dir, exist_ok=True)

    #birth time precision can be toggled for memory/performance
    birth_dtype = np.float32 if cfg.birth_time_dtype == "float32" else np.float64
    pop, bank = initialize_population_multi_fast(
        celltypes, cfg.ncells, cfg.mtcn, rng, birth_dtype=birth_dtype
    )

    #initial snapshot before any dynamics
    write_snapshot_fast(os.path.join(rep_dir, "t000_init.txt"), pop, bank, alphabet=alphabet)

    t = 0
    mtdna_loss_accum = 0.0
    cell_loss_accum = 0.0

    #main simulation loop over mitotic generations
    for g in range(1, cfg.mitotic_timesteps + 1):

        #pre-mitotic phase: mtDNA-level dynamics within cells
        for _ in range(cfg.pre_mitotic_timesteps):
            maybe_transfer_event(pop, cfg, rng)
            #transfer introduces mixing between cells
            pop.sim_time += float(cfg.pre_age_step)
            #time advances continuously even outside division
            moran_step_population(
                pop, cfg, rng, bank, mut_model,
                mito_dup_sel, mito_decay_sel
            )
            #within-cell Moran updates for mtDNA

        #mtDNA loss accumulates gradually and is applied in integer chunks
        if cfg.mtdna_loss_rate > 0.0 and g > cfg.mtdna_loss_gen:
            mtdna_loss_accum += cfg.mtdna_loss_rate
            k_loss = int(mtdna_loss_accum)
            if k_loss > 0:
                apply_mtdna_loss(pop, cfg, rng, bank, k_loss, mito_decay_sel)
                mtdna_loss_accum -= k_loss

        #cell loss similarly accumulates over generations
        if cfg.cell_loss_rate > 0.0 and g > cfg.cell_loss_gen:
            cell_loss_accum += cfg.cell_loss_rate
            k_cell = int(cell_loss_accum)
            if k_cell > 0:
                pop = apply_cell_loss(pop, cfg, rng, bank, k_cell, cell_decay_sel)
                cell_loss_accum -= k_cell

        #mitotic step: cell-level death + division with mtDNA dynamics inside
        pop = mitotic_generation(
            pop, cfg, rng, bank, mut_model,
            dup_sel=mito_dup_sel,
            decay_sel=mito_decay_sel,
            cell_dup_sel=cell_dup_sel,
            cell_decay_sel=cell_decay_sel,
        )

        t += 1

        #snapshot after each mitotic generation
        write_snapshot_fast(
            os.path.join(rep_dir, f"t{t:03d}_mito.txt"),
            pop, bank, alphabet=alphabet
        )

    return rep_dir


def run(cfg: SimConfig) -> None:
    #runs all replicates sequentially
    #could be parallelized but kept simple here
    os.makedirs(cfg.out_dir, exist_ok=True)

    for rep in range(1, cfg.n_reps + 1):
        _run_one_rep(cfg, rep)


if __name__ == "__main__":
    #entry point: construct config and run simulation
    cfg = SimConfig()
    run(cfg)
    print(f"Done. Snapshots in: {cfg.out_dir}")
