# main.py:
from __future__ import annotations
import os
import random
import numpy as np
from config import SimConfig
from init_fasta import read_init_fasta_multi
from sim_core import (
    initialize_population_multi,
    moran_step_population,
    mitotic_generation,
    increment_ages,
    apply_mtdna_loss,
    apply_cell_loss,
    maybe_transfer_event
)
from snapshot import write_snapshot
from mutation_tracks import read_mutation_tracks
from selection_tracks import read_selection_tracks

def run(cfg: SimConfig) -> None:

    #read initial fasta
    celltypes = read_init_fasta_multi(cfg.init_fasta_path)
    spec = celltypes[0].spec

    #process mtdna selection
    mito_dup_sel = None
    mito_decay_sel = None

    #read in tracks if present; error if specified length not equal to actual
    if cfg.mito_dup_selection_path:
        mito_dup_sel = read_selection_tracks(cfg.mito_dup_selection_path)
        if mito_dup_sel.length != spec.length:
            raise ValueError(f"mito dup selection length {mito_dup_sel.length} != init length {spec.length}")

    if cfg.mito_decay_selection_path:
        mito_decay_sel = read_selection_tracks(cfg.mito_decay_selection_path)
        if mito_decay_sel.length != spec.length:
            raise ValueError(f"mito decay selection length {mito_decay_sel.length} != init length {spec.length}")

    #repeat for cell selection
    cell_dup_sel = None
    cell_decay_sel = None

    if cfg.cell_dup_selection_path:
        cell_dup_sel = read_selection_tracks(cfg.cell_dup_selection_path)
        if cell_dup_sel.length != spec.length:
            raise ValueError(f"cell_dup length {cell_dup_sel.length} != init length {spec.length}")

    if cfg.cell_decay_selection_path:
        cell_decay_sel = read_selection_tracks(cfg.cell_decay_selection_path)
        if cell_decay_sel.length != spec.length:
            raise ValueError(f"cell_decay length {cell_decay_sel.length} != init length {spec.length}")

    #process mutation model
    mut_model = None
    if cfg.mutation_track_path:
        mut_model = read_mutation_tracks(cfg.mutation_track_path)

        #check that specified lengths match init sequences
        if mut_model.length != spec.length:
            raise ValueError(
                f"Mutation model length {mut_model.length} != init fasta length {spec.length}"
            )

    #check if outdir exists; else create
    os.makedirs(cfg.out_dir, exist_ok=True)
    #loop iteration for each rep
    for rep in range(1, cfg.n_reps + 1):
        #specify default random number generator
        rng = np.random.default_rng()

        #create directory for each rep (or check if exists)
        rep_dir = os.path.join(cfg.out_dir, f"rep_{rep:03d}")
        os.makedirs(rep_dir, exist_ok=True)

        #initialize population
        pop = initialize_population_multi(celltypes, cfg.ncells, cfg.mtcn, rng)

        #write t=0 initial snapshot
        write_snapshot(os.path.join(rep_dir, "t000_init.txt"), pop)

        #file counter
        t = 0

        #track loss accumulated
        mtdna_loss_accum = 0.0
        cell_loss_accum = 0.0

        #run for #mitotic timesteps
        for _g in range(1, cfg.mitotic_timesteps + 1):
            #run for #pre-mitotic timesteps
            for _ in range(cfg.pre_mitotic_timesteps):
                #transfer event
                maybe_transfer_event(pop, cfg, rng)
                #increment ages
                increment_ages(pop, cfg.pre_age_step)
                #runs a particular moran step
                moran_step_population(pop, cfg, rng, mut_model, mito_dup_sel, mito_decay_sel)

            #if mtdna loss occurs
            if cfg.mtdna_loss_rate > 0.0 and _g > cfg.mtdna_loss_gen:
                mtdna_loss_accum += cfg.mtdna_loss_rate #mtdna loss increment (accumulate fractional loss)
                #convert to int floor (how many mtdna copies should be removed now)
                k_loss = int(mtdna_loss_accum)
                if k_loss > 0: #if one or more removed
                    #loss model occurs
                    apply_mtdna_loss(pop, cfg, rng, k_loss, mito_decay_sel)
                    #subtracting used portion from accumulator, keeping fractional remainder
                    mtdna_loss_accum -= k_loss

            #if cell loss occurs; same logic as mtdna loss
            if cfg.cell_loss_rate > 0.0 and _g > cfg.cell_loss_gen:
                cell_loss_accum += cfg.cell_loss_rate
                k_cell = int(cell_loss_accum)
                if k_cell > 0:
                    pop = apply_cell_loss(pop, cfg, rng, k_cell, cell_decay_sel)
                    cell_loss_accum -= k_cell

            #mitosis occurs
            pop = mitotic_generation(pop, cfg, rng, mut_model, mito_dup_sel, mito_decay_sel, cell_dup_sel, cell_decay_sel)

            #save to snapshot after mitosis
            t += 1
            write_snapshot(os.path.join(rep_dir, f"t{t:03d}_mito.txt"), pop)

#run main
if __name__ == "__main__":
    cfg = SimConfig()   #uses defaults from config.py
    run(cfg)
    print(f"Done. Snapshots in: {cfg.out_dir}")
