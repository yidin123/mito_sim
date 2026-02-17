from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

#creates a read-only dataclass from specified initial
#configurations (as follows)
@dataclass(frozen=True)
class SimConfig:
    #input files
    init_fasta_path: str = "input_files/mtDNA_init.fasta" #initial mtDNA sequence variants
    mutation_track_path: Optional[str] = "input_files/mutation_tracks.txt" #possible mutation states of mtDNA loci
    mito_dup_selection_path: Optional[str] = "input_files/mito_dup_selection.txt" #additive selection for mtDNA duplication
    mito_decay_selection_path: Optional[str] = "input_files/mito_decay_selection.txt" #additive selection for mtDNA decay
    cell_dup_selection_path: Optional[str] = "input_files/cell_dup_selection.txt" #additive selection for cell duplication
    cell_decay_selection_path: Optional[str] = "input_files/cell_decay_selection.txt" #additive selection for cell decay

    #basic simulation setup
    ncells: int = 25 #number of cells
    mtcn: int = 50 #number of mtDNA copies per cell
    pre_mitotic_timesteps: int = 50 #pre_mitotic mtDNA death/duplication rounds
    mitotic_timesteps: int = 100 #rounds of cell division (across the simulation; each round has ncells death/duplication rounds)
    out_dir: str = "transfer_test" #output directory for simulation states
    n_reps: int = 30 #number of replications

    #mtdna/cell age model
    pre_age_step: float = 1.0 #aging increment per pre-mitotic timestep
    mitotic_age_step: float = 1.0 #aging increment per each mitotic round (ncells rounds total in one round of cell division)
    mtdna_dup_refresh_parent: float = 1.0 #age refresh applied to parent upon duplication (1.0 = none, < 1 = "younger", > 1 = "older")
    mtdna_dup_refresh_child: float = 1.0 #age refresh applied to child upon duplication
    cell_div_refresh_a: float = 1.0 #age refresh randomly assigned to one daughter cell at mitosis
    cell_div_refresh_b: float = 1.0 #age refresh randomly assigned to other daugher cell at mitosis
    mtdna_death_age_bias: float = 1.0 #bias for mtdna death by age (1.0 = none, >1 = more likely for older to die, <1 = less likely)
    cell_death_age_bias: float = 1.0 #bias for cell death by age

    #selection model
    mtdna_dup_sel_strength: float = 1.0 #selection strength for mtdna duplication (higher = more selection)
    mtdna_decay_sel_strength: float = 1.0 #selection strength for mtdna decay
    cell_dup_sel_strength: float = 1.0 # selection strength for cell duplication
    cell_decay_sel_strength: float = 1.0 #selection strength for cell decay

    #aging loss model
    mtdna_loss_gen: int = 100 #start losing mtdna copies after this mitotic generation
    mtdna_loss_rate: float = 0.0 #rate of mtdna loss per cell per mitotic generation (decimals accounted for by skipping rounds- eg. 2 = 2 lost per round, 1.5 = 1 lost one round, 2 the other, 0.5 = 1 lost every other round)
    cell_loss_gen: int = 100 #start losing cells after this mitotic generation
    cell_loss_rate: float = 0.0 #rate of cellular loss per mitotic generation

    #mtdna transfer model
    transfer_event_prob: float = 0.05 #probability of a cell's transfer event at a particualr pre-mitotic generation
    transfer_number: int = 5 #number of mtdna copies exchanged between cell pairs at a transfer event
