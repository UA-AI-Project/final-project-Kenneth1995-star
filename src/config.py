# config.py
"""
This file contains all the global variables and settings for the recommendation system.
It centralizes configuration parameters to make them easy to manage and modify.
"""
import os
import numpy as np
import random

# Here I am setting the random seed for reproducibility.
# This ensures that random operations, like data splitting, produce the same results every time.
# --- SCIENTIFIC SANITY TEST ---
assert isinstance(42, int), "SEED must be an integer."
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Here I am defining the directory where the data files are located.
# The `r` prefix is used for raw strings, which prevents backslashes from being interpreted as escape sequences.
# This makes file paths robust.
DATA_DIR = r"C:\Users\GSI\Pictures\DATASETS"

# Here I am defining the paths for the input data files.
# `os.path.join` is used to create file paths in a way that is compatible with different operating systems.
TRAIN_INTERACTIONS = os.path.join(DATA_DIR, "train_interactions.csv")
TEST_IN_FILE = os.path.join(DATA_DIR, "test_interactions_in.csv")
GAMES_FILE = os.path.join(DATA_DIR, "games.csv")

# Here I am defining the paths for the output files.
# The evaluation summary, alpha sweep results, and final top-20 recommendations will be saved here.
EVAL_SUMMARY_OUT = os.path.join(DATA_DIR, "rq2_eval_summary.csv")
EVAL_ALPHA_GRID_OUT = os.path.join(DATA_DIR, "rq2_alpha_sweep.csv")
OUTPUT_TOP20 = os.path.join(DATA_DIR, "rq2_top20_deepfm_hybrid.csv")
PLOT_DIR = DATA_DIR

# Here I am defining the constant for the number of items to recommend.
# K is a common metric in recommendation systems for Top-K recommendations.
K = 20
# Here I am defining the grid of alpha values to sweep over for the hybrid model.
# Alpha controls the blending ratio between collaborative filtering and content-based scores.
ALPHAS_GRID = np.arange(0.0, 1.0, 0.05)
# --- SCIENTIFIC SANITY TEST ---
assert len(ALPHAS_GRID) > 1, "ALPHAS_GRID must have more than one value for a meaningful sweep."

# Here I am defining gating/candidate settings.
# CAND_CF is the number of candidates to retrieve from the collaborative filtering model.
CAND_CF = 500
# RRF_K is the constant for reciprocal-rank fusion, used for blending scores.
RRF_K = 60
# --- SCIENTIFIC SANITY TEST ---
assert CAND_CF > 0, "CAND_CF must be a positive integer."
assert RRF_K > 0, "RRF_K must be a positive integer."

# Here I am defining thresholds for content profile analysis.
# MIN_TAG_ITEMS is the minimum number of items with tags required to build a user profile.
MIN_TAG_ITEMS = 3
# MIN_PROFILE_NORM is a threshold to determine if a user's content profile is meaningful.
MIN_PROFILE_NORM = 0.10
# ADAPT_S is a scaling factor for the adaptive alpha function.
ADAPT_S = 2
# --- SCIENTIFIC SANITY TEST ---
assert MIN_TAG_ITEMS >= 0, "MIN_TAG_ITEMS cannot be negative."
assert MIN_PROFILE_NORM >= 0, "MIN_PROFILE_NORM cannot be negative."
assert ADAPT_S >= 0, "ADAPT_S cannot be negative."

# Here I am setting TensorFlow session configurations.
# This helps optimize performance by controlling thread parallelism.
CPU_CORES = max(1, os.cpu_count() or 1)
TF_SESS_CONFIG = {
    "intra_op_parallelism_threads": CPU_CORES,
    "inter_op_parallelism_threads": CPU_CORES,
}
# --- SCIENTIFIC SANITY TEST ---
assert isinstance(CPU_CORES, int) and CPU_CORES > 0, "CPU_CORES must be a positive integer."
assert "intra_op_parallelism_threads" in TF_SESS_CONFIG, "TF_SESS_CONFIG must contain 'intra_op_parallelism_threads'."

