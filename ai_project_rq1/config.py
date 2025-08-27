import os
import numpy as np

# --- HERE I AM DEFINING THE CORE PROJECT SETTINGS AND DIRECTORY STRUCTURE ---

# The base directory where the project is located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Here I am creating a data directory to store all input files like CSVs.
DATA_DIR = os.path.join(BASE_DIR, "data")
# Here I am creating a results directory to store all output files and plots.
RESULTS_DIR = os.path.join(BASE_DIR, "results")
# Here I am creating a separate directory within results for saving plots.
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

# --- HERE I AM SPECIFYING THE PATHS FOR INPUT AND OUTPUT FILES ---
TRAIN_INTERACTIONS = os.path.join(DATA_DIR, "train_interactions.csv")
TEST_IN_FILE = os.path.join(DATA_DIR, "test_interactions_in.csv")
GAMES_FILE = os.path.join(DATA_DIR, "games.csv")

OUTPUT_TOP20 = os.path.join(RESULTS_DIR, "top20_recommendations.csv")
EVAL_SUMMARY_OUT = os.path.join(RESULTS_DIR, "evaluation_summary.csv")

# --- HERE I AM SETTING UP THE GLOBAL HYPERPARAMETERS AND CONSTANTS ---
# Using a fixed seed for reproducibility.
SEED = 42
np.random.seed(SEED)
random_seed = SEED

# Top-K recommendation settings.
K = 20
K_LIST = [5, 10, 20]

# Hybrid model alpha grid for tuning.
ALPHAS_GRID = np.arange(0.0, 1.0, 0.05)

# Gating and candidate settings for the hybrid model.
ADAPT_M, ADAPT_S = 5, 2
CAND_CF = 500  # Number of candidates for collaborative filtering.
CAND_CB = 400  # Number of candidates for content-based.
RRF_K = 60     # RRF (Reciprocal Rank Fusion) constant.
MIN_TAG_ITEMS = 3
MIN_PROFILE_NORM = 0.10

# Here I am getting the number of CPU cores for parallel processing.
import os
CPU_CORES = max(1, os.cpu_count() or 1)
TF_SESS_CONFIG = {"intra_op_parallelism_threads": CPU_CORES, "inter_op_parallelism_threads": CPU_CORES}