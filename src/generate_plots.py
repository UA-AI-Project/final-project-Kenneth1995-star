import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Import the plot_results function from your updated evaluation.py
from evaluation import plot_results

# Define the constants from config.py that plot_results needs
PLOT_DIR = r"C:\Users\GSI\Pictures\DATASETS"
K = 20

# Load the CSV files you've already generated
summary_df = pd.read_csv(os.path.join(PLOT_DIR, "rq2_eval_summary.csv"))
alpha_df = pd.read_csv(os.path.join(PLOT_DIR, "rq2_alpha_sweep.csv"))

# Call the plotting function
plot_results(summary_df, alpha_df)

print("Plots have been generated successfully in:", PLOT_DIR)