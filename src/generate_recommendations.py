# generate_recommendations.py
"""
This script generates the final top-20 recommendations
using the saved evaluation results without re-running
the entire training pipeline.
"""
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from config import K, OUTPUT_TOP20, EVAL_ALPHA_GRID_OUT
from data_prep import load_and_prepare_data
from models import retrain_final_deepfm, hyperparameter_search_and_train_deepfm
from evaluation import (
    conservative_alpha,
    cf_scores_deepfm, cb_scores_user, rrf_blend_from_scores, topk_from_scores
)

def exclude_consumed_idx(user_id, user2hist_orig, all_items_universe):
    """
    Returns a numpy array of item indices that the user has not consumed.
    This function was likely defined in one of the other project files.
    """
    consumed = user2hist_orig.get(int(user_id), set())
    return np.array([i for i, it in enumerate(all_items_universe) if int(it) not in consumed], dtype=int)


def generate_and_export_recs():
    """
    Generates and exports the final top-20 recommendations.
    """
    print("\nLoading data and models for final recommendation export...")
    # Load and prepare data, but skip model training steps
    (
        inter_df, games_df, test_in_df, train_df, eval_df, test_df,
        dfm_train_data, dfm_eval_data, dfm_datainfo,
        icf_train_data, icf_datainfo,
        all_items_universe, item_universe_pos,
        item_content_tfidf, pop_score, user2hist_orig,
        user_prof, user_prof_norm, user_tagged_counts,
        train_plus_eval_df
    ) = load_and_prepare_data()

    # Get the best alpha value from the saved CSV
    alpha_df = pd.read_csv(EVAL_ALPHA_GRID_OUT)
    best_alpha = float(alpha_df.sort_values(["recall", "ndcg"], ascending=False).iloc[0]["alpha"])
    print(f"Loaded best alpha from previous run: {best_alpha:.2f}")

    # Re-train the final DeepFM model. This is faster than a full run.
    print("\nRe-training final DeepFM model for inference...")
    best_dfm = hyperparameter_search_and_train_deepfm(
        dfm_train_data, dfm_eval_data, dfm_datainfo, eval_df, all_items_universe
    )
    deepfm = retrain_final_deepfm(best_dfm, train_plus_eval_df)
    
    # Export the final top-20 recommendations.
    print(f"\n=== Exporting top-{K} recommendations (DeepFM+Tags best-α) ===")
    if test_in_df is None:
        print("No test_interactions_in.csv found; skipping final recommendations.")
    else:
        out_rows = []
        for u in tqdm(test_in_df["user"].unique(), desc=f"Inferring RQ2 top-{K}"):
            allow = exclude_consumed_idx(u, user2hist_orig, all_items_universe)
            a_u = conservative_alpha(u, best_alpha, user_prof_norm, user_tagged_counts)
            cf_all = cf_scores_deepfm(u, deepfm, all_items_universe)
            cb_all = cb_scores_user(u, user_prof, item_content_tfidf)
            blended = rrf_blend_from_scores(cf_all, cb_all, a_u)
            idxs = topk_from_scores(blended, allow, K)
            for pos in idxs:
                out_rows.append((int(u), int(all_items_universe[pos])))
        out_df = pd.DataFrame(out_rows, columns=["user_id", "item_id"])
        out_df.to_csv(OUTPUT_TOP20, index=False)
        print(f"Saved top-{K} recommendations to {OUTPUT_TOP20} ({len(out_df):,} rows)")

if __name__ == "__main__":
    generate_and_export_recs()