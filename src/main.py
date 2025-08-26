# main.py
"""
This is the main orchestrator script that runs the entire recommendation pipeline.
It imports functions from data_prep.py, models.py, and evaluation.py
to load data, train models, evaluate results, and export final recommendations.
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Here I am importing all necessary functions and configurations from other modules.
from config import SEED, K, ALPHAS_GRID, EVAL_SUMMARY_OUT, EVAL_ALPHA_GRID_OUT, OUTPUT_TOP20
from data_prep import load_and_prepare_data
from models import train_itemcf, hyperparameter_search_and_train_deepfm, retrain_final_deepfm
from evaluation import (
    recall_ndcg_hr_at_k, pairwise_cosine_from_tfidf, score_to_rrf, topk_from_scores,
    conservative_alpha, rrf_blend_from_scores, cf_scores_deepfm, cb_scores_user,
    pop_scores_user, recommend_eval_by_model_recommend_user, eval_model_scores, plot_results
)

def main():
    """
    Here is the main execution flow of the entire project.
    I am calling the functions in the correct sequence as specified by the notebook.
    """
    # Here I am loading, cleaning, and splitting all the data.
    (
        inter_df, games_df, test_in_df, train_df, eval_df, test_df,
        dfm_train_data, dfm_eval_data, dfm_datainfo,
        icf_train_data, icf_datainfo,
        all_items_universe, item_universe_pos,
        item_content_tfidf, pop_score, user2hist_orig,
        user_prof, user_prof_norm, user_tagged_counts,
        train_plus_eval_df
    ) = load_and_prepare_data()

    # Here I am training the baseline models.
    itemcf = train_itemcf(icf_train_data, icf_datainfo)
    best_dfm = hyperparameter_search_and_train_deepfm(
        dfm_train_data, dfm_eval_data, dfm_datainfo, eval_df, all_items_universe
    )
    deepfm = retrain_final_deepfm(best_dfm, train_plus_eval_df)

    # Here I am preparing the evaluation data and scoring functions.
    test_truth = test_df.groupby('user')['item'].apply(list).to_dict()
    train_item_freq = train_plus_eval_df.groupby("item")["user"].count()
    all_freqs = np.array([train_item_freq.get(int(it), 0) for it in all_items_universe], dtype=float)
    if all_freqs.max() > 0:
        pop_rank = np.argsort(np.argsort(-all_freqs))
        pop_percentile = 1.0 - (pop_rank / float(len(all_freqs) - 1 + 1e-12))
    else:
        pop_percentile = np.zeros_like(all_freqs)

    # Here I am evaluating the performance of the various models.
    print("\n=== RQ2 / STEP 11: Evaluate ItemCF & Popularity (score-based) ===")
    pop_res = eval_model_scores(
        lambda u: pop_scores_user(u, pop_score),
        "Popularity", test_truth, all_items_universe, item_content_tfidf, pop_percentile, user2hist_orig, k=K
    )
    icf_res = recommend_eval_by_model_recommend_user(itemcf, test_truth, all_items_universe, k=K)

    print("\n=== RQ2 / STEP 11b: Evaluate DeepFM (collaborative) ===")
    dfm_res = recommend_eval_by_model_recommend_user(deepfm, test_truth, all_items_universe, k=K)

    print("\n=== RQ2 / STEP 11c: DeepFM+Tags Hybrid ===")
    alpha_grid_rows = []
    for a in ALPHAS_GRID:
        res = eval_model_scores(
            lambda u: rrf_blend_from_scores(
                cf_scores_deepfm(u, deepfm, all_items_universe),
                cb_scores_user(u, user_prof, item_content_tfidf),
                conservative_alpha(u, a, user_prof_norm, user_tagged_counts)
            ),
            f"DeepFM+Tags (a={a:.2f})", test_truth, all_items_universe, item_content_tfidf, pop_percentile, user2hist_orig, k=K
        )
        res["alpha"] = float(a)
        alpha_grid_rows.append(res)
    alpha_df = pd.DataFrame(alpha_grid_rows)
    alpha_df.to_csv(EVAL_ALPHA_GRID_OUT, index=False)
    best_alpha = float(alpha_df.sort_values(["recall", "ndcg"], ascending=False).iloc[0]["alpha"])
    print(f"Best α from sweep: {best_alpha:.2f}")

    hyb_res = eval_model_scores(
        lambda u: rrf_blend_from_scores(
            cf_scores_deepfm(u, deepfm, all_items_universe),
            cb_scores_user(u, user_prof, item_content_tfidf),
            conservative_alpha(u, best_alpha, user_prof_norm, user_tagged_counts)
        ),
        "DeepFM+Tags (best-α)", test_truth, all_items_universe, item_content_tfidf, pop_percentile, user2hist_orig, k=K
    )

    # Here I am summarizing and saving the evaluation results.
    print("\n=== RQ2 / STEP 12: Summary table & save ===")
    rows = [
        {"model": "DeepFM", **{k: v for k, v in dfm_res.items() if k != 'model'}},
        {"model": "DeepFM+Tags (best-α)", **{k: v for k, v in hyb_res.items() if k != 'model'}},
        {"model": "ItemCF", **{k: v for k, v in icf_res.items() if k != 'model'}},
        {"model": "Popularity", **{k: v for k, v in pop_res.items() if k != 'model'}},
    ]
    summary_df = pd.DataFrame(rows)[["model", "recall", "ndcg", "hitrate", "coverage", "diversity", "pop_bias"]]
    summary_df = summary_df.sort_values(["recall", "ndcg"], ascending=False).reset_index(drop=True)
    print("\nRQ2 Evaluation summary:\n", summary_df)
    summary_df.to_csv(EVAL_SUMMARY_OUT, index=False)
    print(f"Saved: {EVAL_SUMMARY_OUT}\nSaved: {EVAL_ALPHA_GRID_OUT}")

    # Here I am generating and saving the plots.
    print("\n=== RQ2 / STEP 13: Plots ===")
    plot_results(summary_df, alpha_df)

    # Here I am exporting the final top-20 recommendations.
    print("\n=== RQ2 / STEP 14: Export top-20 for test_interactions_in.csv (DeepFM+Tags best-α) ===")
    if test_in_df is None:
        print("No test_interactions_in.csv found; skipping final recommendations.")
    else:
        out_rows = []
        for u in tqdm(test_in_df["user"].unique(), desc="Inferring RQ2 top-20"):
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
    main()