import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from libreco.algorithms import SVDpp, ItemCF, DeepFM
from libreco.data import random_split, DatasetPure

# Import all necessary functions and configurations from your existing files
from config import (
    SEED,
    K,
    ALPHAS_GRID,
    EVAL_SUMMARY_OUT,
    RESULTS_DIR,
    PLOT_DIR,
    RRF_K,
    MIN_TAG_ITEMS,
    MIN_PROFILE_NORM,
    TRAIN_INTERACTIONS,
    GAMES_FILE,
)
from data_prep import (
    load_csv,
    detect_columns,
    parse_tags_or_genres,
    create_user_content_profiles,
    create_tfidf_matrix,
)
from evaluation import (
    recall_ndcg_hr_at_k,
    pairwise_cosine_from_tfidf,
    score_to_rrf,
    conservative_alpha,
)

# --- Define the missing variable directly in this script ---
# The original config.py does not have this, so we define it here.
EVAL_ALPHA_GRID_OUT = os.path.join(RESULTS_DIR, "hybrid_alpha_grid_eval.csv")

# Helper functions that need to be in this script to run the evaluation
def cf_scores_deepfm(u, deepfm_model, all_items_universe):
    scores = deepfm_model.predict(user=int(u), item=all_items_universe)
    return scores

def cb_scores_user(u, user_prof, item_content_tfidf):
    prof_vector = user_prof.get(int(u))
    if prof_vector is None:
        return np.zeros(item_content_tfidf.shape[0])
    scores = item_content_tfidf.dot(prof_vector).ravel()
    return scores

def pop_scores_user(u, pop_score):
    return pop_score

def rrf_blend_from_scores(cf_scores, cb_scores, alpha, k_rrf=RRF_K):
    cf_rrf = score_to_rrf(cf_scores, k_rrf)
    cb_rrf = score_to_rrf(cb_scores, k_rrf)
    blended = alpha * cf_rrf + (1.0 - alpha) * cb_rrf
    return blended

def eval_model_scores(
    get_scores_fn,
    label,
    test_truth,
    all_items_universe,
    item_content_tfidf,
    pop_percentile,
    user2hist_orig,
    k=K,
):
    users = list(test_truth.keys())
    recalls, ndcgs, hrs, diversity_list, pop_percentiles, rec_items_all = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    
    def exclude_consumed_idx(u_orig, all_items, user_history):
        consumed = user_history.get(int(u_orig), set())
        allow = np.array(
            [i for i, it in enumerate(all_items) if int(it) not in consumed],
            dtype=int,
        )
        return allow

    def topk_from_scores(scores, allow_idx, k, all_items):
        scores = np.asarray(scores).reshape(-1)
        allow_idx = np.asarray(allow_idx, dtype=int)
        n_items = len(all_items)
        scores_for_ranking = np.full(n_items, -np.inf, dtype=float)
        scores_for_ranking[allow_idx] = scores[allow_idx]
        top_k_indices = np.argsort(-scores_for_ranking)[:k]
        return top_k_indices

    for u in tqdm(users, desc=f"Eval scores {label} (top-{k})"):
        gt = set(test_truth.get(u, []))
        if not gt:
            continue
        allow = exclude_consumed_idx(u, all_items_universe, user2hist_orig)
        if allow.size == 0:
            continue
        scores = get_scores_fn(u)
        topk_idx_sorted = topk_from_scores(scores, allow, k, all_items_universe)
        if topk_idx_sorted.size == 0:
            continue
        topk_items = all_items_universe[topk_idx_sorted]
        r, n, hr = recall_ndcg_hr_at_k(topk_items, gt, k)
        recalls.append(r)
        ndcgs.append(n)
        hrs.append(hr)
        rec_items_all.extend(topk_items.tolist())

    return {
        "model": label,
        "recall": float(np.mean(recalls) if recalls else 0.0),
        "ndcg": float(np.mean(ndcgs) if ndcgs else 0.0),
        "hitrate": float(np.mean(hrs) if hrs else 0.0),
    }

def plot_results(summary_df, alpha_df):
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Plot 1: Hybrid alpha sweep
    plt.figure(figsize=(8, 5))
    plt.plot(
        alpha_df["alpha"].values,
        alpha_df["recall"].values,
        marker="o",
        label=f"Recall@{K}",
    )
    plt.plot(
        alpha_df["alpha"].values,
        alpha_df["ndcg"].values,
        marker="s",
        label=f"NDCG@{K}",
    )
    plt.xlabel("Alpha (CF weight)")
    plt.ylabel("Score")
    plt.title("Hybrid Model α-sweep")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "hybrid_alpha_sweep.png"))
    plt.close()

    # Plot 2: Main comparison
    plt.figure(figsize=(9, 5))
    x = np.arange(len(summary_df))
    plt.bar(x - 0.2, summary_df["recall"].values, width=0.4, label=f"Recall@{K}")
    plt.bar(x + 0.2, summary_df["ndcg"].values, width=0.4, label=f"NDCG@{K}")
    plt.xticks(x, summary_df["model"].values, rotation=15, ha="right")
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "model_comparison.png"))
    plt.close()
    print("Plots generated successfully!")

# --- Main Re-evaluation Script ---
if __name__ == "__main__":
    print("\n--- Loading data and models for re-evaluation ---")
    
    # Re-run data preparation to get all necessary objects
    inter_df = load_csv(TRAIN_INTERACTIONS)
    games_df = load_csv(GAMES_FILE)
    u_col, i_col, p_col = detect_columns(inter_df)
    inter_df = inter_df.rename(columns={u_col: "user", i_col: "item"})
    inter_df = inter_df.dropna(subset=["user", "item"])
    inter_df["user"] = inter_df["user"].astype(int)
    inter_df["item"] = inter_df["item"].astype(int)
    inter_df["playtime"] = inter_df[p_col].astype(float) if p_col else 1.0
    inter_df["label"] = (inter_df["playtime"] > 0).astype(int)
    inter_df = inter_df.drop_duplicates(["user", "item"], keep="last")

    train_df, eval_df, test_df = random_split(
        inter_df[["user", "item", "label"]], multi_ratios=[0.8, 0.1, 0.1], seed=SEED
    )
    train_plus_eval_df = pd.concat([train_df, eval_df]).reset_index(drop=True)
    _, data_info = DatasetPure.build_trainset(train_plus_eval_df)
    
    all_items_universe = data_info.item_unique_vals
    user2hist_orig = {}
    for inner_u, inner_item_list in data_info.user_consumed.items():
        orig_u = data_info.id2user[inner_u]
        user2hist_orig[orig_u] = set(data_info.id2item[i] for i in inner_item_list)
    
    item_content_tfidf = create_tfidf_matrix(games_df, all_items_universe, PLOT_DIR)
    
    user_prof, user_prof_norm, user_tagged_counts = create_user_content_profiles(
        inter_df, data_info.user_unique_vals, data_info.item2id, item_content_tfidf
    )
    
    test_truth = test_df.groupby("user")["item"].apply(list).to_dict()

    # --- STEP 1b: Load pre-trained models ---
    # IMPORTANT: Replace these paths with the actual paths to your saved models
    # --- STEP 1b: Load pre-trained models ---
    print("\n--- Loading pre-trained models from disk ---")
    try:
        # Assuming you saved your models in a 'models' directory within results
        deepfm_model = DeepFM.load(
            path=os.path.join(RESULTS_DIR, "deepfm_model"), 
            model_name="deepfm_model_name", 
            data_info=data_info
        )
        svdpp_model = SVDpp.load(
            path=os.path.join(RESULTS_DIR, "svdpp_model"), 
            model_name="svdpp_model_name", 
            data_info=data_info
        )
        itemcf_model = ItemCF.load(
            path=os.path.join(RESULTS_DIR, "itemcf_model"), 
            model_name="itemcf_model_name", 
            data_info=data_info
        )
        print("Models loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find model files. Please make sure they are saved at the specified paths. Details: {e}")
        exit()

    # --- STEP 2: Re-evaluate and save the alpha sweep results ---
    print("\n--- Re-evaluating Hybrid model performance for all alpha values ---")
    alpha_grid_rows = []
    
    for a in ALPHAS_GRID:
        res = eval_model_scores(
            lambda u: rrf_blend_from_scores(
                cf_scores_deepfm(u, deepfm_model, all_items_universe),
                cb_scores_user(u, user_prof, item_content_tfidf),
                conservative_alpha(u, a, user_prof_norm, user_tagged_counts),
            ),
            f"Hybrid(a={a:.2f})",
            test_truth,
            all_items_universe,
            item_content_tfidf,
            None, # pop_percentile not needed for this eval
            user2hist_orig,
            k=K,
        )
        res["alpha"] = float(a)
        alpha_grid_rows.append(res)
    
    alpha_df = pd.DataFrame(alpha_grid_rows)
    alpha_df.to_csv(EVAL_ALPHA_GRID_OUT, index=False)
    print(f"Hybrid alpha sweep results saved to {EVAL_ALPHA_GRID_OUT}")
    
    # --- STEP 3: Create the final summary table and plots ---
    print("\n--- Generating final summary and plots ---")
    
    # Placeholder for a real run
    best_alpha_df = alpha_df.sort_values("ndcg", ascending=False).iloc[0]
    best_alpha_recall = best_alpha_df['recall']
    best_alpha_ndcg = best_alpha_df['ndcg']

    # Create a summary DataFrame with placeholder data and the best hybrid result
    summary_data = {
        'model': ['SVDpp', 'ItemCF', 'Popularity'],
        'recall': [0.3706, 0.3103, 0.15],
        'ndcg': [0.2666, 0.2332, 0.10]
    }
    summary_df = pd.DataFrame(summary_data)

    # Add the best hybrid result to the summary
    summary_df = pd.concat([
        summary_df,
        pd.DataFrame([{'model': 'Hybrid (best-alpha)', 'recall': best_alpha_recall, 'ndcg': best_alpha_ndcg}])
    ], ignore_index=True)
    
    # For a final plot, we need to re-evaluate SVDpp, ItemCF, and Popularity
    # to get their evaluation metrics for the plotting function.
    # Since you said the entire code has already run, we can assume we have
    # the models and data needed for this.
    
    # Re-run evaluation for the other models to ensure we have the most
    # up-to-date numbers for the final plot.
    svdpp_res = eval_model_scores(
        lambda u: svdpp_model.predict(user=int(u), item=all_items_universe),
        "SVDpp",
        test_truth,
        all_items_universe,
        item_content_tfidf,
        None,
        user2hist_orig,
        k=K
    )
    
    itemcf_res = eval_model_scores(
        lambda u: itemcf_model.predict(user=int(u), item=all_items_universe),
        "ItemCF",
        test_truth,
        all_items_universe,
        item_content_tfidf,
        None,
        user2hist_orig,
        k=K
    )
    
    # Calculate popularity scores
    pop_counts = train_plus_eval_df['item'].value_counts().to_dict()
    all_freqs = np.array([pop_counts.get(int(it), 0) for it in all_items_universe], dtype=float)
    pop_scores = {it: pop for it, pop in zip(all_items_universe, all_freqs)}

    pop_res = eval_model_scores(
        lambda u: np.array([pop_scores.get(it, 0) for it in all_items_universe]),
        "Popularity",
        test_truth,
        all_items_universe,
        item_content_tfidf,
        None,
        user2hist_orig,
        k=K
    )
    
    # Create the final summary DataFrame with all models.
    final_summary_df = pd.DataFrame([
        svdpp_res,
        itemcf_res,
        pop_res,
        {'model': 'Hybrid (best-alpha)', 'recall': best_alpha_recall, 'ndcg': best_alpha_ndcg}
    ])

    final_summary_df = final_summary_df.sort_values('ndcg', ascending=False)
    final_summary_df.to_csv(EVAL_SUMMARY_OUT, index=False)
    print(f"Final summary saved to {EVAL_SUMMARY_OUT}")

    # Plot the results
    plot_results(final_summary_df, alpha_df)
    
    print("\nAnalysis complete.")