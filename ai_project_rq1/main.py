import os
import pandas as pd
from collections import defaultdict
import numpy as np
from libreco.data import DatasetPure
from tqdm import tqdm

# --- HERE I AM IMPORTING ALL THE COMPONENTS FROM MY PROJECT FILES ---
from config import *
from data_prep import prepare_data, create_user_content_profiles, create_tfidf_matrix
from models import train_svdpp_model, train_itemcf_model, get_best_svdpp_params
from evaluation import (
    eval_model_scores,
    recommend_eval_by_model_recommend_user,
    rrf_blend_from_scores,
    conservative_alpha,
    generate_plots
)

# --- HELPER FUNCTIONS FOR SCORING AND INFERENCE ---
# HERE I AM CREATING WRAPPER FUNCTIONS TO GET SCORES FROM DIFFERENT MODELS.

def cf_scores_user(u, svdpp, all_items, item2pos, top_n=CAND_CF):
    """HERE I AM GETTING SVD++ SCORES FOR A USER."""
    if int(u) not in svdpp.data_info.user2id:
        return np.zeros(len(all_items), dtype=float)
    recs = svdpp.recommend_user(int(u), n_rec=top_n, filter_consumed=True, inner_id=False)
    arr = recs.get(int(u), np.array([], dtype=int)) if isinstance(recs, dict) else np.asarray(recs)
    scores = np.zeros(len(all_items), dtype=float)
    for r, it in enumerate(arr[:top_n]):
        pos = item2pos.get(int(it), None)
        if pos is not None:
            scores[pos] = 1.0 / float(r + 1)
    return scores

def icf_scores_user(u, itemcf, all_items, item2pos, top_n=CAND_CF):
    """HERE I AM GETTING ITEMCF SCORES FOR A USER."""
    if int(u) not in itemcf.data_info.user2id:
        return np.zeros(len(all_items), dtype=float)
    recs = itemcf.recommend_user(int(u), n_rec=top_n, filter_consumed=True, inner_id=False)
    arr = recs.get(int(u), np.array([], dtype=int)) if isinstance(recs, dict) else np.asarray(recs)
    scores = np.zeros(len(all_items), dtype=float)
    for r, it in enumerate(arr[:top_n]):
        pos = item2pos.get(int(it), None)
        if pos is not None:
            scores[pos] = 1.0 / float(r + 1)
    return scores

def cb_scores_user(u, user_prof, item_content_tfidf, all_items):
    """HERE I AM GETTING CONTENT-BASED SCORES FOR A USER."""
    prof = user_prof.get(int(u))
    if prof is None: return np.zeros(len(all_items), dtype=float)
    s = item_content_tfidf.dot(prof)
    return np.asarray(s).ravel()

def pop_scores_user(u, pop_score):
    """HERE I AM GETTING POPULARITY-BASED SCORES FOR A USER."""
    return pop_score

def topk_from_scores(scores, allow_idx, k: int):
    """HERE I AM SELECTING THE TOP K ITEMS FROM A LIST OF SCORES."""
    scores = np.asarray(scores).reshape(-1)
    allow_idx = np.asarray(allow_idx, dtype=int)
    n_items = len(scores)

    if allow_idx.size == 0:
        return np.array([], dtype=int)

    take = min(k, allow_idx.size)
    if allow_idx.size == 1:
        return allow_idx.copy()

    part = np.argpartition(-scores[allow_idx], take - 1)[:take]
    cand = allow_idx[part]
    order = cand[np.argsort(-scores[cand])]
    return order[:k]

def exclude_consumed_idx(u_orig, user2hist_orig, all_items):
    """HERE I AM IDENTIFYING AND EXCLUDING ITEMS THE USER HAS ALREADY CONSUMED."""
    consumed = user2hist_orig.get(int(u_orig), set())
    allow = np.array([i for i, it in enumerate(all_items) if int(it) not in consumed], dtype=int)
    return allow

def main():
    """
    HERE I AM ORCHESTRATING THE ENTIRE RECOMMENDATION PIPELINE FROM DATA PREP TO FINAL INFERENCE.
    """
    # 1. Data Preparation
    (
        inter_df, train_df, eval_df, test_df, test_in_df,
        train_data, eval_data, test_data, data_info,
        all_items, all_users, item2pos, user2hist_orig, games_df
    ) = prepare_data()

    # 2. Content TF-IDF
    item_content_tfidf = create_tfidf_matrix(games_df, all_items, PLOT_DIR)

    # 3. Hyperparameter Tuning
    eval_truth = eval_df.groupby('user')['item'].apply(list).to_dict()
    best_params = get_best_svdpp_params(train_data, eval_truth, data_info)

    # 4. Retrain Models on Combined Data
    train_plus_eval_df = pd.concat([train_df, eval_df], ignore_index=True)
    train_data_final, data_info_final = DatasetPure.build_trainset(train_plus_eval_df)
    
    # Update global data info and mappings
    all_items = np.array(data_info_final.item_unique_vals, dtype=int)
    all_users = np.array(data_info_final.user_unique_vals, dtype=int)
    item2pos = {it: i for i, it in enumerate(all_items)}
    id2user = data_info_final.id2user
    id2item = data_info_final.id2item
    user2hist_orig = {}
    for inner_u, inner_item_list in data_info_final.user_consumed.items():
        orig_u = id2user[inner_u]
        user2hist_orig[orig_u] = set(id2item[i] for i in inner_item_list)
        
    svdpp = train_svdpp_model(data_info_final, train_data_final, best_params)
    itemcf = train_itemcf_model(data_info_final, train_data_final)

    # 5. Build User Content Profiles
    user_prof, user_prof_norm, user_tagged_counts = create_user_content_profiles(inter_df, all_users, item2pos, item_content_tfidf)
    
    # Recompute popularity on the final combined dataset.
    pop_counts = train_plus_eval_df['item'].value_counts().to_dict()
    pop_score = np.array([pop_counts.get(int(it), 0) for it in all_items], dtype=float)
    pop_score = pop_score / (pop_score.max() or 1.0)

    # 6. Build Test Truth and Evaluate Models
    print("\n--- STEP 7: Building test truth and evaluating models ---")
    test_truth = test_df.groupby('user')['item'].apply(list).to_dict()

    svd_res = recommend_eval_by_model_recommend_user(svdpp, test_truth)
    print(f"SVDpp -> recall@{K}: {svd_res['recall']:.4f}, ndcg@{K}: {svd_res['ndcg']:.4f}")

    icf_res = recommend_eval_by_model_recommend_user(itemcf, test_truth)
    print(f"ItemCF -> recall@{K}: {icf_res['recall']:.4f}, ndcg@{K}: {icf_res['ndcg']:.4f}")
    
    cb_res = eval_model_scores(
        lambda u: cb_scores_user(u, user_prof, item_content_tfidf, all_items),
        "Content", test_truth, train_plus_eval_df, item_content_tfidf, all_items, item2pos, user2hist_orig
    )
    pop_res = eval_model_scores(
        lambda u: pop_scores_user(u, pop_score),
        "Popularity", test_truth, train_plus_eval_df, item_content_tfidf, all_items, item2pos, user2hist_orig
    )

    rows = [
        {"model": "SVDpp", "recall": svd_res["recall"], "ndcg": svd_res["ndcg"], "hitrate": svd_res["hitrate"], "coverage": svd_res.get("coverage", 0.0), "diversity": None, "pop_bias": None},
        {"model": "Content", "recall": cb_res["recall"], "ndcg": cb_res["ndcg"], "hitrate": cb_res["hitrate"], "coverage": cb_res["coverage"], "diversity": cb_res["diversity"], "pop_bias": cb_res["pop_bias"]},
        {"model": "Popularity", "recall": pop_res["recall"], "ndcg": pop_res["ndcg"], "hitrate": pop_res["hitrate"], "coverage": pop_res["coverage"], "diversity": pop_res["diversity"], "pop_bias": pop_res["pop_bias"]},
        {"model": "ItemCF", "recall": icf_res["recall"], "ndcg": icf_res["ndcg"], "hitrate": icf_res["hitrate"], "coverage": icf_res.get("coverage", 0.0), "diversity": None, "pop_bias": None},
    ]

    hgrid = []
    for a in ALPHAS_GRID:
        res = eval_model_scores(
            lambda u: rrf_blend_from_scores(exclude_consumed_idx(u, user2hist_orig, all_items), cf_scores_user(u, svdpp, all_items, item2pos), cb_scores_user(u, user_prof, item_content_tfidf, all_items), conservative_alpha(u, a, user_prof_norm, user_tagged_counts)),
            f"Hybrid(a={a:.2f})", test_truth, train_plus_eval_df, item_content_tfidf, all_items, item2pos, user2hist_orig
        )
        print(f"alpha={a:.2f} -> recall@{K}: {res['recall']:.4f}, ndcg@{K}: {res['ndcg']:.4f}")
        res["alpha"] = float(a)
        hgrid.append(res)
    
    had_res = eval_model_scores(
        lambda u: rrf_blend_from_scores(exclude_consumed_idx(u, user2hist_orig, all_items), cf_scores_user(u, svdpp, all_items, item2pos), cb_scores_user(u, user_prof, item_content_tfidf, all_items), conservative_alpha(u, 0.5, user_prof_norm, user_tagged_counts)),
        "Hybrid-Adapt", test_truth, train_plus_eval_df, item_content_tfidf, all_items, item2pos, user2hist_orig
    )
    rows.append({"model": "Hybrid-Adapt", "recall": had_res["recall"], "ndcg": had_res["ndcg"], "hitrate": had_res["hitrate"], "coverage": had_res["coverage"], "diversity": had_res["diversity"], "pop_bias": had_res["pop_bias"]})
    
    summary_df = pd.DataFrame(rows)[["model", "recall", "ndcg", "hitrate", "coverage", "diversity", "pop_bias"]]
    summary_df = summary_df.sort_values(["recall", "ndcg"], ascending=False).reset_index(drop=True)
    print("\nEvaluation summary:\n", summary_df)
    summary_df.to_csv(EVAL_SUMMARY_OUT, index=False)
    print(f"Saved evaluation summary to {EVAL_SUMMARY_OUT}")

    # 7. Generate Plots
    generate_plots(summary_df, hgrid)

    # 8. Generate Final Recommendations
    print("\n--- STEP 10: Generate final top-20 recommendations ---")
    if test_in_df is None:
        print("No test_interactions_in.csv found; skipping final recommendations.")
    else:
        if hgrid:
            dfh = pd.DataFrame(hgrid)
            best_alpha = float(dfh.sort_values(["recall", "ndcg"], ascending=False).iloc[0]["alpha"])
            print(f"Choosing best_alpha from sweep: {best_alpha:.2f}")
        else:
            best_alpha = 0.95

        out_rows = []
        for u in tqdm(test_in_df["user"].unique(), desc="Inferring"):
            allow = exclude_consumed_idx(u, user2hist_orig, all_items)
            a_u = conservative_alpha(u, best_alpha, user_prof_norm, user_tagged_counts)
            cf_all = cf_scores_user(u, svdpp, all_items, item2pos)
            cb_all = cb_scores_user(u, user_prof, item_content_tfidf, all_items)
            scores = rrf_blend_from_scores(allow, cf_all, cb_all, a_u)
            idxs = topk_from_scores(scores, allow, K)
            for pos in idxs:
                out_rows.append((int(u), int(all_items[pos])))
        out_df = pd.DataFrame(out_rows, columns=["user_id", "item_id"])
        out_df.to_csv(OUTPUT_TOP20, index=False)
        print(f"Saved top-{K} recommendations to {OUTPUT_TOP20} ({len(out_df):,} rows)")

if __name__ == "__main__":
    main()