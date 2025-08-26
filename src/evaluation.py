# evaluation.py
"""
This file contains all the functions required for evaluating the performance
of the recommendation models, including metrics calculation, score blending,
and plotting the results.
"""
import os
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import sparse

# Here I am importing all necessary configurations and data from other modules.
from config import K, RRF_K, CAND_CF, MIN_TAG_ITEMS, MIN_PROFILE_NORM, ADAPT_S, EVAL_SUMMARY_OUT, EVAL_ALPHA_GRID_OUT, OUTPUT_TOP20, PLOT_DIR, ALPHAS_GRID

def sigmoid(x):
    """Calculates the sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))

def recall_ndcg_hr_at_k(topk_items, gt_set, k):
    """Calculates Recall, NDCG, and Hit Rate for a top-K recommendation list."""
    if not gt_set:
        return 0.0, 0.0, 0.0
    hits = [1 if int(it) in gt_set else 0 for it in topk_items]
    recall = sum(hits) / float(len(gt_set))
    hr = 1.0 if sum(hits) > 0 else 0.0
    dcg = sum(rel / math.log2(r + 2) for r, rel in enumerate(hits))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gt_set), k) + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, ndcg, hr

def pairwise_cosine_from_tfidf(rows):
    """Calculates the average pairwise cosine distance (diversity) from a TF-IDF matrix."""
    if rows.shape[0] < 2:
        return 0.0
    sim = rows @ rows.T
    sim = sim.toarray()
    iu = np.triu_indices(sim.shape[0], k=1)
    return float(1.0 - sim[iu].mean())

def score_to_rrf(scores, k_rrf=RRF_K):
    """Converts a list of scores to reciprocal-rank fusion (RRF) scores."""
    s = np.asarray(scores).astype(float).ravel()
    n = s.size
    if n == 0:
        return s
    if np.all(np.isfinite(s)) and (s.max() == s.min()):
        return np.ones(n) / (k_rrf + 1.0)
    order = np.argsort(-s, kind="mergesort")
    rrf = np.zeros(n, dtype=float)
    for r, pos in enumerate(order):
        rrf[pos] = 1.0 / (k_rrf + r + 1.0)
    if rrf.max() > 0:
        rrf = rrf / rrf.max()
    return rrf

def topk_from_scores(scores, allow_idx, k: int):
    """Selects the top-K item indices from a set of scores, excluding disallowed items."""
    scores = np.asarray(scores).reshape(-1)
    allow_idx = np.asarray(allow_idx, dtype=int)
    if allow_idx.size == 0:
        return np.array([], dtype=int)
    take = min(k, allow_idx.size)
    part = np.argpartition(-scores[allow_idx], take - 1)[:take]
    cand = allow_idx[part]
    order = cand[np.argsort(-scores[cand])]
    return order[:k]

def conservative_alpha(u, base_alpha, user_prof_norm, user_tagged_counts):
    """
    Calculates an adaptive alpha value based on a user's content profile quality.
    This function implements a "conservative" blending strategy.
    """
    prof_norm = user_prof_norm.get(int(u), 0.0)
    tagged_cnt = user_tagged_counts.get(int(u), 0)
    if prof_norm <= 0 or tagged_cnt < MIN_TAG_ITEMS:
        bump = 0.5 + 0.5 * (MIN_TAG_ITEMS - min(tagged_cnt, MIN_TAG_ITEMS)) / float(max(1, MIN_TAG_ITEMS))
        return float(min(1.0, base_alpha + (1.0 - base_alpha) * bump))
    factor = sigmoid((prof_norm - MIN_PROFILE_NORM) * ADAPT_S)
    return float(base_alpha * (0.5 + 0.5 * factor))

def rrf_blend_from_scores(cf_scores, cb_scores, alpha):
    """
    Performs reciprocal-rank fusion (RRF) to blend collaborative filtering and
    content-based scores using a given alpha.
    """
    cf_rrf = score_to_rrf(cf_scores)
    cb_rrf = score_to_rrf(cb_scores)
    return alpha * cf_rrf + (1.0 - alpha) * cb_rrf

def cf_scores_deepfm(u, deepfm, all_items_universe, top_n=CAND_CF):
    """Generates collaborative filtering scores using the DeepFM model."""
    if int(u) not in deepfm.data_info.user2id:
        return np.zeros(len(all_items_universe), dtype=float)
    recs = deepfm.recommend_user(int(u), n_rec=top_n, filter_consumed=True, inner_id=False)
    arr = recs.get(int(u), np.array([], dtype=int)) if isinstance(recs, dict) else np.asarray(recs)
    scores = np.zeros(len(all_items_universe), dtype=float)
    for r, it in enumerate(arr[:top_n]):
        pos = all_items_universe.index(int(it)) if int(it) in all_items_universe else None
        if pos is not None:
            scores[pos] = 1.0 / float(r + 1)
    return scores

def cb_scores_user(u, user_prof, item_content_tfidf):
    """Generates content-based scores using the user's content profile and item TF-IDF matrix."""
    v = user_prof.get(int(u))
    if v is None or not isinstance(v, np.ndarray) or v.size == 0:
        return np.zeros(item_content_tfidf.shape[0], dtype=float)
    s = item_content_tfidf.dot(v)
    return np.asarray(s).ravel()

def pop_scores_user(u, pop_score):
    """Generates popularity scores for a user."""
    return pop_score

def recommend_eval_by_model_recommend_user(model, test_truth_map, all_items_universe, k, users_sample=None, SEED=None):
    """
    Evaluates a model by calling its `recommend_user` method and calculating metrics.
    This is used for evaluating pure collaborative filtering models like ItemCF and DeepFM.
    """
    users = [u for u in test_truth_map.keys() if int(u) in model.data_info.user2id]
    if users_sample is not None and len(users) > users_sample:
        if SEED is not None:
            random.seed(SEED)
        users = random.sample(users, users_sample)
    recalls, ndcgs, hrs, rec_items_all = [], [], [], []
    for u in tqdm(users, desc=f"Eval {model.__class__.__name__} (top-{k})"):
        gt = set(test_truth_map.get(u, []))
        if not gt:
            continue
        recs = model.recommend_user(int(u), n_rec=k, filter_consumed=True, inner_id=False)
        topk = recs.get(int(u), np.array([], dtype=int)) if isinstance(recs, dict) else np.asarray(recs, dtype=int)
        if topk.size == 0:
            continue
        r, n, hr = recall_ndcg_hr_at_k(topk, gt, k)
        recalls.append(r)
        ndcgs.append(n)
        hrs.append(hr)
        rec_items_all.extend(topk.tolist())
    cov = len(set(rec_items_all)) / float(len(all_items_universe)) if rec_items_all else 0.0
    return {"model": model.__class__.__name__, "recall": np.mean(recalls) if recalls else 0.0,
            "ndcg": np.mean(ndcgs) if ndcgs else 0.0, "hitrate": np.mean(hrs) if hrs else 0.0,
            "coverage": cov}

def eval_model_scores(get_scores_fn, label, test_truth, all_items_universe, item_content_tfidf, pop_percentile, user2hist_orig, k=K, users_sample=None, SEED=None):
    """
    Evaluates a recommendation method that returns raw scores, calculating
    multiple metrics including diversity and popularity bias.
    This is used for evaluating popularity and hybrid models.
    """
    users = list(test_truth.keys())
    if users_sample is not None and len(users) > users_sample:
        if SEED is not None:
            random.seed(SEED)
        users = random.sample(users, users_sample)
    recalls, ndcgs, hrs, rec_items_all = [], [], [], []
    diversity_list, pop_percentiles = [], []

    for u in tqdm(users, desc=f"Eval scores {label} (top-{k})"):
        gt = set(test_truth.get(u, []))
        if not gt:
            continue
        consumed = user2hist_orig.get(int(u), set())
        allow = np.array([i for i, it in enumerate(all_items_universe) if int(it) not in consumed], dtype=int)
        if allow.size == 0:
            continue
        scores = get_scores_fn(u)
        idxs = topk_from_scores(scores, allow, k)
        if idxs.size == 0:
            continue
        topk_items = [all_items_universe[i] for i in idxs]
        r, n, hr = recall_ndcg_hr_at_k(topk_items, gt, k)
        recalls.append(r)
        ndcgs.append(n)
        hrs.append(hr)
        rec_items_all.extend(topk_items)
        try:
            rows = item_content_tfidf[idxs]
            diversity_list.append(pairwise_cosine_from_tfidf(rows))
        except Exception:
            pass
        pop_percentiles.append(pop_percentile[idxs].mean() if idxs.size > 0 else 0.0)

    return {
        "model": label,
        "recall": float(np.mean(recalls) if recalls else 0.0),
        "ndcg": float(np.mean(ndcgs) if ndcgs else 0.0),
        "hitrate": float(np.mean(hrs) if hrs else 0.0),
        "coverage": len(set(rec_items_all)) / float(len(all_items_universe)) if rec_items_all else 0.0,
        "diversity": float(np.mean(diversity_list) if diversity_list else 0.0),
        "pop_bias": float(np.mean(pop_percentiles) if pop_percentiles else 0.0),
    }

def plot_results(summary_df, alpha_df):
    """
    Here I am generating and saving various plots to visualize the evaluation results.
    This includes accuracy, coverage & diversity, popularity bias, and the alpha sweep.
    """
    # Plotting accuracy bars
    plt.figure(figsize=(9, 5))
    x = np.arange(len(summary_df))
    plt.bar(x - 0.2, summary_df["recall"].values, width=0.4, label=f"Recall@{K}")
    plt.bar(x + 0.2, summary_df["ndcg"].values, width=0.4, label=f"NDCG@{K}")
    plt.xticks(x, summary_df["model"].values, rotation=15, ha="right")
    plt.ylabel("Score")
    plt.title("RQ2: Top-K Ranking Quality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rq2_accuracy_bars.png"))
    plt.close()

    # Plotting coverage & diversity
    plt.figure(figsize=(9, 5))
    plt.bar(x - 0.2, summary_df["coverage"].values, width=0.4, label=f"Catalog Coverage@{K}")
    plt.bar(x + 0.2, summary_df["diversity"].fillna(0).values, width=0.4, label=f"Intra-List Diversity@{K}")
    plt.xticks(x, summary_df["model"].values, rotation=15, ha="right")
    plt.ylabel("Score")
    plt.title("RQ2: Coverage & Diversity Trade-offs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rq2_coverage_diversity_bars.png"))
    plt.close()

    # Plotting popularity bias
    plt.figure(figsize=(9, 5))
    plt.bar(x, summary_df["pop_bias"].fillna(0).values)
    plt.xticks(x, summary_df["model"].values, rotation=15, ha="right")
    plt.ylabel("Avg popularity percentile (higher ⇒ more head)")
    plt.title("RQ2: Popularity Bias")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rq2_pop_bias.png"))
    plt.close()

    # Plotting alpha-sweep curves
    if not alpha_df.empty:
        plt.figure(figsize=(8, 5))
        plt.plot(alpha_df["alpha"].values, alpha_df["recall"].values, marker='o', label=f"Recall@{K}")
        plt.plot(alpha_df["alpha"].values, alpha_df["ndcg"].values, marker='s', label=f"NDCG@{K}")
        plt.xlabel("Alpha (CF weight)")
        plt.ylabel("Score")
        plt.title("RQ2: DeepFM+Tags α-sweep")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "rq2_hybrid_alpha_sweep.png"))
        plt.close()

    print("Saved RQ2 plots to:", PLOT_DIR)
