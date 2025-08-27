import numpy as np
import pandas as pd
import math
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from libreco.data import DatasetPure

# --- HERE I AM IMPORTING GLOBAL SETTINGS AND HELPER FUNCTIONS ---
from config import K, K_LIST, PLOT_DIR, SEED, ALPHAS_GRID, EVAL_SUMMARY_OUT, RRF_K, MIN_TAG_ITEMS, MIN_PROFILE_NORM
from data_prep import create_tfidf_matrix
from models import train_svdpp_model, train_itemcf_model

# --- HELPER FUNCTIONS FOR EVALUATION METRICS ---

def recall_ndcg_hr_at_k(topk_items, gt_set, k=K):
    """
    HERE I AM COMPUTING RECALL, NDCG, AND HIT RATE AT K.
    """
    if not gt_set: return 0.0, 0.0, 0.0
    hits = [1 if int(it) in gt_set else 0 for it in topk_items]
    recall = sum(hits) / float(len(gt_set))
    hr = 1.0 if sum(hits) > 0 else 0.0
    dcg = sum(rel / math.log2(r + 2) for r, rel in enumerate(hits))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gt_set), k) + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, ndcg, hr

def pairwise_cosine_from_tfidf(rows):
    """
    HERE I AM CALCULATING INTRA-LIST DIVERSITY FROM TF-IDF VECTORS.
    """
    # rows: csr matrix [k x F]
    if rows.shape[0] < 2: return 0.0
    sim = rows @ rows.T
    sim = sim.toarray()
    iu = np.triu_indices(sim.shape[0], k=1)
    return float(1.0 - sim[iu].mean())  # diversity = 1 - mean similarity

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def score_to_rrf(scores, k_rrf=RRF_K):
    """
    HERE I AM CONVERTING A LIST OF SCORES INTO RECIPROCAL RANK FUSION (RRF) SCORES.
    """
    n = len(scores)
    s = np.asarray(scores).astype(float)
    if np.all(np.isfinite(s)) and (s.max() == s.min()):
        return np.ones(n) / (k_rrf + 1.0)
    order = np.argsort(-s, kind="mergesort")
    rrf = np.zeros(n, dtype=float)
    for r, pos in enumerate(order):
        rrf[pos] = 1.0 / (k_rrf + r + 1.0)
    if rrf.max() > 0:
        rrf = rrf / rrf.max()
    return rrf

def rrf_blend_from_scores(allow_idx, cf_scores, cb_scores, alpha):
    """
    HERE I AM BLENDING COLLABORATIVE AND CONTENT-BASED SCORES USING RRF.
    """
    cf_rrf = score_to_rrf(cf_scores)
    cb_rrf = score_to_rrf(cb_scores)
    blended = alpha * cf_rrf + (1.0 - alpha) * cb_rrf
    return blended

def conservative_alpha(u, base_alpha, user_prof_norm, user_tagged_counts):
    """
    HERE I AM ADAPTING THE HYBRID ALPHA FOR A SPECIFIC USER BASED ON THEIR CONTENT PROFILE STRENGTH.
    """
    prof_norm = user_prof_norm.get(int(u), 0.0)
    tagged_cnt = user_tagged_counts.get(int(u), 0)
    if prof_norm <= 0 or tagged_cnt < MIN_TAG_ITEMS:
        bump = 0.5 + 0.5 * (MIN_TAG_ITEMS - min(tagged_cnt, MIN_TAG_ITEMS)) / float(max(1, MIN_TAG_ITEMS))
        return float(min(1.0, base_alpha + (1.0 - base_alpha) * bump))
    factor = sigmoid((prof_norm - MIN_PROFILE_NORM) * 2)
    return float(base_alpha * (0.5 + 0.5 * factor))

# --- HERE I AM DEFINING CORE EVALUATION LOGIC ---

def recommend_eval_by_model_recommend_user(model, test_truth_map, k=K, users_sample=None):
    """
    HERE I AM EVALUATING LIBRECO MODELS USING THEIR BUILT-IN RECOMMEND FUNCTION.
    """
    users = [u for u in test_truth_map.keys() if int(u) in model.data_info.user2id]
    if users_sample is not None and len(users) > users_sample:
        random.seed(SEED)
        users = random.sample(users, users_sample)

    recalls, ndcgs, hrs = [], [], []
    rec_items_all = []
    for u in tqdm(users, desc=f"Eval {model.__class__.__name__} (top-{k})"):
        gt = set(test_truth_map.get(u, []))
        if not gt: continue
        recs = model.recommend_user(int(u), n_rec=k, filter_consumed=True, inner_id=False)
        if isinstance(recs, dict):
            topk = recs.get(int(u), np.array([], dtype=int))
        else:
            topk = np.asarray(recs, dtype=int)
        if topk.size == 0:
            continue
        r, n, hr = recall_ndcg_hr_at_k(topk, gt, k)
        recalls.append(r); ndcgs.append(n); hrs.append(hr)
        rec_items_all.extend(topk.tolist())
    return {
        "model": model.__class__.__name__,
        "recall": float(np.mean(recalls) if recalls else 0.0),
        "ndcg": float(np.mean(ndcgs) if ndcgs else 0.0),
        "hitrate": float(np.mean(hrs) if hrs else 0.0),
        "coverage": len(set(rec_items_all)) / float(len(model.data_info.item_unique_vals)) if rec_items_all else 0.0,
        "diversity": None,
        "pop_bias": None
    }

def eval_model_scores(get_scores_fn, label, test_truth, train_plus_eval_df, item_content_tfidf, all_items, item2pos, user2hist_orig, k=K, users_sample=None):
    """
    HERE I AM EVALUATING MODELS BASED ON THEIR RAW SCORES. This provides more control for hybrid models.
    """
    users = list(test_truth.keys())
    if users_sample is not None and len(users) > users_sample:
        random.seed(SEED)
        users = random.sample(users, users_sample)

    recalls, ndcgs, hrs = [], [], []
    rec_items_all = []
    diversity_list = []
    pop_percentiles = []

    def exclude_consumed_idx(u_orig):
        consumed = user2hist_orig.get(int(u_orig), set())
        allow = np.array([i for i, it in enumerate(all_items) if int(it) not in consumed], dtype=int)
        return allow
    
    def topk_from_scores(scores, allow_idx, k: int):
        scores = np.asarray(scores).reshape(-1)
        allow_idx = np.asarray(allow_idx, dtype=int)
        n_items = len(all_items)

        if scores.size != n_items:
            if scores.size == 1:
                scores = np.full(n_items, float(scores[0]), dtype=float)
            elif scores.size == allow_idx.size and allow_idx.size > 0:
                tmp = np.full(n_items, -np.inf, dtype=float)
                tmp[allow_idx] = scores
                scores = tmp
            elif scores.ravel().size == n_items:
                scores = scores.ravel()
            else:
                scores = np.full(n_items, -np.inf, dtype=float)

        if allow_idx.size == 0:
            return np.array([], dtype=int)

        take = min(k, allow_idx.size)
        if allow_idx.size == 1:
            return allow_idx.copy()

        part = np.argpartition(-scores[allow_idx], take - 1)[:take]
        cand = allow_idx[part]
        order = cand[np.argsort(-scores[cand])]
        return order[:k]

    # Popularity from train + eval
    pop_counts = train_plus_eval_df['item'].value_counts().to_dict()
    all_freqs = np.array([pop_counts.get(int(it), 0) for it in all_items], dtype=float)
    if all_freqs.max() > 0:
        pop_rank = np.argsort(np.argsort(-all_freqs))
        pop_percentile = 1.0 - (pop_rank / float(len(all_freqs) - 1 + 1e-12))
    else:
        pop_percentile = np.zeros_like(all_freqs)

    for u in tqdm(users, desc=f"Eval scores {label} (top-{k})"):
        gt = set(test_truth.get(u, []))
        if not gt: continue
        allow = exclude_consumed_idx(u)
        if allow.size == 0: continue
        scores = get_scores_fn(u)
        topk_idx_sorted = topk_from_scores(scores, allow, k)
        if topk_idx_sorted.size == 0: continue
        topk_items = all_items[topk_idx_sorted]
        r, n, hr = recall_ndcg_hr_at_k(topk_items, gt, k)
        recalls.append(r); ndcgs.append(n); hrs.append(hr)
        rec_items_all.extend(topk_items.tolist())

        try:
            rows = item_content_tfidf[topk_idx_sorted]
            diversity_list.append(pairwise_cosine_from_tfidf(rows))
        except Exception:
            pass
        pp = pop_percentile[topk_idx_sorted].mean() if topk_idx_sorted.size > 0 else 0.0
        pop_percentiles.append(pp)

    return {
        "model": label,
        "recall": float(np.mean(recalls) if recalls else 0.0),
        "ndcg": float(np.mean(ndcgs) if ndcgs else 0.0),
        "hitrate": float(np.mean(hrs) if hrs else 0.0),
        "coverage": len(set(rec_items_all)) / float(len(all_items)) if rec_items_all else 0.0,
        "diversity": float(np.mean(diversity_list) if diversity_list else 0.0),
        "pop_bias": float(np.mean(pop_percentiles) if pop_percentiles else 0.0),
    }

def generate_plots(summary_df, hgrid):
    """
    HERE I AM GENERATING ALL THE VISUALIZATIONS FOR THE EVALUATION RESULTS.
    """
    print("\n--- STEP 9: Result plots ---")
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    x = np.arange(len(summary_df))
    
    # A) Accuracy bars
    plt.figure(figsize=(9, 5))
    plt.bar(x - 0.2, summary_df["recall"].values, width=0.4, label=f"Recall@{K}")
    plt.bar(x + 0.2, summary_df["ndcg"].values, width=0.4, label=f"NDCG@{K}")
    plt.xticks(x, summary_df["model"].values, rotation=15, ha="right")
    plt.ylabel("Score"); plt.title("Top-K Ranking Quality"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "res_accuracy_bars.png")); plt.close()

    # B) Coverage & Diversity
    plt.figure(figsize=(9, 5))
    plt.bar(x - 0.2, summary_df["coverage"].values, width=0.4, label=f"Catalog Coverage@{K}")
    plt.bar(x + 0.2, summary_df["diversity"].fillna(0).values, width=0.4, label=f"Intra-List Diversity@{K}")
    plt.xticks(x, summary_df["model"].values, rotation=15, ha="right")
    plt.ylabel("Score"); plt.title("Coverage & Diversity Trade-offs"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "res_coverage_diversity_bars.png")); plt.close()

    # C) Popularity bias
    plt.figure(figsize=(9, 5))
    plt.bar(x, summary_df["pop_bias"].fillna(0).values)
    plt.xticks(x, summary_df["model"].values, rotation=15, ha="right")
    plt.ylabel("Avg popularity percentile (higher=>head)"); plt.title("Popularity Bias of Recommendations")
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "res_pop_bias.png")); plt.close()

    # D) alpha sweep curves
    if hgrid:
        dfh = pd.DataFrame(hgrid)
        plt.figure(figsize=(8, 5))
        plt.plot(dfh["alpha"].values, dfh["recall"].values, marker='o', label=f"Recall@{K}")
        plt.plot(dfh["alpha"].values, dfh["ndcg"].values, marker='s', label=f"NDCG@{K}")
        plt.xlabel("Alpha (CF weight)"); plt.ylabel("Score")
        plt.title("Hybrid α-sweep"); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "res_hybrid_alpha_sweep.png")); plt.close()

    print("Saved plots.")