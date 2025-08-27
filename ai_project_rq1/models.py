# C:\Users\kerne\Pictures\Rq_1_recommender_system\models.py

import tensorflow as tf
from libreco.algorithms import SVDpp, ItemCF
import numpy as np # <-- MOVE THIS IMPORT HERE

# --- HERE I AM IMPORTING THE GLOBAL SETTINGS FROM THE CONFIG FILE ---
from config import SEED, TF_SESS_CONFIG, CPU_CORES

def train_svdpp_model(data_info, train_data, best_params):
    """
    HERE I AM TRAINING THE FINAL SVD++ MODEL WITH THE OPTIMIZED HYPERPARAMETERS.
    """
    print("\nRetraining SVD++ on train+eval with best params ...")
    try:
        # HERE I AM RESETTING THE TENSORFLOW GRAPH to avoid conflicts.
        tf.compat.v1.reset_default_graph()
    except Exception:
        pass

    svdpp = SVDpp(
        task="ranking",
        data_info=data_info,
        embed_size=best_params[0],
        n_epochs=best_params[3],
        lr=best_params[1],
        batch_size=2048,
        num_neg=best_params[2],
        tf_sess_config=TF_SESS_CONFIG,
        seed=SEED,
    )
    svdpp.fit(train_data, neg_sampling=True, verbose=0, shuffle=True)
    return svdpp

def train_itemcf_model(data_info, train_data):
    """
    HERE I AM TRAINING THE ITEMCF BASELINE MODEL.
    """
    print("\n--- STEP 5b: Train ItemCF (ItemKNN) baseline ---")
    itemcf = ItemCF(
        task="ranking",
        data_info=data_info,
        sim_type="cosine",
        k_sim=50,
        store_top_k=True,
        block_size=None,
        num_threads=CPU_CORES,
        min_common=1,
        mode="invert",
        seed=SEED,
    )
    itemcf.fit(train_data, neg_sampling=True, verbose=0)
    return itemcf

def get_best_svdpp_params(train_data, eval_truth, data_info):
    """
    HERE I AM PERFORMING A HYPERPARAMETER SWEEP TO FIND THE BEST SETTINGS FOR SVD++.
    """
    print("\n--- STEP 5: Hyperparameter Tuning for SVD++ (recommend_user-based eval) ---")
    param_grid = {
        "embed_size": [32],
        "lr": [5e-4],
        "num_neg": [5],
        "n_epochs": [10],
    }
    best_score = -np.inf # <-- THIS LINE NOW WORKS
    best_params = None
    EVAL_SAMPLE_USERS = min(2000, len(eval_truth))

    from itertools import product
    from evaluation import recommend_eval_by_model_recommend_user
    # REMOVE THE LOCAL IMPORT `import numpy as np`
    
    for embed_size, lr, num_neg, n_epochs in product(param_grid["embed_size"], param_grid["lr"], param_grid["num_neg"], param_grid["n_epochs"]):
        print(f"\n>>> SVD++ trial: embed={embed_size}, lr={lr}, num_neg={num_neg}, epochs={n_epochs}")
        try:
            tf.compat.v1.reset_default_graph()
        except Exception:
            pass

        m = SVDpp(
            task="ranking",
            data_info=data_info,
            embed_size=embed_size,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=2048,
            num_neg=num_neg,
            tf_sess_config=TF_SESS_CONFIG,
            seed=SEED,
        )
        m.fit(train_data, neg_sampling=True, verbose=0, shuffle=True)
        r = recommend_eval_by_model_recommend_user(m, eval_truth, users_sample=EVAL_SAMPLE_USERS)
        print(f"Trial -> recall@{20}: {r['recall']:.4f}, ndcg@{20}: {r['ndcg']:.4f}")
        if r["ndcg"] > best_score:
            best_score = r["ndcg"]
            best_params = (embed_size, lr, num_neg, n_epochs)

    print(f"\nBest SVD++ params (by ndcg@{20} on eval sample): {best_params}, ndcg={best_score:.4f}")
    return best_params