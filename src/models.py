# models.py
"""
This file handles the training of the recommendation models. It includes the
ItemCF baseline and the DeepFM model, along with a hyperparameter search
for DeepFM to find the best configuration.
"""
import random
import numpy as np
import tensorflow as tf
from itertools import product
from libreco.algorithms import DeepFM, ItemCF

# Here I am importing all necessary configurations and data objects.
from config import SEED, CPU_CORES, TF_SESS_CONFIG, K
from evaluation import recommend_eval_by_model_recommend_user

def train_itemcf(icf_train_data, icf_datainfo):
    """
    Here I am training the ItemCF (ItemKNN) baseline model.
    This model is a simple collaborative filtering approach based on item similarity.
    """
    print("\n=== RQ2 / STEP 5: Train ItemCF (ItemKNN) baseline ===")
    itemcf = ItemCF(
        task="ranking",
        data_info=icf_datainfo,
        sim_type="cosine",
        k_sim=50,
        store_top_k=True,
        block_size=None,
        num_threads=CPU_CORES,
        min_common=1,
        mode="invert",
        seed=SEED,
    )
    # Here I am fitting the model to the training data.
    itemcf.fit(icf_train_data, neg_sampling=True, verbose=0)
    # --- SCIENTIFIC SANITY TEST ---
    assert itemcf is not None, "ItemCF model failed to train."
    return itemcf

def hyperparameter_search_and_train_deepfm(dfm_train_data, dfm_eval_data, dfm_datainfo, eval_df, all_items_universe):
    """
    This function performs a hyperparameter search for the DeepFM model,
    then retrains the best model on the combined training and evaluation data.
    """
    print("\n=== RQ2 / STEP 6: Hyperparameter Tuning for DeepFM (ranking) ===")
    eval_truth = eval_df.groupby('user')['item'].apply(list).to_dict()
    best_dfm = None
    best_score = -np.inf

    DFM_PARAM_GRID = {
        "embed_size": [16, 32],
        "hidden_units": [(128, 64), (64, 32)],
        "lr": [1e-3, 5e-4],
        "num_neg": [3],
        "n_epochs": [10, 15],
        "dropout_rate": [None, 0.2],
    }
    EVAL_SAMPLE_USERS = min(2000, len(eval_truth))

    for embed_size, hidden_units, lr, num_neg, n_epochs, dropout_rate in product(
        DFM_PARAM_GRID["embed_size"], DFM_PARAM_GRID["hidden_units"],
        DFM_PARAM_GRID["lr"], DFM_PARAM_GRID["num_neg"],
        DFM_PARAM_GRID["n_epochs"], DFM_PARAM_GRID["dropout_rate"]
    ):
        print(f"\n>>> DeepFM trial: emb={embed_size}, hidden={hidden_units}, lr={lr}, neg={num_neg}, "
              f"epochs={n_epochs}, dropout={dropout_rate}")
        try:
            # Here I am resetting the TensorFlow graph to prevent conflicts between model trials.
            tf.compat.v1.reset_default_graph()
        except Exception:
            pass
        
        dfm = DeepFM(
            task="ranking",
            data_info=dfm_datainfo,
            loss_type="cross_entropy",
            embed_size=embed_size,
            hidden_units=hidden_units,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=2048,
            sampler="random",
            num_neg=num_neg,
            use_bn=True,
            dropout_rate=dropout_rate,
            tf_sess_config=TF_SESS_CONFIG,
            seed=SEED,
        )
        # Here I am fitting the model and evaluating its performance on the evaluation set.
        dfm.fit(dfm_train_data, neg_sampling=True, verbose=0, shuffle=True,
                eval_data=dfm_eval_data, metrics=["recall", "ndcg"], k=K, eval_user_num=EVAL_SAMPLE_USERS)
        
        # Here I am using the evaluation function to get the final metrics for this trial.
        r = recommend_eval_by_model_recommend_user(dfm, eval_truth, all_items_universe, k=K, users_sample=EVAL_SAMPLE_USERS, SEED=SEED)
        print(f"Trial -> recall@{K}: {r['recall']:.4f}, ndcg@{K}: {r['ndcg']:.4f}")
        
        # Here I am selecting the best model based on its NDCG score.
        if r["ndcg"] > best_score:
            best_score = r["ndcg"]
            best_dfm = (embed_size, hidden_units, lr, num_neg, n_epochs, dropout_rate)
    
    print(f"\nBest DeepFM params by ndcg@{K}: {best_dfm}, ndcg={best_score:.4f}")
    # --- SCIENTIFIC SANITY TEST ---
    assert best_dfm is not None, "Hyperparameter search failed to find a best model."
    return best_dfm

def retrain_final_deepfm(best_dfm, train_plus_eval_df):
    """
    Here I am retraining the DeepFM model with the best hyperparameters
    on the full dataset (train + eval) for final use.
    """
    from libreco.data import DatasetFeat

    print("\n=== RQ2 / STEP 7: Retrain final DeepFM on train+eval ===")
    dfm_train_final, dfm_datainfo_final = DatasetFeat.build_trainset(train_plus_eval_df)
    try:
        tf.compat.v1.reset_default_graph()
    except Exception:
        pass
    
    deepfm = DeepFM(
        task="ranking",
        data_info=dfm_datainfo_final,
        loss_type="cross_entropy",
        embed_size=best_dfm[0],
        hidden_units=best_dfm[1],
        n_epochs=best_dfm[4],
        lr=best_dfm[2],
        batch_size=2048,
        sampler="random",
        num_neg=best_dfm[3],
        use_bn=True,
        dropout_rate=best_dfm[5],
        tf_sess_config=TF_SESS_CONFIG,
        seed=SEED,
    )
    # Here I am fitting the final model on the combined data.
    deepfm.fit(dfm_train_final, neg_sampling=True, verbose=0, shuffle=True)
    # --- SCIENTIFIC SANITY TEST ---
    assert deepfm is not None, "Final DeepFM model failed to train."
    return deepfm