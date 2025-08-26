# data_prep.py
"""
This file handles the loading, cleaning, and preparation of the datasets.
It includes functions for reading CSVs, detecting column names,
and splitting the data for training, evaluation, and testing.
"""
import os
import re
import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from libreco.data import DatasetPure, DatasetFeat, random_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Here I am importing all necessary configurations.
from config import (
    TRAIN_INTERACTIONS, TEST_IN_FILE, GAMES_FILE, SEED, K,
    MIN_TAG_ITEMS, MIN_PROFILE_NORM, ADAPT_S
)

# --- HELPERS (Copied from notebook for clarity and organization) ---

DELIMS_RE = re.compile(r"[\,\|/;]+")
PHRASE_SYNONYMS = {
    "fps":"first_person_shooter","tps":"third_person_shooter",
    "rpg":"role_playing","arpg":"action_rpg",
    "roguelike":"rogue_like","rogue_like":"rogue_like","rogue_lite":"rogue_lite",
}
def normalize_phrase(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+"," ", s).replace("-", "_").replace(" ","_")
    s = re.sub(r"[^a-z0-9_]+","", s)
    return PHRASE_SYNONYMS.get(s, s) if s else ""

def parse_tags_or_genres(raw: str) -> list:
    if pd.isna(raw): return []
    parts = [p for p in DELIMS_RE.split(str(raw)) if p and p.strip()]
    toks, seen = [], set()
    for p in parts:
        tok = normalize_phrase(p)
        if tok and tok not in seen:
            seen.add(tok); toks.append(tok)
    return toks

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def load_csv(path):
    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} rows; columns={list(df.columns)[:8]}{'...' if df.shape[1]>8 else ''}")
    return df

def detect_columns(df):
    cols = [c.lower() for c in df.columns]
    def pick(cands):
        for cand in cands:
            if cand in cols: return df.columns[cols.index(cand)]
        return None
    return (
        pick(("user_id","user","userid","user id","steamid")),
        pick(("item_id","itemid","item","game_id","appid","app_id","id")),
        pick(("playtime","play_time","play_minutes","playtime_forever","rating","label"))
    )

def exclude_consumed_idx(u, user2hist_orig, all_items_universe):
    consumed = user2hist_orig.get(int(u), set())
    allow = np.array([i for i, it in enumerate(all_items_universe) if int(it) not in consumed], dtype=int)
    return allow

def build_profiles(interactions_df, inter_df, item_universe_pos, item_content_tfidf):
    wdf = interactions_df.merge(inter_df[['user','item','playtime']], on=['user','item'], how='left')
    wdf['w'] = np.clip(np.log1p(wdf['playtime'].astype(float)), 0.0, 10.0)
    user_item_w = defaultdict(list)
    for _, r in wdf.iterrows():
        u, it, w = int(r['user']), int(r['item']), float(r['w'])
        pos = item_universe_pos.get(it, None)
        if pos is not None: user_item_w[u].append((pos, w))
    user_prof, user_prof_norm, user_tagged_counts = {}, {}, {}
    for u, pairs in tqdm(user_item_w.items(), desc="Building user profiles"):
        if not pairs:
            user_prof[u] = None; user_prof_norm[u] = 0.0; user_tagged_counts[u] = 0
            continue
        idxs = np.array([p[0] for p in pairs], dtype=int)
        ws   = np.array([p[1] for p in pairs], dtype=float)
        sw   = ws.sum()
        user_tagged_counts[u] = int((item_content_tfidf[idxs].getnnz(axis=1) > 0).sum())
        if sw <= 0:
            user_prof[u] = None; user_prof_norm[u] = 0.0; continue
        v = (item_content_tfidf[idxs].T.dot(ws) / (sw + 1e-12)).ravel()
        n = np.linalg.norm(v)
        if n > 0: v = v / n
        user_prof[u] = v; user_prof_norm[u] = float(n)
    return user_prof, user_prof_norm, user_tagged_counts

def load_and_prepare_data():
    """
    Here is the main function for loading and preparing all data files.
    This function handles cleaning, splitting, and building the necessary
    data objects for the recommendation pipeline.
    """
    print("\n=== RQ2 / STEP 1: Load datasets ===")
    inter_df = load_csv(TRAIN_INTERACTIONS)
    u_col, i_col, p_col = detect_columns(inter_df)
    print(f"Detected -> user: {u_col}, item: {i_col}, playtime/label: {p_col}")

    # Here I am renaming, cleaning, and type-casting the data.
    inter_df = inter_df.rename(columns={u_col:"user", i_col:"item"})
    
    # Robust conversion to numeric, handling errors.
    inter_df['user'] = pd.to_numeric(inter_df['user'], errors='coerce')
    inter_df['item'] = pd.to_numeric(inter_df['item'], errors='coerce')
    inter_df = inter_df.dropna(subset=['user', 'item'])
    
    inter_df["user"] = inter_df["user"].astype(np.int64)
    inter_df["item"] = inter_df["item"].astype(np.int64)
    assert inter_df['user'].dtype == np.int64, "User column should be of integer type."
    assert inter_df['item'].dtype == np.int64, "Item column should be of integer type."
    
    if p_col: inter_df["playtime"] = inter_df[p_col].astype(float)
    else:     inter_df["playtime"] = 1.0
    inter_df["label"] = (inter_df["playtime"] > 0).astype(int)
    inter_df = inter_df.drop_duplicates(["user","item"], keep="last")
    print(f"After cleaning: {len(inter_df):,} rows, users={inter_df['user'].nunique():,}, items={inter_df['item'].nunique():,}")

    games_df = load_csv(GAMES_FILE)

    test_in_df = None
    if os.path.exists(TEST_IN_FILE):
        test_in_df = load_csv(TEST_IN_FILE)
        tu, ti, _ = detect_columns(test_in_df)
        test_in_df = test_in_df.rename(columns={tu:'user', ti:'item'})
        test_in_df['user'] = pd.to_numeric(test_in_df['user'], errors='coerce')
        test_in_df['item'] = pd.to_numeric(test_in_df['item'], errors='coerce')
        test_in_df = test_in_df.dropna(subset=['user', 'item'])
        test_in_df['user'] = test_in_df['user'].astype(np.int64)
        test_in_df['item'] = test_in_df['item'].astype(np.int64)

    print("\n=== RQ2 / STEP 2: random_split ===")
    train_df, eval_df, test_df = random_split(inter_df[['user','item','label']], multi_ratios=[0.8,0.1,0.1], seed=SEED)
    print(f"Split -> train={len(train_df):,}, eval={len(eval_df):,}, test={len(test_df):,}")

    train_plus_eval_df = pd.concat([train_df, eval_df], ignore_index=True)
    all_items_universe = sorted(train_plus_eval_df['item'].unique().tolist())
    item_universe_pos = {it:i for i,it in enumerate(all_items_universe)}

    print("\n=== RQ2 / STEP 3: Build DatasetFeat (DeepFM) and DatasetPure (ItemCF) ===")
    dfm_train_data, dfm_datainfo = DatasetFeat.build_trainset(train_df)
    dfm_eval_data = DatasetFeat.build_evalset(eval_df)
    icf_train_data, icf_datainfo = DatasetPure.build_trainset(train_df)
    
    print("\n=== RQ2 / STEP 4: TF-IDF over tags/genres (for hybrid & diversity) ===")
    cg = {c.lower(): c for c in games_df.columns}
    g_item = cg.get('item_id') or cg.get('appid') or cg.get('app_id') or cg.get('id')
    g_tags = cg.get('tags') or cg.get('tag')
    g_genre = cg.get('genres') or cg.get('genre')

    meta = games_df[[g_item] + ([g_tags] if g_tags else []) + ([g_genre] if g_genre else [])].copy()
    meta = meta.rename(columns={g_item:'item', g_tags:'tags', g_genre:'genres'})
    meta['item'] = pd.to_numeric(meta['item'], errors='coerce')
    meta = meta.dropna(subset=['item'])
    meta['item'] = meta['item'].astype(np.int64)
    meta = meta[meta['item'].isin(all_items_universe)].copy()
    meta['toks_tags'] = meta['tags'].apply(parse_tags_or_genres) if 'tags' in meta.columns else [[]]*len(meta)
    meta['toks_genres'] = meta['genres'].apply(parse_tags_or_genres) if 'genres' in meta.columns else [[]]*len(meta)
    meta_map = meta.set_index('item')

    content_docs = []
    miss_tags = miss_gen = 0
    for it in all_items_universe:
        if it in meta_map.index:
            r = meta_map.loc[it]
            tt = r['toks_tags'] if 'toks_tags' in meta_map.columns else []
            gg = r['toks_genres'] if 'toks_genres' in meta_map.columns else []
        else:
            tt, gg = [], []
        if not tt: miss_tags += 1
        if not gg: miss_gen += 1
        prefixed = [f"t:{t}" for t in tt] + [f"g:{g}" for g in gg]
        content_docs.append(" ".join(prefixed))
    print(f"Items missing tags: {miss_tags:,}/{len(all_items_universe):,}; missing genres: {miss_gen:,}/{len(all_items_universe):,}")

    vec = TfidfVectorizer(token_pattern=r"(?u)\b[\w:]+\b", lowercase=False, sublinear_tf=True, use_idf=True, norm='l2', min_df=2)
    item_content_tfidf = vec.fit_transform(content_docs).tocsr()
    
    pop_counts = train_plus_eval_df['item'].value_counts().to_dict()
    pop_score = np.array([pop_counts.get(int(it), 0) for it in all_items_universe], dtype=float)
    pop_score = pop_score / (pop_score.max() or 1.0)
    
    print("\n=== RQ2 / STEP 8: User histories & content profiles (log-playtime weighted) ===")
    user2hist_orig = train_plus_eval_df.groupby("user")["item"].agg(lambda x: set(map(int, x))).to_dict()
    user_prof, user_prof_norm, user_tagged_counts = build_profiles(train_plus_eval_df, inter_df, item_universe_pos, item_content_tfidf)

    return (
        inter_df, games_df, test_in_df, train_df, eval_df, test_df,
        dfm_train_data, dfm_eval_data, dfm_datainfo,
        icf_train_data, icf_datainfo,
        all_items_universe, item_universe_pos,
        item_content_tfidf, pop_score, user2hist_orig,
        user_prof, user_prof_norm, user_tagged_counts,
        train_plus_eval_df
    )
