import os
import re
import pandas as pd
from libreco.data import random_split, DatasetPure
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# --- HERE I AM IMPORTING THE GLOBAL SETTINGS FROM THE CONFIG FILE ---
from config import *

# --- HERE I AM DEFINING HELPER FUNCTIONS FOR DATA LOADING AND PROCESSING ---

def load_csv(path):
    """
    HERE I AM LOADING A CSV FILE INTO A PANDAS DATAFRAME.
    """
    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} rows, columns: {list(df.columns)[:8]}{'...' if len(df.columns) > 8 else ''}")
    return df

def detect_columns(df):
    """
    HERE I AM AUTOMATICALLY DETECTING THE USER, ITEM, AND PLAYTIME COLUMNS.
    """
    cols = [c.lower() for c in df.columns]
    def pick(cands):
        for cand in cands:
            if cand in cols: return df.columns[cols.index(cand)]
        return None
    return (
        pick(("user_id", "user", "userid", "user id", "steamid")),
        pick(("item_id", "itemid", "item", "game_id", "appid", "app_id", "id")),
        pick(("playtime", "play_time", "play_minutes", "playtime_forever", "rating", "label"))
    )

DELIMS_RE = re.compile(r"[\,\|/;]+")
PHRASE_SYNONYMS = {
    "fps": "first_person_shooter", "tps": "third_person_shooter",
    "rpg": "role_playing", "arpg": "action_rpg",
    "roguelike": "rogue_like", "rogue_like": "rogue_like", "rogue_lite": "rogue_lite",
}
def normalize_phrase(s: str) -> str:
    """
    HERE I AM NORMALIZING A STRING (TAG OR GENRE) FOR CONSISTENT TOKENIZATION.
    """
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s).replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return PHRASE_SYNONYMS.get(s, s) if s else ""

def parse_tags_or_genres(raw: str) -> list:
    """
    HERE I AM PARSING RAW TAG/GENRE STRINGS INTO A LIST OF NORMALIZED TOKENS.
    """
    if pd.isna(raw): return []
    parts = [p for p in DELIMS_RE.split(str(raw)) if p and p.strip()]
    toks, seen = [], set()
    for p in parts:
        tok = normalize_phrase(p)
        if tok and tok not in seen:
            seen.add(tok)
            toks.append(tok)
    return toks

def prepare_data():
    """
    HERE I AM ORCHESTRATING THE ENTIRE DATA PREPARATION PIPELINE.
    This function will load, clean, split, and build the datasets for training.
    """
    print("\n--- STEP 1: Load and clean datasets ---")
    inter_df = load_csv(TRAIN_INTERACTIONS)
    u_col, i_col, p_col = detect_columns(inter_df)
    print(f"Detected -> user: {u_col}, item: {i_col}, playtime/label: {p_col}")

    # HERE I AM RENAMING AND CLEANING THE INTERACTION DATA.
    inter_df = inter_df.rename(columns={u_col: "user", i_col: "item"})
    inter_df = inter_df.dropna(subset=["user", "item"])
    inter_df["user"] = inter_df["user"].astype(int)
    inter_df["item"] = inter_df["item"].astype(int)
    if p_col:
        inter_df["playtime"] = inter_df[p_col].astype(float)
    else:
        inter_df["playtime"] = 1.0
    inter_df["label"] = (inter_df["playtime"] > 0).astype(int)
    inter_df = inter_df.drop_duplicates(["user", "item"], keep="last")
    print(f"After cleaning: {len(inter_df):,} rows, users={inter_df['user'].nunique():,}, items={inter_df['item'].nunique():,}")

    games_df = load_csv(GAMES_FILE)

    # HERE I AM CHECKING FOR AN OPTIONAL TEST INFERENCE FILE.
    if os.path.exists(TEST_IN_FILE):
        test_in_df = load_csv(TEST_IN_FILE)
        tu, ti, _ = detect_columns(test_in_df)
        test_in_df = test_in_df.rename(columns={tu: 'user', ti: 'item'})
        test_in_df['user'] = test_in_df['user'].astype(int)
        test_in_df['item'] = test_in_df['item'].astype(int)
    else:
        test_in_df = None

    print("\n--- STEP 2: Randomly splitting data (0.8 / 0.1 / 0.1) ---")
    train_df, eval_df, test_df = random_split(inter_df[['user', 'item', 'label']], multi_ratios=[0.8, 0.1, 0.1], seed=SEED)
    print(f"Split -> train={len(train_df):,}, eval={len(eval_df):,}, test={len(test_df):,}")

    print("\n--- STEP 3: Building Libreco DatasetPure objects ---")
    train_data, data_info = DatasetPure.build_trainset(train_df)
    eval_data = DatasetPure.build_evalset(eval_df)
    test_data = DatasetPure.build_testset(test_df)
    
    # HERE I AM STORING THE FINAL DATA INFO AND USER/ITEM MAPPINGS.
    all_items = data_info.item_unique_vals
    all_users = data_info.user_unique_vals
    item2pos = {it: i for i, it in enumerate(all_items)}
    id2user = data_info.id2user
    id2item = data_info.id2item
    user2hist_orig = {}
    for inner_u, inner_item_list in data_info.user_consumed.items():
        orig_u = id2user[inner_u]
        user2hist_orig[orig_u] = set(id2item[i] for i in inner_item_list)

    return (inter_df, train_df, eval_df, test_df, test_in_df,
            train_data, eval_data, test_data, data_info,
            all_items, all_users, item2pos, user2hist_orig, games_df)

def create_user_content_profiles(inter_df, all_users, item2pos, item_content_tfidf):
    """
    HERE I AM BUILDING USER CONTENT PROFILES BASED ON THEIR PLAYTIME-WEIGHTED INTERACTIONS.
    """
    print("\n--- STEP 6: Building user content profiles ---")
    
    # HERE I AM USING THE PASSED INTERACTION DATAFRAME DIRECTLY.
    wdf = inter_df

    user_item_w = defaultdict(list)
    for _, r in tqdm(wdf.iterrows(), total=len(wdf), desc="Processing interactions"):
        # I HAVE REMOVED THE UNNECESSARY .astype(float) which was causing the error.
        u, it, w = int(r['user']), int(r['item']), float(np.clip(np.log1p(r['playtime']), 0.0, 10.0))
        pos = item2pos.get(it, None)
        if pos is not None:
            user_item_w[u].append((pos, w))

    user_prof, user_prof_norm, user_tagged_counts = {}, {}, {}
    for u in tqdm(all_users, desc="Building user profiles"):
        pairs = user_item_w.get(int(u), [])
        if not pairs:
            user_prof[int(u)] = None
            user_prof_norm[int(u)] = 0.0
            user_tagged_counts[int(u)] = 0
            continue
        idxs = np.array([p[0] for p in pairs], dtype=int)
        ws = np.array([p[1] for p in pairs], dtype=float)
        
        # ... (rest of the function is unchanged)
        sw = ws.sum()
        user_tagged_counts[int(u)] = int((item_content_tfidf[idxs].getnnz(axis=1) > 0).sum())
        if sw <= 0:
            user_prof[int(u)] = None
            user_prof_norm[int(u)] = 0.0
            continue
        # HERE I AM COMPUTING THE WEIGHTED SUM OF ITEM TF-IDF VECTORS.
        v = (item_content_tfidf[idxs].T.dot(ws) / (sw + 1e-12)).ravel()
        n = np.linalg.norm(v)
        if n > 0: v = v / n
        user_prof[int(u)] = v
        user_prof_norm[int(u)] = float(n)

    return user_prof, user_prof_norm, user_tagged_counts


def create_tfidf_matrix(games_df, all_items, plot_dir):
    """
    HERE I AM CREATING THE TF-IDF MATRIX FOR CONTENT-BASED RECOMMENDATIONS.
    """
    print("\n--- STEP 4: Content TF-IDF (tags+genres) ---")
    cg = {c.lower(): c for c in games_df.columns}
    g_item = cg.get('item_id') or cg.get('appid') or cg.get('app_id') or cg.get('id')
    g_tags = cg.get('tags') or cg.get('tag')
    g_genre = cg.get('genres') or cg.get('genre')

    meta = games_df[[g_item] + ([g_tags] if g_tags else []) + ([g_genre] if g_genre else [])].copy()
    meta = meta.rename(columns={g_item: 'item', g_tags: 'tags', g_genre: 'genres'})
    meta['item'] = meta['item'].astype(int)
    # HERE I AM FILTERING THE METADATA TO ONLY INCLUDE ITEMS PRESENT IN OUR DATASET.
    meta = meta[meta['item'].isin(all_items)].copy()
    meta['toks_tags'] = meta['tags'].apply(parse_tags_or_genres) if 'tags' in meta.columns else [[]]
    meta['toks_genres'] = meta['genres'].apply(parse_tags_or_genres) if 'genres' in meta.columns else [[]]
    meta_map = meta.set_index('item')

    content_docs = []
    miss_tags = miss_gen = 0
    # HERE I AM BUILDING A "DOCUMENT" FOR EACH ITEM, COMPRISING ITS TAGS AND GENRES.
    for it in tqdm(all_items, desc="Building content docs"):
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

    print(f"Items missing tags: {miss_tags:,} / {len(all_items):,}; missing genres: {miss_gen:,} / {len(all_items):,}")

    vec = TfidfVectorizer(token_pattern=r"(?u)\b[\w:]+\b", lowercase=False, sublinear_tf=True, use_idf=True, norm='l2', min_df=2)
    item_content_tfidf = vec.fit_transform(content_docs).tocsr()
    print(f"TF-IDF shape: {item_content_tfidf.shape}")

    # HERE I AM PLOTTING A SCIENTIFIC CHECK FOR CONTENT COVERAGE.
    nz = item_content_tfidf.getnnz(axis=1)
    if hasattr(nz, 'A1'):
        nz = nz.A1
    plt.figure(figsize=(7, 4))
    plt.hist(nz, bins=50)
    plt.xlabel("#content tokens per item")
    plt.ylabel("Count")
    plt.title("Content Coverage per Item")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "content_coverage_hist.png"))
    plt.close()

    return item_content_tfidf