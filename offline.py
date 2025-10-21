# tests_offline.py
from pathlib import Path
import numpy as np, pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier

import phish_guard_onefile as pg  # your onefile

DATA = Path("data/raw")
DROP = {"defacement","malware"}

def first_host(urls):
    if not urls: return ""
    try: return (urlparse(urls[0]).hostname or "").lower()
    except: return ""

def feats_from_df(df, mode="both"):
    # mode: "both" | "url_only" | "text_only"
    feats = []
    for r in df.itertuples(index=False):
        raw = r.raw_email if mode in ("both","text_only") else ""
        urls = r.urls if mode in ("both","url_only") else []
        feats.append(pg.extract_basic_features(raw_email=raw, urls=urls))
    return feats

def train_eval(X_tr, y_tr, X_te, y_te):
    pos = max(int((y_tr==1).sum()),1); neg = max(int((y_tr==0).sum()),1)
    clf = XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=4, scale_pos_weight=neg/pos
    )
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:,1]
    return roc_auc_score(y_te, proba)

def main():
    print("[ingest] loading...")
    # Use the same explicit column mappings that worked when running the onefile CLI
    # (these ensure the CSVs with different headers are parsed the same way)
    df = pg.ingest_folder(
        DATA,
        text_col_arg=None,
        url_col_arg="url",
        label_col_arg="type",
        drop_labels=DROP,
    )
    print(df.head(2))

    # ===== (A) 5-fold GroupKFold by host =====
    print("\n[A] GroupKFold by first host (5-fold)")
    feats = feats_from_df(df, "both")
    y = df["label"].values
    hosts = np.array([first_host(u) for u in df["urls"].tolist()])
    mask = hosts != ""  # need groups
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(feats)

    gkf = GroupKFold(n_splits=5)
    aucs = []
    for tr, te in gkf.split(X[mask], y[mask], groups=hosts[mask]):
        auc = train_eval(X[mask][tr], y[mask][tr], X[mask][te], y[mask][te])
        aucs.append(auc)
    print(f"  AUCs: {[round(a,4) for a in aucs]}  mean={np.mean(aucs):.4f}")

    # ===== (B) Random-label sanity check =====
    print("\n[B] Random-label sanity (expect ~0.50 AUC)")
    rng = np.random.default_rng(42)
    y_shuf = rng.permutation(y)
    tr = np.arange(len(y)) % 5 != 0
    te = ~tr
    auc = train_eval(X[tr], y_shuf[tr], X[te], y_shuf[te])
    print(f"  Shuffled-label AUC={auc:.4f}")

    # ===== (C) Ablation: URL-only vs Text-only =====
    print("\n[C] Ablation")
    for mode in ["url_only","text_only"]:
        feats_m = feats_from_df(df, mode)
        X_m = dv.transform(feats_m) if mode=="both" else DictVectorizer(sparse=False).fit_transform(feats_m)
        # simple 80/20 split by index for stability
        n = len(y); cut = int(0.8*n)
        auc = train_eval(X_m[:cut], y[:cut], X_m[cut:], y[cut:])
        print(f"  {mode:9s} AUC={auc:.4f}")

    # ===== (D) Cross-source train→test matrix =====
    if "source" in df.columns and df["source"].nunique() >= 2:
        print("\n[D] Cross-source matrix")
        sources = sorted(df["source"].unique().tolist())
        feats_all = feats_from_df(df, "both")
        for s_tr in sources:
            idx_tr = df["source"]==s_tr
            dv2 = DictVectorizer(sparse=False)
            X_tr = dv2.fit_transform([feats_all[i] for i in np.where(idx_tr)[0]])
            y_tr = y[idx_tr]
            for s_te in sources:
                if s_te == s_tr: continue
                idx_te = df["source"]==s_te
                X_te = dv2.transform([feats_all[i] for i in np.where(idx_te)[0]])
                y_te = y[idx_te]
                auc = train_eval(X_tr, y_tr, X_te, y_te)
                print(f"  train={s_tr:25s}  test={s_te:25s}  AUC={auc:.4f}")

if __name__ == "__main__":
    main()

# tests_offline.py (updated to mirror onefile pipeline: Dict + TF-IDF, URL toggles)
from pathlib import Path
import numpy as np, pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier

import phish_guard_onefile as pg  # your onefile

DATA = Path("data/raw")
DROP = {"defacement","malware"}

def first_host(urls):
  if not urls: return ""
  try: return (urlparse(urls[0]).hostname or "").lower()
  except: return ""

def dict_feats(df, *, mode="both"):
  """
  Build dict features using pg.extract_basic_features with proper URL/text toggles.
  mode: 'both' | 'url_only' | 'text_only'
  """
  feats = []
  for r in df.itertuples(index=False):
    if mode == "url_only":
      feats.append(pg.extract_basic_features(raw_email="", urls=r.urls, ignore_urls=False))
    elif mode == "text_only":
      feats.append(pg.extract_basic_features(raw_email=r.raw_email, urls=[], ignore_urls=True))
    else:
      feats.append(pg.extract_basic_features(raw_email=r.raw_email, urls=r.urls, ignore_urls=False))
  return feats

def text_feats(df, *, mode="both"):
    """
    Build TF-IDF inputs depending on mode.
    - url_only: return an empty sparse matrix (no TF-IDF) to avoid empty-vocab errors
    - text_only/both: use raw_email, but guard against empty/stopword-only corpora
    Returns (tfidf_vectorizer_or_None, X_text_sparse)
    """
    from scipy.sparse import csr_matrix

    if mode == "url_only":
        # No text signal by design in this ablation
        return None, csr_matrix((len(df), 0))

    texts = df["raw_email"].fillna("").astype(str).values
    if not any(t.strip() for t in texts):
        return None, csr_matrix((len(texts), 0))

    try:
        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words="english")
        X_text = tfidf.fit_transform(texts)
    except ValueError:
        # e.g., "empty vocabulary; perhaps the documents only contain stop words"
        tfidf = None
        X_text = csr_matrix((len(texts), 0))

    return tfidf, X_text

def build_Xy(df, *, mode="both"):
  feats = dict_feats(df, mode=mode)
  dv = DictVectorizer(sparse=False)
  X_dict = dv.fit_transform(feats)
  tfidf, X_text = text_feats(df, mode=mode)
  X = hstack([X_dict, X_text]).tocsr()
  y = df["label"].values
  return dv, tfidf, X, y

def train_eval(X_tr, y_tr, X_te, y_te):
  pos = max(int((y_tr==1).sum()),1); neg = max(int((y_tr==0).sum()),1)
  clf = XGBClassifier(
      objective="binary:logistic", eval_metric="auc",
      n_estimators=200, max_depth=4, learning_rate=0.1,
      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
      random_state=42, n_jobs=4, scale_pos_weight=neg/pos
  )
  clf.fit(X_tr, y_tr)
  proba = clf.predict_proba(X_te)[:,1]
  return roc_auc_score(y_te, proba)

def main():
  print("[ingest] loading...")
  df = pg.ingest_folder(DATA, text_col_arg=None, url_col_arg=None, label_col_arg=None, drop_labels=DROP)
  print(df.head(2))

  # ===== (A) 5-fold GroupKFold by host (using Dict+TFIDF) =====
  print("\n[A] GroupKFold by first host (5-fold)")
  hosts = np.array([first_host(u) for u in df["urls"].tolist()])
  mask = hosts != ""  # require a host to group by
  # Build features on the masked subset to avoid empty groups
  dv, tfidf, X_all, y_all = build_Xy(df[mask].reset_index(drop=True), mode="both")
  gkf = GroupKFold(n_splits=5)
  aucs = []
  for tr_idx, te_idx in gkf.split(X_all, y_all, groups=hosts[mask]):
    auc = train_eval(X_all[tr_idx], y_all[tr_idx], X_all[te_idx], y_all[te_idx])
    aucs.append(auc)
  print(f"  AUCs: {[round(a,4) for a in aucs]}  mean={np.mean(aucs):.4f}")

  # ===== (B) Random-label sanity check =====
  print("\n[B] Random-label sanity (expect ~0.50 AUC)")
  rng = np.random.default_rng(42)
  y_shuf = rng.permutation(y_all)
  n = len(y_all); tr = np.arange(n) % 5 != 0; te = ~tr
  auc = train_eval(X_all[tr], y_shuf[tr], X_all[te], y_shuf[te])
  print(f"  Shuffled-label AUC={auc:.4f}")

  # ===== (C) Ablation: URL-only vs Text-only (Dict+TFIDF correctly toggled) =====
  print("\n[C] Ablation")
  for mode in ["url_only","text_only"]:
    dv_m, tfidf_m, X_m, y_m = build_Xy(df, mode=mode)
    n = len(y_m); cut = int(0.8*n)
    auc = train_eval(X_m[:cut], y_m[:cut], X_m[cut:], y_m[cut:])
    print(f"  {mode:9s} AUC={auc:.4f}")

  # ===== (D) Cross-source train→test matrix (no leakage: fit DV/TFIDF on train only) =====
  if "source" in df.columns and df["source"].nunique() >= 2:
    print("\n[D] Cross-source matrix")
    sources = sorted(df["source"].unique().tolist())
    for s_tr in sources:
      idx_tr = (df["source"]==s_tr).values
      dv_tr = DictVectorizer(sparse=False)
      feats_tr = dict_feats(df[idx_tr], mode="both")
      X_tr_dict = dv_tr.fit_transform(feats_tr)
      tfidf_tr, X_tr_txt = text_feats(df[idx_tr], mode="both")
      X_tr = hstack([X_tr_dict, X_tr_txt]).tocsr()
      y_tr = df.loc[idx_tr, "label"].values

      for s_te in sources:
        if s_te == s_tr: continue
        idx_te = (df["source"]==s_te).values
        feats_te = dict_feats(df[idx_te], mode="both")
        X_te_dict = dv_tr.transform(feats_te)
        from scipy.sparse import csr_matrix
        if tfidf_tr is None:
            X_te_txt = csr_matrix((int(idx_te.sum()), 0))
        else:
            X_te_txt = tfidf_tr.transform(df.loc[idx_te, "raw_email"].fillna("").astype(str).values)
        X_te = hstack([X_te_dict, X_te_txt]).tocsr()
        y_te = df.loc[idx_te, "label"].values
        auc = train_eval(X_tr, y_tr, X_te, y_te)
        print(f"  train={s_tr:25s}  test={s_te:25s}  AUC={auc:.4f}")

if __name__ == "__main__":
  main()
# offline.py — evaluation runner mirroring phish_guard_onefile.py knobs
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier

import phish_guard_onefile as pg

# -----------------------------
# Helpers
# -----------------------------

def first_host(urls):
    if not urls:
        return ""
    try:
        return (urlparse(urls[0]).hostname or "").lower()
    except Exception:
        return ""

def registrable_domain(host: str) -> str:
    """Lightweight eTLD+1 heuristic (keep in sync with onefile)."""
    host = (host or "").lower().strip(".")
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    second_level_cc = {"co", "com", "gov", "ac", "net", "org"}
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in second_level_cc:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])

def first_domain(urls):
    h = first_host(urls)
    return registrable_domain(h) if h else ""

# Build dict features with URL/text toggles to mirror ablations

def dict_feats(df: pd.DataFrame, *, mode: str = "both"):
    feats = []
    for r in df.itertuples(index=False):
        if mode == "url_only":
            feats.append(pg.extract_basic_features(raw_email="", urls=r.urls, ignore_urls=False))
        elif mode == "text_only":
            feats.append(pg.extract_basic_features(raw_email=r.raw_email, urls=[], ignore_urls=True))
        else:  # both
            feats.append(pg.extract_basic_features(raw_email=r.raw_email, urls=r.urls, ignore_urls=False))
    return feats

# Text features (robust to empty/stopword-only corpora); uses char-ngrams like the onefile

def text_feats(df: pd.DataFrame, *, mode: str = "both"):
    if mode == "url_only":
        return None, csr_matrix((len(df), 0))
    texts = df["raw_email"].fillna("").astype(str).values
    if not any(t.strip() for t in texts):
        return None, csr_matrix((len(df), 0))
    try:
        tfidf = TfidfVectorizer(max_features=2000, analyzer="char_wb", ngram_range=(3,5))
        X_text = tfidf.fit_transform(texts)
    except ValueError:
        tfidf = None
        X_text = csr_matrix((len(df), 0))
    return tfidf, X_text

# Train/eval helper

def train_eval(X_tr, y_tr, X_te, y_te):
    pos = max(int((y_tr == 1).sum()), 1)
    neg = max(int((y_tr == 0).sum()), 1)
    clf = XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=4, scale_pos_weight=neg/pos
    )
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, proba)

# Build feature matrix

def build_Xy(df: pd.DataFrame, *, mode: str = "both"):
    feats = dict_feats(df, mode=mode)
    dv = DictVectorizer(sparse=False)
    X_dict = dv.fit_transform(feats)
    tfidf, X_text = text_feats(df, mode=mode)
    X = hstack([X_dict, X_text]).tocsr()
    y = df["label"].values
    return dv, tfidf, X, y

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Offline eval for phish_guard_onefile.py")
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--drop-labels", default="defacement,malware")
    ap.add_argument("--url-col", default="url")
    ap.add_argument("--label-col", default="type")
    ap.add_argument("--sources", default="", help="Comma-separated substrings of filenames to include (e.g., ceas,phishing)")
    ap.add_argument("--max-per-source", type=int, default=30000)
    ap.add_argument("--group-by", choices=["host","domain"], default="host")
    ap.add_argument("--ablation", choices=["both","url_only","text_only"], default="both")
    args = ap.parse_args()

    DATA = Path(args.data)
    DROP = {s.strip() for s in args.drop_labels.split(',') if s.strip()}
    include = {s.strip().lower() for s in args.sources.split(',') if s.strip()} or None

    print("[ingest] loading...")
    df = pg.ingest_folder(
        DATA,
        text_col_arg=None,
        url_col_arg=args.url_col,
        label_col_arg=args.label_col,
        drop_labels=DROP,
        include_sources=include,
        max_per_source=args.max_per_source,
    )
    print(df.head(2))

    # ===== (A) 5-fold GroupKFold by host/domain (Dict+TFIDF) =====
    print("\n[A] GroupKFold by", ("domain" if args.group_by=="domain" else "first host"), "(5-fold)")
    if args.group_by == "domain":
        groups_all = np.array([first_domain(u) for u in df["urls"].tolist()])
    else:
        groups_all = np.array([first_host(u) for u in df["urls"].tolist()])
    mask = groups_all != ""

    dv, tfidf, X_all, y_all = build_Xy(df[mask].reset_index(drop=True), mode=args.ablation)
    groups = groups_all[mask]
    gkf = GroupKFold(n_splits=5)
    aucs = []
    for tr_idx, te_idx in gkf.split(X_all, y_all, groups=groups):
        auc = train_eval(X_all[tr_idx], y_all[tr_idx], X_all[te_idx], y_all[te_idx])
        aucs.append(auc)
    print(f"  AUCs: {[round(a,4) for a in aucs]}  mean={np.mean(aucs):.4f}")

    # ===== (B) Random-label sanity =====
    print("\n[B] Random-label sanity (expect ~0.50 AUC)")
    rng = np.random.default_rng(42)
    y_shuf = rng.permutation(y_all)
    n = len(y_all)
    tr = np.arange(n) % 5 != 0
    te = ~tr
    auc = train_eval(X_all[tr], y_shuf[tr], X_all[te], y_shuf[te])
    print(f"  Shuffled-label AUC={auc:.4f}")

    # ===== (C) Ablation report (URL-only vs Text-only) =====
    print("\n[C] Ablation")
    for mode in ["url_only", "text_only"]:
        dv_m, tfidf_m, X_m, y_m = build_Xy(df, mode=mode)
        n = len(y_m)
        cut = int(0.8 * n)
        auc = train_eval(X_m[:cut], y_m[:cut], X_m[cut:], y_m[cut:])
        print(f"  {mode:9s} AUC={auc:.4f}")

    # ===== (D) Cross-source train→test matrix (fit DV/TFIDF on TRAIN only) =====
    if "source" in df.columns and df["source"].nunique() >= 2:
        print("\n[D] Cross-source matrix")
        sources = sorted(df["source"].unique().tolist())
        for s_tr in sources:
            idx_tr = (df["source"] == s_tr).values
            dv_tr = DictVectorizer(sparse=False)
            feats_tr = dict_feats(df[idx_tr], mode=args.ablation)
            X_tr_dict = dv_tr.fit_transform(feats_tr)
            tfidf_tr, X_tr_txt = text_feats(df[idx_tr], mode=args.ablation)
            X_tr = hstack([X_tr_dict, X_tr_txt]).tocsr()
            y_tr = df.loc[idx_tr, "label"].values

            for s_te in sources:
                if s_te == s_tr:
                    continue
                idx_te = (df["source"] == s_te).values
                feats_te = dict_feats(df[idx_te], mode=args.ablation)
                X_te_dict = dv_tr.transform(feats_te)
                if tfidf_tr is None:
                    X_te_txt = csr_matrix((int(idx_te.sum()), 0))
                else:
                    X_te_txt = tfidf_tr.transform(df.loc[idx_te, "raw_email"].fillna("").astype(str).values)
                X_te = hstack([X_te_dict, X_te_txt]).tocsr()
                y_te = df.loc[idx_te, "label"].values
                auc = train_eval(X_tr, y_tr, X_te, y_te)
                print(f"  train={s_tr:25s}  test={s_te:25s}  AUC={auc:.4f}")

if __name__ == "__main__":
    main()