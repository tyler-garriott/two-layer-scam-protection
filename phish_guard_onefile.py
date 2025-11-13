#!/usr/bin/env python3
"""
phish_guard_onefile.py

One-file runner to:
1) Ingest CSVs from a folder into a unified schema (raw_email, urls, label).
2) Train a quick Stage-1 baseline (Logistic Regression with class_weight=balanced).
3) Launch a FastAPI server exposing /scan-email and /explain-email using the trained model.

Example:
  python phish_guard_onefile.py --data data/raw \
      --out artifacts \
      --url-col url --label-col label --drop-labels "defacement,malware" \
      --port 8000

If your email dataset uses "Email Text" and "Label":
  python phish_guard_onefile.py --data data/raw \
      --out artifacts \
      --text-col "Email Text" --label-col Label \
      --port 8000
"""
import argparse, os, re, json, sys, time
from pathlib import Path
from typing import List, Dict, Tuple, Any
from urllib.parse import urlparse
from collections import Counter
import math

import pandas as pd
import numpy as np

# Model bits
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import joblib

# API bits
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# -----------------------------
# Ingestion
# -----------------------------

LABEL_CANDIDATES = ("label","Label","labels","class","Class","target","is_spam","phishing","Result")
TEXT_CANDIDATES  = ("email_text","text","body","raw_email","message","Email Text","Email","content")
URL_CANDIDATES   = ("url","URL","urls","link","Links")

URL_RE = re.compile(r'https?://\S+', re.IGNORECASE)

def autodetect_columns(df: pd.DataFrame):
    label = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    text  = next((c for c in TEXT_CANDIDATES  if c in df.columns), None)
    url   = next((c for c in URL_CANDIDATES   if c in df.columns), None)
    return text, url, label

def normalize_label(val):
    if pd.isna(val): return None
    # numeric
    if isinstance(val, (int, float, np.integer, np.floating)):
        return int(val >= 1)
    # string (robust substring checks)
    s = str(val).strip().lower()
    if any(k in s for k in ("phish","spam","fraud","malicious","bad","phishing")):
        return 1
    if any(k in s for k in ("ham","benign","legit","legitimate","good","normal","safe")):
        return 0
    # binary-like strings
    if s in ("1","true","yes"): return 1
    if s in ("0","false","no"): return 0
    # fallback: anything not clearly positive -> 0
    try:
        return int(float(s) >= 1.0)
    except:
        return 0

def extract_urls(text):
    if not isinstance(text, str): return []
    return URL_RE.findall(text)

def ingest_folder(folder: Path,
                  text_col_arg: str|None,
                  url_col_arg: str|None,
                  label_col_arg: str|None,
                  drop_labels: set[str],
                  include_sources: set[str] | None = None,
                  max_per_source: int = 30000) -> pd.DataFrame:
    files = sorted([p for p in folder.glob("**/*.csv")])
    rows = []
    for f in files:
        # Optional: include only selected sources (match by filename substring)
        if include_sources:
            low = f.name.lower()
            if not any(tok in low for tok in include_sources):
                continue
        try:
            df = pd.read_csv(f, encoding="utf-8", engine="python", on_bad_lines="skip")
        except Exception:
            try:
                df = pd.read_csv(f, encoding="latin-1", engine="python", on_bad_lines="skip")
            except Exception as e:
                print(f"[ingest] Skipping {f} due to read error: {e}", file=sys.stderr)
                continue

        # Heuristics: prefer explicit args, then per-file defaults, then autodetect
        fname = f.name.lower()
        text_col, url_col, label_col = autodetect_columns(df)

        # Per-file known mappings
        if "malicious_phish.csv" in fname:
            url_col = url_col_arg or "url"
            label_col = label_col_arg or "type"
            # ensure we drop extra classes
            drop_labels = set(drop_labels) | {"defacement", "malware"}
        elif "phishing_email.csv" in fname:
            # common variations in this dataset
            text_col = text_col_arg or ("Email Text" if "Email Text" in df.columns else "Email" if "Email" in df.columns else text_col)
            # prefer "Email Type" for label if present
            if "Email Type" in df.columns:
                label_col = label_col_arg or "Email Type"
            else:
                label_col = label_col_arg or ("Label" if "Label" in df.columns else label_col)
        elif "ceas_08.csv" in fname or "ceas" in fname:
            candidate_text = [c for c in ["body","text","message","Email Text","Email","content","subject"] if c in df.columns]
            candidate_label = [c for c in ["label","Label","Class","class","spam","Spam","is_spam","category"] if c in df.columns]
            text_col = text_col_arg or (candidate_text[0] if candidate_text else text_col)
            label_col = label_col_arg or (candidate_label[0] if candidate_label else label_col)

        # Final override with explicit args (only if that column exists in this file)
        if text_col_arg and text_col_arg in df.columns:
            text_col = text_col_arg
        if url_col_arg and url_col_arg in df.columns:
            url_col = url_col_arg
        if label_col_arg and label_col_arg in df.columns:
            label_col = label_col_arg

        # Normalize text column to a consistent 'raw_email' if present
        if "Email Text" in df.columns:
            df["raw_email"] = df["Email Text"].astype(str)
            text_col = "raw_email"
        elif "body" in df.columns:
            df["raw_email"] = df["body"].astype(str)
            text_col = "raw_email"

        # Robust column presence flags (avoid per-row `in` checks)
        has_text_col = bool(text_col and text_col in df.columns)
        has_url_col = bool(url_col and url_col in df.columns)
        has_label_col = bool(label_col and label_col in df.columns)
        if not has_label_col:
            # extra fallback for common label names if autodetect missed
            for alt in ("Email Type", "Label", "label", "Class", "class", "spam", "is_spam", "category"):
                if alt in df.columns:
                    label_col = alt
                    has_label_col = True
                    break
        if not has_text_col:
            # extra fallback for common text fields
            for alt in ("raw_email", "Email Text", "Email", "body", "text", "message", "content", "subject"):
                if alt in df.columns:
                    text_col = alt
                    has_text_col = True
                    break

        # Optionally print debug info about detected columns
        # print(f"[ingest][debug] {f.name}: text_col={text_col!r} present={has_text_col}, url_col={url_col!r} present={has_url_col}, label_col={label_col!r} present={has_label_col}")

        if not any([text_col, url_col, label_col]):
            print(f"[ingest] Warning: could not detect columns for {f}. Columns: {list(df.columns)}", file=sys.stderr)

        ingested_before = len(rows)
        dropped_missing_label = 0
        dropped_empty_both = 0
        for _, r in df.iterrows():
            raw_email = str(r[text_col]) if has_text_col else ""
            urls = []
            if has_url_col:
                val = r[url_col]
                if isinstance(val, str):
                    urls = [u.strip() for u in val.split() if isinstance(u, str) and u.strip().startswith("http")]
                elif isinstance(val, (list, tuple)):
                    urls = [u for u in val if isinstance(u, str) and u.strip().startswith("http")]
            if not urls:
                urls = extract_urls(raw_email)

            raw_label = r[label_col] if has_label_col else None
            if isinstance(raw_label, str) and raw_label.strip().lower() in drop_labels:
                continue
            label = normalize_label(raw_label) if raw_label is not None else None
            if label is None:
                dropped_missing_label += 1
                continue

            # Keep row if either text or URL exists
            if (raw_email and raw_email.strip()) or (urls and len(urls) > 0):
                rows.append({"raw_email": raw_email, "urls": urls, "label": int(label), "source": f.name})
            else:
                dropped_empty_both += 1

        kept = len(rows) - ingested_before
        print(f"[ingest] {f.name}: +{kept} rows (cols={list(df.columns)}) [text_col={text_col!r} present={has_text_col}, url_col={url_col!r} present={has_url_col}, label_col={label_col!r} present={has_label_col}, dropped_label={dropped_missing_label}, dropped_empty={dropped_empty_both}]")
    # Warn if known sources produced zero rows
    try:
        seen_files = {p.name for p in files}
        if "CEAS_08.csv".lower() in {n.lower() for n in seen_files} and not any("CEAS_08.csv" == r["source"] for r in rows):
            print("[ingest][warn] CEAS_08.csv parsed 0 rows — verify label/text column detection.", file=sys.stderr)
        if "Phishing_Email.csv".lower() in {n.lower() for n in seen_files} and not any("Phishing_Email.csv" == r["source"] for r in rows):
            print("[ingest][warn] Phishing_Email.csv parsed 0 rows — verify label/text column detection.", file=sys.stderr)
    except Exception:
        pass

    if not rows:
        raise SystemExit("No rows ingested. Check your column mappings and data folder.")
    df_out = pd.DataFrame(rows)
    # basic clean
    df_out = df_out[(df_out["raw_email"].fillna("") != "") | (df_out["urls"].map(len) > 0)]
    # build a robust dedupe key: prefer text (truncated), otherwise the first URL
    def _make_key(row):
        text = (row.get("raw_email") or "").strip()
        if text:
            return "T:" + text[:200]
        urls = row.get("urls") or []
        if urls:
            return "U:" + urls[0]
        return "EMPTY"
    df_out["dedupe_key"] = df_out.apply(_make_key, axis=1)
    df_out = df_out.drop_duplicates(subset=["dedupe_key"]).reset_index(drop=True)
    df_out = df_out.drop(columns=["dedupe_key"])
    # debug: show label counts
    try:
        print("[ingest] total rows after dedupe:", len(df_out))
        print("[ingest] label counts:", df_out["label"].value_counts(dropna=False).to_dict())
        try:
            print("[ingest] by-source counts:", df_out["source"].value_counts(dropna=False).to_dict())
        except Exception:
            pass
    except Exception:
        pass
    # Optional: percent non-empty text by source
    try:
        if "source" in df_out.columns:
            pct = (df_out.assign(has_text=(df_out["raw_email"].fillna("").str.len() > 0))
                         .groupby("source")["has_text"].mean().round(3).to_dict())
            print("[ingest] % non-empty raw_email by source:", pct)
    except Exception:
        pass
    # Optional: cap rows per source to reduce dataset imbalance
    try:
        if "source" in df_out.columns and max_per_source and max_per_source > 0:
            df_out = (df_out
                      .groupby("source", group_keys=False)
                      .apply(lambda g: g.sample(min(len(g), max_per_source), random_state=42))
                      .reset_index(drop=True))
            print("[ingest] capped per-source rows at", max_per_source, "-> total:", len(df_out))
            try:
                print("[ingest] by-source after cap:", df_out["source"].value_counts(dropna=False).to_dict())
            except Exception:
                pass
    except Exception as e:
        print("[ingest] per-source cap skipped due to:", e, file=sys.stderr)
    return df_out

# -----------------------------
# Features
# -----------------------------
URGENCY = ("urgent","immediately","verify","reset","suspended","action required","confirm","click here")

def _hostname_entropy(host: str) -> float:
    if not host:
        return 0.0
    c = Counter(host)
    n = sum(c.values())
    return -sum((v/n) * math.log2(v/n) for v in c.values())

SAFE_DOMAINS = (
    "google.com", "calendar.google.com", "gmail.com", "microsoft.com", "outlook.com",
    "apple.com", "icloud.com", "github.com", "gitlab.com", "amazon.com",
    "cloudflare.com", "zoom.us", "slack.com", "notion.so", "dropbox.com",
)

LOGIN_WORDS = ("login", "signin", "verify", "reset", "update", "secure", "account")

def _host_endswith(host: str, domain: str) -> bool:
    host = host.lower().rstrip('.')
    domain = domain.lower().rstrip('.')
    return host == domain or host.endswith('.' + domain)

# Simple eTLD+1 heuristic for registrable domain
def _registrable_domain(host: str) -> str:
    """Very small heuristic for eTLD+1 without external deps."""
    host = (host or "").lower().strip(".")
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    # handle a few common 2-level country TLDs
    second_level_cc = {"co", "com", "gov", "ac", "net", "org"}
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in second_level_cc:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])

SUSPicious_EXT = {".zip", ".rar", ".7z", ".iso", ".apk"}

def extract_basic_features(*, raw_email: str = "", urls: List[str] | None = None, ignore_urls: bool = False) -> Dict[str, float | int]:
    text = raw_email or ""
    found_urls = [] if ignore_urls else (urls or URL_RE.findall(text))
    lower = text.lower()

    feats: Dict[str, float | int] = {
        # text cues
        "len_text": len(text),
        "num_urls": len(found_urls),
        "num_exclam": text.count("!"),
        "num_upper_tokens": sum(1 for t in re.findall(r"[A-Z]{2,}", text)),
        "urgency_hits": sum(1 for u in URGENCY if u in lower),
        "has_bank": int("bank" in lower or "account" in lower),
        "has_password": int("password" in lower or "credential" in lower),
        # url aggregate placeholders (filled below if any URL)
        "avg_url_len": 0.0,
        "has_punycode": 0,
        "avg_host_len": 0.0,
        "avg_path_len": 0.0,
        "avg_query_len": 0.0,
        "avg_path_depth": 0.0,
        "avg_num_digits": 0.0,
        "avg_num_hyphens": 0.0,
        "avg_num_dots": 0.0,
        "avg_num_at": 0.0,
        "avg_num_params": 0.0,
        "frac_https": 0.0,
        "frac_ip_host": 0.0,
        "avg_host_entropy": 0.0,
        "frac_suspicious_ext": 0.0,
    }

    if not found_urls:
        return feats

    # per-URL lexical metrics
    url_lens = []
    host_lens = []
    path_lens = []
    query_lens = []
    path_depths = []
    num_digits = []
    num_hyphens = []
    num_dots = []
    num_at = []
    num_params = []
    https_flags = []
    ip_flags = []
    puny_flags = []
    host_entropies = []
    susp_ext_flags = []
    safe_hits = []
    login_hits = []

    for u in found_urls:
        # sanitize common trailing punctuation that may break parsing
        u_clean = (u or "").strip().strip("'\"<>[](){}.,; ")
        url_lens.append(len(u_clean))
        try:
            p = urlparse(u_clean)
            host = p.hostname or ""
            path = p.path or ""
            query = p.query or ""
            scheme = (p.scheme or "").lower()
        except Exception:
            # Malformed URL (e.g., invalid IPv6). Treat as empty components.
            host = ""
            path = ""
            query = ""
            scheme = ""

        host_lens.append(len(host))
        path_lens.append(len(path))
        query_lens.append(len(query))

        path_depths.append(path.count("/"))
        num_digits.append(sum(ch.isdigit() for ch in u_clean))
        num_hyphens.append(u_clean.count("-"))
        num_dots.append(u_clean.count("."))
        num_at.append(u_clean.count("@"))
        num_params.append(u_clean.count("&") + u_clean.count("=") + int("?" in u_clean))

        https_flags.append(int(scheme == "https"))
        # IPv4 or IPv6 host check (strip brackets for IPv6)
        _h = host.strip("[]")
        is_ipv4 = bool(re.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", _h))
        is_ipv6 = ":" in _h and all(part for part in _h.split(":"))
        ip_flags.append(int(is_ipv4 or is_ipv6))
        puny_flags.append(int("xn--" in host))
        host_entropies.append(_hostname_entropy(host.lower()))

        # domain allowlist (major providers)
        safe_hit = int(any(_host_endswith(host, d) for d in SAFE_DOMAINS))
        safe_hits.append(safe_hit)

        # risk cue: presence of login/verify words in host or path
        lp = (host + "/" + (path or "")).lower()
        login_hits.append(int(any(w in lp for w in LOGIN_WORDS)))

        # suspicious extension check on last path segment
        last_seg = path.rsplit("/", 1)[-1] if path else ""
        susp_ext_flags.append(int(any(last_seg.lower().endswith(ext) for ext in SUSPicious_EXT)))

    n = len(found_urls)
    def avg(arr): return float(sum(arr) / n) if n else 0.0
    def frac(arr): return float(sum(arr) / n) if n else 0.0

    feats.update({
        "avg_url_len": avg(url_lens),
        "has_punycode": int(any(puny_flags)),
        "avg_host_len": avg(host_lens),
        "avg_path_len": avg(path_lens),
        "avg_query_len": avg(query_lens),
        "avg_path_depth": avg(path_depths),
        "avg_num_digits": avg(num_digits),
        "avg_num_hyphens": avg(num_hyphens),
        "avg_num_dots": avg(num_dots),
        "avg_num_at": avg(num_at),
        "avg_num_params": avg(num_params),
        "frac_https": frac(https_flags),
        "frac_ip_host": frac(ip_flags),
        "avg_host_entropy": avg(host_entropies),
        "frac_suspicious_ext": frac(susp_ext_flags),
        "frac_safe_domain": frac(safe_hits),
        "frac_login_words": frac(login_hits),
    })
    return feats

# -----------------------------
# Model (in-memory)
# -----------------------------
class Stage1Runtime:
    def __init__(self):
        self.dv: DictVectorizer | None = None
        self.clf: XGBClassifier | None = None
        self.order: List[str] | None = None
        self.trained: bool = False
        self.tfidf: TfidfVectorizer | None = None
        self.ignore_urls: bool = False
        self.group_by: str = "host"  # 'host' | 'domain' | 'random'
        self.balance_sources: bool = False

    def fit(self, df: pd.DataFrame, out_dir: Path | None = None, ignore_urls: bool = False, group_by: str = "host", balance_sources: bool = False, ablation: str = "both", do_loso: bool = True) -> float:
        self.ignore_urls = ignore_urls
        self.group_by = group_by
        self.balance_sources = balance_sources
        # featurize dict features
        feats = []
        for r in df.itertuples(index=False):
            raw_txt = r.raw_email if ablation in ("both", "text_only") else ""
            urls_in = ([] if ablation == "text_only" else r.urls)
            feats.append(extract_basic_features(raw_email=raw_txt, urls=urls_in, ignore_urls=(ignore_urls or ablation=="text_only")))
        y = df["label"].values

        # Dict features
        dv = DictVectorizer(sparse=False)
        X_dict = dv.fit_transform(feats)

        # Text TF-IDF features (lightweight) — robust to empty/stopword-only corpora
        texts = (df["raw_email"] if ablation in ("both","text_only") else pd.Series([""]*len(df))).fillna("").astype(str).values
        from scipy.sparse import hstack, csr_matrix
        if not any(t.strip() for t in texts):
            tfidf = None
            X_text = csr_matrix((len(texts), 0))
        else:
            try:
                tfidf = TfidfVectorizer(max_features=2000, analyzer="char_wb", ngram_range=(3,5))
                X_text = tfidf.fit_transform(texts)
            except ValueError:
                tfidf = None
                X_text = csr_matrix((len(texts), 0))
        X = hstack([X_dict, X_text]).tocsr()

        # Build groups by first URL host/domain/random
        def _first_host(urls):
            if not urls:
                return ""
            try:
                return (urlparse(urls[0]).hostname or "").lower()
            except Exception:
                return ""
        def _first_domain(urls):
            h = _first_host(urls)
            return _registrable_domain(h) if h else ""
        if group_by == "domain":
            groups = np.array([_first_domain(r.urls) for r in df.itertuples(index=False)])
        elif group_by == "host":
            groups = np.array([_first_host(r.urls) for r in df.itertuples(index=False)])
        else:
            groups = np.array([""] * len(df))

        # Robust split (prefer group-aware split by first host/domain to reduce leakage)
        unique, counts = np.unique(y, return_counts=True)
        can_stratify = (len(unique) == 2 and counts.min() >= 2 and len(y) >= 10)
        has_test = len(y) >= 5
        unique_groups = np.unique(groups)
        can_group_split = has_test and (len(unique_groups) >= 5) and (np.count_nonzero(groups != "") >= 20)
        if can_group_split:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            (tr_idx, te_idx), = gss.split(X, y, groups=groups)
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            print(f"[split] group-by-{group_by}: train_groups={len(np.unique(groups[tr_idx]))}, test_groups={len(np.unique(groups[te_idx]))}")
        elif has_test:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, stratify=(y if can_stratify else None), random_state=42
            )
            print("[split] fallback: random 80/20 split")
        else:
            X_tr, y_tr = X, y
            X_te, y_te = X[:0], y[:0]

        # Handle class imbalance
        # scale_pos_weight ~ (negatives / positives)
        pos = max(int((y_tr == 1).sum()), 1)
        neg = max(int((y_tr == 0).sum()), 1)
        spw = neg / pos

        sample_weight = None
        if balance_sources and "source" in df.columns:
            sources_all = df["source"].astype(str).values
            src_counts = pd.Series(sources_all).value_counts().to_dict()
            w_map = {s: (1.0 / c) for s, c in src_counts.items()}
            weights = np.array([w_map[s] for s in sources_all], dtype=float)
            sample_weight = weights

        clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=4,
            random_state=42,
            scale_pos_weight=spw,
        )
        if sample_weight is not None:
            clf.fit(X_tr, y_tr, sample_weight=sample_weight[tr_idx] if 'tr_idx' in locals() else sample_weight)
        else:
            clf.fit(X_tr, y_tr)

        if len(y_te) > 0:
            proba = clf.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, proba)
            print(f"[train] AUC={auc:.4f}")
            print(classification_report(y_te, (proba >= 0.5).astype(int)))
        else:
            auc = float("nan")
            print("[train] Note: too few samples for a test split; trained on all data.")

        # ===== Leave-One-Source-Out (LOSO) evaluation to detect overfitting/memorization =====
        if do_loso:
            try:
                sources = df["source"].astype(str).values if "source" in df.columns else None
            except Exception:
                sources = None
            if sources is not None and len(np.unique(sources)) >= 2:
                print("[eval] LOSO (leave-one-source-out) AUCs:")
                loso_aucs = []
                unique_sources = sorted(np.unique(sources).tolist())
                feats_all = feats  # list of dicts built above
                for src in unique_sources:
                    te_idx = np.where(sources == src)[0]
                    tr_idx = np.where(sources != src)[0]
                    if len(te_idx) < 10 or len(tr_idx) < 10:
                        print(f"  {src}: skipped (insufficient samples: n_te={len(te_idx)}, n_tr={len(tr_idx)})")
                        continue
                    # Dict features
                    dv_fold = DictVectorizer(sparse=False)
                    feats_tr = [feats_all[i] for i in tr_idx]
                    feats_te = [feats_all[i] for i in te_idx]
                    X_tr_dict = dv_fold.fit_transform(feats_tr)
                    X_te_dict = dv_fold.transform(feats_te)

                    # Text features, robust to empty/stopword-only
                    texts_tr = (df["raw_email"].iloc[tr_idx] if ablation in ("both","text_only") else pd.Series([""]*len(tr_idx), index=tr_idx)).fillna("").astype(str).values
                    texts_te = (df["raw_email"].iloc[te_idx] if ablation in ("both","text_only") else pd.Series([""]*len(te_idx), index=te_idx)).fillna("").astype(str).values
                    from scipy.sparse import hstack, csr_matrix
                    if not any(t.strip() for t in texts_tr):
                        tfidf_fold = None
                        X_tr_txt = csr_matrix((len(texts_tr), 0))
                        X_te_txt = csr_matrix((len(texts_te), 0))
                    else:
                        try:
                            tfidf_fold = TfidfVectorizer(max_features=2000, analyzer="char_wb", ngram_range=(3,5))
                            X_tr_txt = tfidf_fold.fit_transform(texts_tr)
                            X_te_txt = tfidf_fold.transform(texts_te)
                        except ValueError:
                            tfidf_fold = None
                            X_tr_txt = csr_matrix((len(texts_tr), 0))
                            X_te_txt = csr_matrix((len(texts_te), 0))
                    X_tr_fold = hstack([X_tr_dict, X_tr_txt]).tocsr()
                    X_te_fold = hstack([X_te_dict, X_te_txt]).tocsr()

                    y_tr_fold = y[tr_idx]
                    y_te_fold = y[te_idx]
                    # balance by observed skew in the training fold
                    pos_fold = max(int((y_tr_fold == 1).sum()), 1)
                    neg_fold = max(int((y_tr_fold == 0).sum()), 1)
                    spw_fold = neg_fold / pos_fold
                    clf_fold = XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="auc",
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        n_jobs=4,
                        random_state=42,
                        scale_pos_weight=spw_fold,
                    )
                    clf_fold.fit(X_tr_fold, y_tr_fold)
                    proba_fold = clf_fold.predict_proba(X_te_fold)[:, 1]
                    auc_fold = roc_auc_score(y_te_fold, proba_fold)
                    loso_aucs.append((src, auc_fold, len(te_idx)))
                    print(f"  {src}: AUC={auc_fold:.4f} (n_te={len(te_idx)})")
                if loso_aucs:
                    wavg_auc = np.average([a for (_, a, n) in loso_aucs], weights=[n for (_, a, n) in loso_aucs])
                    print(f"[eval] LOSO weighted AUC: {wavg_auc:.4f}")
        else:
            print("[eval] LOSO disabled (use --no-loso to enable speed mode)")

        # Persist in-memory
        self.dv = dv
        self.tfidf = tfidf
        self.order = dv.get_feature_names_out().tolist()
        self.clf = clf
        self.trained = True

        # Optional: save artifacts
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.dv, out_dir / "dv.pkl")
            joblib.dump(self.clf, out_dir / "xgb.pkl")
            joblib.dump(self.tfidf, out_dir / "tfidf.pkl")
        return float(auc)

    def predict(self, feats: Dict[str, Any], raw_email: str = "") -> Tuple[float, List[str]]:
        if not self.trained or self.dv is None or self.clf is None or self.order is None:
            return 0.50, ["model_untrained"]
        row_dict = self.dv.transform([feats])
        from scipy.sparse import hstack, csr_matrix
        if self.tfidf is not None:
            try:
                row_txt = self.tfidf.transform([raw_email or ""])
            except Exception:
                row_txt = csr_matrix((1, 0))
            row = hstack([row_dict, row_txt]).tocsr()
        else:
            row = row_dict
        score = float(self.clf.predict_proba(row)[:, 1][0])
        present = [(k, feats.get(k, 0.0)) for k in self.order]
        present = [k for k, v in present if v]
        return score, present[:3]

# -----------------------------
# API
# -----------------------------
class ScanEmailRequest(BaseModel):
    raw_email: str = ""
    urls: List[str] = []

class Stage1Result(BaseModel):
    score: float
    bucket: str
    top_factors: List[str] = []

class ExplainEmailRequest(BaseModel):
    raw_email: str = ""
    urls: List[str] = []
    stage1: Stage1Result

class ExplainEmailResponse(BaseModel):
    score: float
    verdict: str
    factors: List[str]
    explanation: str

def make_app(runtime: Stage1Runtime) -> FastAPI:
    app = FastAPI(title="Phish Guard — Onefile")
    # Enable CORS for development and extension access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For development; restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    ignore_urls = runtime.ignore_urls

    def bucketize(score: float) -> str:
        if score < 0.20: return "benign"       # very low threshold for green
        if score < 0.50: return "suspicious"   # gray band widened
        return "high-risk"                     # treat as phishing in tests

    @app.post("/scan-email")
    async def scan_email(req: ScanEmailRequest) -> Stage1Result:
        feats = extract_basic_features(raw_email=req.raw_email, urls=req.urls, ignore_urls=ignore_urls)
        score, factors = runtime.predict(feats, raw_email=req.raw_email)
        return Stage1Result(score=score, bucket=bucketize(score), top_factors=factors)

    @app.post("/scan-stage1")
    async def scan_stage1(req: ScanEmailRequest):
        feats = extract_basic_features(raw_email=req.raw_email, urls=req.urls, ignore_urls=ignore_urls)
        score, factors = runtime.predict(feats, raw_email=req.raw_email)

        # Guardrail for well-known providers: if URLs are mostly HTTPS on major safe domains,
        # with no obvious abuse cues (IP host or suspicious file extensions), treat as benign.
        # Note: allow "login" paths on safe domains to avoid false positives on real portals.
        safe_guard = (
            feats.get("frac_safe_domain", 0.0) >= 0.6 and
            feats.get("frac_https", 0.0) >= 0.9 and
            feats.get("frac_ip_host", 0.0) == 0 and
            feats.get("frac_suspicious_ext", 0.0) == 0
        )
        if safe_guard:
            # Strongly down-weight the score so the UI reflects low risk
            return {
                "verdict": "benign",
                "score": float(score) * 0.1,
                "factors": factors + ["safe_domain_guard"],
                "actions": []
            }

        verdict = ("benign" if score < 0.20 else "suspicious" if score < 0.50 else "phishing")
        return {"verdict": verdict, "score": float(score), "factors": factors, "actions": []}

    @app.get("/whoami")
    async def whoami():
        return {"app": "phish-guard-onefile", "guardrail": True}

    @app.post("/explain-email")
    async def explain_email(req: ExplainEmailRequest) -> ExplainEmailResponse:
        verdict = ("High risk" if req.stage1.score >= 0.70 else
                   "Suspicious" if req.stage1.score >= 0.40 else
                   "Benign")
        return ExplainEmailResponse(
            score=req.stage1.score,
            verdict=verdict,
            factors=req.stage1.top_factors,
            explanation="Template explanation (swap in LLM later)."
        )

    @app.get("/health")
    async def health():
        return {"ok": True, "trained": runtime.trained}

    @app.post("/debug-features")
    async def debug_features(req: ScanEmailRequest):
        feats = extract_basic_features(raw_email=req.raw_email, urls=req.urls, ignore_urls=ignore_urls)
        return {"features": feats}

    return app

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="One-file pipeline: ingest -> train -> serve")
    ap.add_argument("--data", required=True, help="Folder containing CSVs (recursively)")
    ap.add_argument("--out", default="artifacts", help="Where to save minimal artifacts")
    ap.add_argument("--text-col")
    ap.add_argument("--url-col")
    ap.add_argument("--label-col")
    ap.add_argument("--drop-labels", default="", help="Comma-separated labels to drop (e.g., defacement,malware)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-urls", action="store_true", help="Ignore URL features entirely (text-only stage-1)")
    ap.add_argument("--max-per-source", type=int, default=30000, help="Cap rows per source after dedupe (0 disables)")
    ap.add_argument("--sources", default="", help="Comma-separated substrings of filenames to include (e.g., ceas,phishing)")
    ap.add_argument("--group-by", choices=["host","domain","random"], default="host", help="How to group for the train/test split")
    ap.add_argument("--balance-sources", action="store_true", help="Reweight samples inversely to their source frequency")
    ap.add_argument("--use-only-malicious", action="store_true",
                    help="If set, only ingest files matching 'malicious_phish' (convenience for quick URL-only testing)")
    ap.add_argument("--ablation", choices=["both","url_only","text_only"], default="both", help="Train with both, URLs only, or text only")
    ap.add_argument("--no-loso", action="store_true", help="Disable leave-one-source-out evaluation for speed")
    args = ap.parse_args()

    drop_labels = set([s.strip().lower() for s in args.drop_labels.split(",") if s.strip()])
    folder = Path(args.data)
    if not folder.exists():
        raise SystemExit(f"--data folder not found: {folder}")

    print("[ingest] scanning CSVs...")
    # If requested, force using only the malicious_phish dataset for quick testing
    if args.use_only_malicious:
        include = {"malicious_phish"}
    else:
        include = set([s.strip().lower() for s in args.sources.split(",") if s.strip()]) or None
    df = ingest_folder(folder, args.text_col, args.url_col, args.label_col, drop_labels, include_sources=include, max_per_source=args.max_per_source)
    counts = df["label"].value_counts(dropna=False).to_dict()
    print(f"[ingest] rows: {len(df)}, label_counts: {counts}")

    print("[train] training baseline model...")
    runtime = Stage1Runtime()
    auc = runtime.fit(df, out_dir=Path(args.out), ignore_urls=args.no_urls, group_by=args.group_by, balance_sources=args.balance_sources, ablation=args.ablation, do_loso=not args.no_loso)
    print(f"[train] done. AUC={auc:.4f}")

    print(f"[serve] starting API on http://{args.host}:{args.port}  (endpoints: /scan-email, /scan-stage1, /explain-email, /health)")
    app = make_app(runtime)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()