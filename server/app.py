# server/app.py
import json
import os, re

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from typing import List, Dict

OLLAMA = "http://localhost:11434"

KNOWN_DOMAINS = {
    "joinhandshake.com",
    "mail.joinhandshake.com",
    "google.com", "gmail.com",
    "microsoft.com", "outlook.com",
    "amazon.com", "apple.com",
    "utk.edu", "tennessee.edu", "listserv.utk.edu",
    "qualtrics.com", "utk.co1.qualtrics.com"
}

def _is_whitelisted_host(h: str) -> bool:
    if not h:
        return False
    h = h.lower()
    for d in KNOWN_DOMAINS:
        d = d.lower()
        if h == d or h.endswith("." + d) or h.endswith(d):
            return True
    return False

app = FastAPI(title="Phish-Guard Shim")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Lightweight email body sanitizer to prevent huge prompts/timeouts ---
def sanitize_text(text: str) -> str:
    if not text:
        return ""
    # Remove raw URLs; we already send URL list separately
    t = re.sub(r'https?://\S+', ' ', text)
    # Drop noisy repetitive lines like "Click to view" blocks (Canvas digests, etc.)
    t = re.sub(r'(?mi)^[ \t]*click to view.*$', '', t)
    # Collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    # Cap to a safe size for local models
    return t[:6000]

# --- Robust JSON extractor (handles code fences / extra prose) ---
def extract_json_block(s: str) -> str:
    if not s:
        return ""
    # Fast path: looks like pure JSON
    st = s.strip()
    if st.startswith("{") and st.endswith("}"):
        return st
    # Remove markdown code fences if present
    st = re.sub(r"^```(?:json)?\s*|```$", "", st.strip(), flags=re.IGNORECASE | re.MULTILINE)
    # Find first balanced {...} block
    depth = 0
    start = -1
    for i, ch in enumerate(st):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    return st[start:i+1]
    return ""

def call_llm(email_text: str, subject: str, urls: list[str], headers: dict, sender: str = ""):
    sys_msg = (
        "You are an email safety classifier. Return ONLY one JSON object with keys: "
        '{"verdict":"benign"|"suspicious"|"phishing","score":number,"factors":[string],"actions":[string]}. '
        "Scoring: benign 0.05–0.25, suspicious 0.35–0.65, phishing 0.75–0.95. "
        "Consider sender cues, credential requests, urgency, link targets, and impersonation. "
        "If evidence is weak, prefer suspicious over phishing."
    )
    
    # sanitize and cap to avoid oversized prompts/timeouts
    email_text = sanitize_text(email_text)
    subject = (subject or "")[:512]
    sender = (sender or "")[:256]
    sender_line = f"SENDER: {sender}\n" if sender else ""
    
    user_task = (
        sender_line +
        f"SUBJECT: {subject}\nBODY:\n{email_text}\nURLS: {json.dumps(urls)}\n"
        "Return one compact JSON line."
    )

    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": "phish-guard",
            "format": "json",
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_task},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "top_p": 0.9},
        },
        # timeout=45,
    )
    r.raise_for_status()
    payload = r.json()
    raw = payload.get("message", {}).get("content") or payload.get("response") or ""
    data = extract_json_block(raw)
    try:
        out = json.loads(data) if data else {}
    except Exception:
        out = {}

    verdict = (out.get("verdict") or "suspicious").lower()
    if verdict not in {"benign", "suspicious", "phishing"}:
        verdict = "suspicious"
    score = float(out.get("score", 0.5))
    factors = out.get("factors", [])
    actions = out.get("actions", [])

    if verdict == "phishing":
        score = max(min(score, 0.90), 0.75)
    elif verdict == "benign":
        score = max(min(score, 0.20), 0.05)
    else:
        score = max(min(score, 0.60), 0.35)

    return {"verdict": verdict, "score": score, "factors": factors, "actions": actions}


class ScanReq(BaseModel):
    subject: str = ""
    email_text: str = Field(default="", alias="body")
    urls: List[str] = []
    headers: Dict[str, str] = {}
    sender: str = ""

    class Config:
        populate_by_name = True


def _hostname(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).hostname or ""
    except Exception:
        return ""


@app.post("/scan")
def scan(req: ScanReq):
        # normalize request
    if req.email_text is None:
        req.email_text = ""
    if req.subject is None:
        req.subject = ""
    if getattr(req, "sender", None) is None:
        req.sender = ""
    req.sender = (req.sender or "")[:256]
    # keep at most first 15 URLs to avoid huge payloads
    req.urls = list(req.urls or [])[:15]

    text_all = (req.subject or "") + "\n" + (req.email_text or "")
    has_login_words = bool(re.search(r"\b(login|verify|password|reset|confirm|update\s+account|credential|sign\s*in)\b", text_all, re.I))
    mentions_survey = bool(re.search(r"\b(survey|focus\s*group|qualtrics|questionnaire)\b", text_all, re.I))

    hosts = [(_hostname(u) or "").lower() for u in req.urls]
    all_whitelisted = len(hosts) > 0 and all(_is_whitelisted_host(h) for h in hosts)

    # Sender-domain heuristic: if most links share the sender's domain and there are no login/credential words,
    # treat as benign (marketing/newsletters) rather than phishing.
    sender_domain = ""
    if getattr(req, "sender", None):
        m = re.search(r"@([A-Za-z0-9.-]+)$", req.sender or "")
        if m:
            sender_domain = m.group(1).lower()

    same_brand = False
    if sender_domain and hosts:
        match_count = sum(1 for h in hosts if h.endswith(sender_domain))
        same_brand = match_count >= max(1, len(hosts) // 2)

    # Fast-path: if there are no links and no credential language, skip LLM and return benign
    if len(req.urls) == 0 and not has_login_words:
        out = {"verdict": "benign", "score": 0.15, "factors": ["no links"], "actions": []}
    # If most links share the sender's domain and there are no login words, assume benign marketing/newsletter
    elif same_brand and not has_login_words:
        out = {"verdict": "benign", "score": 0.20, "factors": ["sender domain matches link domains"], "actions": []}
    else:
        out = call_llm(req.email_text, req.subject, req.urls, req.headers, req.sender)

    
    if out.get("verdict") == "phishing" and all_whitelisted and not has_login_words:
        # downgrade to suspicious if all links are whitelisted domains and no login words
        out["verdict"] = "suspicious"
        out["score"] = min(float(out.get("score", 0.8)), 0.55)
        out.setdefault("factors", []).append("all_whitelisted_links")
    elif all_whitelisted and not has_login_words:
        # boost benign if whitelisted and no login words
        out["verdict"] = "benign"
        out["score"] = max(float(out.get("score", 0.4)), 0.20)
        out.setdefault("factors", []).append("all_whitelisted_links")
    elif any((_hostname(u).endswith("utk.edu") or _hostname(u).endswith("tennessee.edu") or _hostname(u).endswith("qualtrics.com") or _hostname(u).endswith("utk.co1.qualtrics.com")) for u in req.urls) and mentions_survey and not has_login_words:
        if out.get("verdict") == "phishing":
            out["verdict"] = "suspicious"
            out["score"] = min(float(out.get("score", 0.8)), 0.55)
        else:
            out["verdict"] = "benign"
            out["score"] = min(float(out.get("score", 0.4)), 0.20)
        out.setdefault("factors", []).append("edu survey link")

    # Downgrade overly harsh verdicts when sender domain aligns with majority of links and no login words
    if out.get("verdict") == "phishing" and same_brand and not has_login_words:
        out["verdict"] = "suspicious"
        out["score"] = min(float(out.get("score", 0.8)), 0.55)
        out.setdefault("factors", []).append("sender-link domain alignment")

    # minimal shape guard
    return {
        "verdict": out.get("verdict", "suspicious"),
        "score": float(out.get("score", 0.5)),
        "factors": out.get("factors", []),
        "actions": out.get("actions", []),
    }
