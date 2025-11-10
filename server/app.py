# server/app.py
import json

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

OLLAMA = "http://localhost:11434"

app = FastAPI(title="Phish-Guard Shim")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScanReq(BaseModel):
    email_text: str
    subject: str = ""
    urls: list[str] = []
    headers: dict = {}


def call_llm(email_text: str, subject: str, urls: list[str], headers: dict):
    sys_msg = (
        "You output one JSON object with exactly these keys and types: "
        '{"score": number 0..1, "verdict": "benign"|"suspicious"|"phishing", '
        '"factors": [string], "actions": [string]}. No prose. Do not invent other keys.'
    )
    user_example = "Example:\nSUBJECT: Account locked\nBODY: Your Apple ID is locked. Verify at http://tiny.cc/login"
    asst_example = '{ "score": 0.93, "verdict": "phishing", "factors": ["shortened_url","credential_request"], "actions": ["report","do not click"] }'
    user_task = f"Analyze this email:\nSUBJECT: {subject}\nBODY: {email_text}\nURLS: {json.dumps(urls)}"

    # primary: chat with one-shot example
    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": "phish-guard",
            "format": "json",
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_example},
                {"role": "assistant", "content": asst_example},
                {"role": "user", "content": user_task},
            ],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=20,
    ).json()
    data = r.get("message", {}).get("content") or r.get("response") or "{}"
    if data.strip() != "{}":
        return json.loads(data)

    # fallback: seeded JSON completion
    seeded_prompt = (
        "Complete this JSON object only. No extra text:\n"
        '{\n  "score": 0.0,\n  "verdict": "",\n  "factors": [],\n  "actions": []\n}\n'
        f"EMAIL:\nSUBJECT: {subject}\nBODY: {email_text}\nURLS: {json.dumps(urls)}"
    )
    r2 = requests.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": "phish-guard",
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
            "system": sys_msg,
            "prompt": seeded_prompt,
        },
        timeout=20,
    ).json()
    return json.loads(r2.get("response", "{}"))


@app.post("/scan")
def scan(req: ScanReq):
    out = call_llm(req.email_text, req.subject, req.urls, req.headers)
    # minimal shape guard
    return {
        "verdict": out.get("verdict", "suspicious"),
        "score": float(out.get("score", 0.5)),
        "factors": out.get("factors", []),
        "actions": out.get("actions", []),
    }
