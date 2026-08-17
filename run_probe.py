#!/usr/bin/env python3
"""
Multi-country jurisdictional-conflation probe (ES / CL / CO / AR), OpenRouter.
Runs the 13 source-verified items in items.json against every model in DEFAULT_MODELS.
Analyse the output with analyze_conflation.py, which adds the distinct-values
subanalysis (cases where the four countries disagree).

Reads a curated item file where every item states one checkable value per country
for a shared unit, plus a hand-written Spanish question. For each item it asks,
per model:
  - unspec : country not named  -> which jurisdiction's value does it give?
  - <país> : one condition per country present -> naming country X, does the model
             give X's value (correct), some OTHER country's value (conflation), or
             neither (other)?

Scoring is deterministic (number extraction). The conflation matrix (to_* columns
and the printed matrix) counts ONLY genuine conflation verdicts, so value
collisions across countries do not inflate it.

Usage:
  python run_probe.py                       # all models in DEFAULT_MODELS
  python run_probe.py --items items.json --runs 3
  python run_probe.py --models qwen/qwen3.7-flash

Outputs (next to this script):
  results_raw.jsonl      one line per API call
  results_summary.csv    per (model, asked_country): correct / conflation / other
Nothing is sent anywhere except OpenRouter.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
API_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODELS = [
    "deepseek/deepseek-v4-flash-0731",
    "nvidia/nemotron-3.5-lightning:free",
    "qwen/qwen3.7-flash",
    "google/gemini-3.7-flash",
    "openai/gpt-5.6-luna-pro",
    "z-ai/glm-5.2",
    "anthropic/claude-sonnet-5"
]

# adjective used in the stem "Segun el Codigo Civil <adj>"
COUNTRY_ADJ = {"ES": "espanol", "CL": "chileno", "CO": "colombiano", "AR": "argentino"}
COUNTRY_NAME = {"ES": "Espana", "CL": "Chile", "CO": "Colombia", "AR": "Argentina"}
UNIT_ES = {"years": "anos", "months": "meses", "days": "dias"}

# ---- deterministic extraction (shared with build_items / run_probe) ----
WORDS = {"un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
         "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11,
         "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16,
         "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
         "veinticinco": 25, "treinta": 30, "cuarenta": 40, "cincuenta": 50,
         "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100}
UNIT_RE = {"years": "anos?", "months": "meses|mes", "days": "dias?"}


def norm(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def extract(text, unit):
    t = norm(text)
    u = UNIT_RE[unit]
    found = set()
    for m in re.finditer(rf"\b(\d{{1,3}})\)?\s*(?:{u})\b", t):
        found.add(int(m.group(1)))
    wp = "|".join(sorted(WORDS, key=len, reverse=True))
    for m in re.finditer(rf"\b({wp})\s+(?:\(\d{{1,3}}\)\s*)?(?:{u})\b", t):
        found.add(WORDS[m.group(1)])
    return found


def classify(text, item, asked):
    """asked = country code or None (unspec). Returns verdict + which countries matched."""
    vals = extract(text, item["unit"])
    matched = [c for c, v in item["values"].items() if v in vals]
    t = norm(text)
    if asked is None:
        if not matched:
            asks = bool(re.search(r"de que pais|que jurisdiccion|depende del pais|no especific|precisa el pais", t))
            verdict = "aclara" if asks else "other"
        else:
            verdict = "matched"
        return {"verdict": verdict, "matched": matched, "vals": sorted(vals)}
    # named-country condition
    if asked in matched:
        verdict = "correct"                      # includes collisions where X shares a value
    elif matched:
        verdict = "conflation"                   # gave another country's value, not X's
    else:
        verdict = "other"
    others = [c for c in matched if c != asked]
    return {"verdict": verdict, "matched": matched, "conflated_to": others, "vals": sorted(vals)}


# ---- OpenRouter call --------------------------------------------------------
def load_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        envf = HERE / ".env"
        if envf.exists():
            for line in envf.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("OPENROUTER_API_KEY"):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("No OPENROUTER_API_KEY found (set the env var or put it in code/.env).")
    return key


def ask(key, model, prompt, max_tokens=2000, max_retries=4):
    body = json.dumps({"model": model, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/nllp2026", "X-Title": "conflation-probe"})
    last_err = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            text = msg.get("content") or ""
            if not text.strip():
                text = (msg.get("reasoning")
                        or (msg.get("reasoning_details") or [{}])[0].get("text") or "")
            return text, None, {"finish_reason": choice.get("finish_reason")}
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
            if e.code in (429, 500, 502, 503, 529):
                time.sleep(2 ** attempt); continue
            break
        except Exception as e:  # noqa: BLE001
            last_err = str(e); time.sleep(2 ** attempt)
    return None, last_err, {}


# ---- runner -----------------------------------------------------------------
def build_prompt(item, cond):
    unit_es = UNIT_ES[item["unit"]]
    q = item["q"].replace("{unit}", unit_es)
    if cond == "unspec":
        stem = "Segun el Codigo Civil"
    else:
        stem = f"Segun el Codigo Civil {COUNTRY_ADJ[cond.upper()]}"
    return f"{stem}, {q} Responde en una sola frase."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="items.json")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2000)
    args = ap.parse_args()

    items_path = HERE / args.items
    if not items_path.exists():
        sys.exit(f"Item file not found: {items_path}")
    items = json.loads(items_path.read_text(encoding="utf-8"))
    key = load_key()

    # conditions per item = unspec + every country that has a value for that item
    jobs = []
    for it in items:
        conds = ["unspec"] + [c for c in ["ES", "CL", "CO", "AR"] if c in it["values"]]
        for cond in conds:
            for r in range(args.runs):
                jobs.append((it, cond, r))
    print(f"{len(jobs) * len(args.models)} calls: {len(items)} items x conditions x "
          f"{args.runs} run(s) across {len(args.models)} model(s)")

    raw_path = HERE / "results_raw.jsonl"
    rows = []
    done = 0
    with raw_path.open("w", encoding="utf-8") as raw, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for m in args.models:
            for (it, cond, r) in jobs:
                prompt = build_prompt(it, cond)
                futs[pool.submit(ask, key, m, prompt, args.max_tokens)] = (m, it, cond, r, prompt)
        for fut in as_completed(futs):
            m, it, cond, r, prompt = futs[fut]
            text, err, meta = fut.result()
            asked = None if cond == "unspec" else cond
            rec = {"model": m, "item": it["id"], "condition": cond, "run": r,
                   "unit": it["unit"], "values": it["values"], "prompt": prompt,
                   "finish_reason": meta.get("finish_reason")}
            if err:
                rec.update({"verdict": "err", "error": err, "text": ""})
            elif not (text or "").strip():
                rec.update({"verdict": "blank", "text": ""})
            else:
                rec.update({"text": text, **classify(text, it, asked)})
            raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)
            done += 1
            if done % 20 == 0 or done == len(futs):
                print(f"  {done}/{len(futs)}")

    # ---- summary: per (model, asked country); to_* counts ONLY conflation rows ----
    summary_path = HERE / "results_summary.csv"
    countries = ["ES", "CL", "CO", "AR"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "asked", "n", "answered", "correct", "conflation",
                    "conflation_rate", "other", "blank", "err",
                    *[f"to_{c}" for c in countries]])
        for m in args.models:
            print(f"\n=== {m} ===")
            us = [x for x in rows if x["model"] == m and x["condition"] == "unspec"]
            lean = defaultdict(int)
            for x in us:
                for c in x.get("matched", []):
                    lean[c] += 1
            nomatch = sum(1 for x in us if not x.get("matched"))
            print("  unspec default lean: "
                  + ", ".join(f"{c}:{lean[c]}" for c in countries) + f"   none:{nomatch}")
            for asked in countries:
                rs = [x for x in rows if x["model"] == m and x["condition"] == asked]
                if not rs:
                    continue
                n = len(rs)
                blank = sum(1 for x in rs if x["verdict"] == "blank")
                err = sum(1 for x in rs if x["verdict"] == "err")
                answered = n - blank - err
                correct = sum(1 for x in rs if x["verdict"] == "correct")
                conf = sum(1 for x in rs if x["verdict"] == "conflation")
                other = sum(1 for x in rs if x["verdict"] == "other")
                # honest conflation matrix: destinations counted only on conflation rows
                to = defaultdict(int)
                for x in rs:
                    if x["verdict"] == "conflation":
                        for c in x.get("conflated_to", []):
                            if c != asked:
                                to[c] += 1
                rate = round(conf / answered, 3) if answered else ""
                w.writerow([m, asked, n, answered, correct, conf, rate, other, blank, err,
                            *[to[c] for c in countries]])
                tostr = " ".join(f"{c}:{to[c]}" for c in countries if to[c])
                print(f"  asked {asked}: correct {correct}/{answered}  conflation {conf} "
                      f"(rate {rate}) -> [{tostr}]  other {other}")

    print(f"\nWrote {raw_path.name} and {summary_path.name}")


if __name__ == "__main__":
    main()
