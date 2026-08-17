#!/usr/bin/env python3
"""
Multi-country jurisdictional-conflation item miner (ES / CL / CO / AR).

Extends the ES-CL pilot (build_items.py) to four civil codes:
  ES  legalize-es/es/BOE-A-1889-4763.md  Codigo Civil 1889
  CL  legalize-cl/cl/CL-172986.md        Codigo Civil (texto refundido)
  CO  legalize-co/co/LEY-84-1873.md       Codigo Civil de los Estados Unidos de Colombia
  AR  legalize-ar/ar/LEY-26994.md         Codigo Civil y Comercial de la Nacion (2015)

Pipeline
  1. parse each code into articles (handles the 4 heading formats).
  2. keep articles that state a checkable period (a number + years/months/days).
  3. align every non-pivot country's period-articles to the pivot (ES) by TF-IDF
     cosine similarity -> a star-shaped cluster per pivot article.
  4. keep clusters where >=2 countries state a value for a shared unit AND at
     least two of those values DIFFER -> discriminative items.

Outputs (next to this script)
  items_candidates.json    full detail per candidate item (all countries' text+values)
  items_candidates.csv    compact table for eyeballing before running any model

Heading formats handled
  ES  '###### Articulo 1.'                       heading only
  CL  '##### Articulo 1'                          heading only (numbering restarts)
  CO  '##### Articulo 1820. La sociedad ...'      text inline on heading line
  AR  '##### ARTICULO 2.- Interpretacion. ...'    uppercase, ordinal, em-dash, inline
"""
import csv
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root (has legalize-*)
CODES = {
    "ES": ROOT / "legalize-es/es/BOE-A-1889-4763.md",
    "CL": ROOT / "legalize-cl/cl/CL-172986.md",
    "CO": ROOT / "legalize-co/co/LEY-84-1873.md",
    "AR": ROOT / "legalize-ar/ar/LEY-26994.md",
}
PIVOT = "ES"
OTHERS = [c for c in CODES if c != PIVOT]

# ---------- parsing ----------------------------------------------------------
# One regex for all four codes: optional accent, any case, optional bis/ter,
# optional ordinal mark, then trailing punctuation, then any inline body text.
# Tolerates markdown emphasis (**bold**, *italic*) around "Articulo" and the
# number, plus ordinal marks (CO "1.º"), em-dashes (AR "2°.-") and periods.
HEAD = re.compile(
    r"^#{1,6}\s*\**\s*art[ií]culo\s+(\d{1,4})\s*"
    r"(bis|ter|qu[aá]ter)?\s*"
    r"[.\-–—°º)\*]*\s*(.*)$",
    re.I,
)
CTX = re.compile(r"^#{1,6}\s*\**\s*(LIBRO|T[ÍI]TULO|CAP[ÍI]TULO|Secci[óo]n|P[áa]rrafo)\b(.*)$", re.I)


def parse(path, jur):
    arts, cur, suf, buf, ctx = [], None, "", [], []
    for line in open(path, encoding="utf-8"):
        s = line.rstrip("\n").strip()
        c = CTX.match(s)
        if c and not HEAD.match(s):
            level = len(s) - len(s.lstrip("#"))
            ctx = [x for x in ctx if x[0] < level] + [(level, c.group(0).lstrip("# ").strip())]
            continue
        m = HEAD.match(s)
        if m:
            if cur is not None:
                arts.append(mk(jur, cur, suf, buf, ctx))
            cur, suf = m.group(1), (m.group(2) or "")
            inline = m.group(3).strip()
            buf = [inline] if inline else []
            continue
        if cur is not None:
            buf.append(s)
    if cur is not None:
        arts.append(mk(jur, cur, suf, buf, ctx))
    return [a for a in arts if len(a["text"]) > 40]


def mk(jur, num, suf, buf, ctx):
    txt = " ".join(x for x in buf if x)
    txt = re.sub(r"^Art\.?\s*[0-9]+\s*[º°]?\.?\s*", "", txt)   # stray inline "Art. 102." prefix
    txt = re.sub(r"\s+", " ", txt).strip()
    art = num + (f" {suf}" if suf else "")
    return {"jur": jur, "art": art, "text": txt[:1400],
            "context": " > ".join(t for _, t in ctx[-2:])}


# ---------- numeric value extraction -----------------------------------------
WORDS = {
    "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11,
    "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16,
    "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
    "veinticinco": 25, "treinta": 30, "cuarenta": 40, "cincuenta": 50,
    "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100,
    "ciento": 100, "doscientos": 200, "trescientos": 300, "cuatrocientos": 400,
    "quinientos": 500,
}
UNITS = {"anos": "years", "ano": "years", "meses": "months", "mes": "months",
         "dias": "days", "dia": "days"}
UNIT_RE = r"anos?|meses|mes|dias?"


def strip_acc(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def values(text):
    """Set of (number, unit) periods stated in the article. Tolerates the
    AR parenthetical gloss 'cinco (5) anos'."""
    t = strip_acc(text)
    out = set()
    for m in re.finditer(rf"\b(\d{{1,3}})\)?\s*({UNIT_RE})\b", t):
        out.add((int(m.group(1)), UNITS[m.group(2)]))
    wp = "|".join(sorted(WORDS, key=len, reverse=True))
    for m in re.finditer(rf"\b({wp})\s+(?:\(\d{{1,3}}\)\s*)?({UNIT_RE})\b", t):
        out.add((WORDS[m.group(1)], UNITS[m.group(2)]))
    return out


# ---------- alignment (star: pivot -> each other country) --------------------
STOP = set(strip_acc(w) for w in """de la el los las en y a que del se por un una con no su al lo
como mas o pero sus le ha si sin sobre este ya entre cuando todo esta ser son dos otro he
cual sea poco ella estar haber estas estaba estamos algunas algo nosotros mi mis tu te ti
tanto aquel cada cual quien cuyo dicho misma mismo tal segun caso casos efecto virtud""".split())


def toks(s):
    return [w for w in re.findall(r"[a-z]+", strip_acc(s)) if len(w) > 3 and w not in STOP]


def build_index(arts):
    df = defaultdict(int)
    for a in arts:
        for w in set(toks(a["text"])):
            df[w] += 1
    N = max(len(arts), 1)
    idf = {w: math.log(N / (1 + c)) for w, c in df.items()}

    def vec(a):
        v = defaultdict(float)
        for w in toks(a["text"]):
            v[w] += idf.get(w, 0)
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / n for w, x in v.items()}

    vecs = [(a, vec(a)) for a in arts]
    inv = defaultdict(list)
    for i, (_, v) in enumerate(vecs):
        for w in v:
            inv[w].append(i)
    return vecs, inv, vec


def best_match(a, vecs, inv, vec, min_overlap, min_shared):
    va = vec(a)
    cand = defaultdict(float)
    for w, x in va.items():
        for i in inv.get(w, []):
            cand[i] += x * vecs[i][1].get(w, 0)
    if not cand:
        return None
    i, sc = max(cand.items(), key=lambda kv: kv[1])
    b = vecs[i][0]
    shared = len(set(toks(a["text"])) & set(toks(b["text"])))
    if sc >= min_overlap and shared >= min_shared:
        return round(sc, 3), b
    return None


# ---------- item assembly ----------------------------------------------------
def units_of(vals):
    return {u for _, u in vals}


def nums_for_unit(vals, unit):
    return frozenset(n for n, u in vals if u == unit)


def main(min_overlap=0.14, min_shared=3):
    parsed = {}
    for jur, path in CODES.items():
        arts = parse(path, jur)
        parsed[jur] = arts
        n_val = sum(1 for a in arts if values(a["text"]))
        print(f"{jur}: parsed {len(arts):5d} articles, {n_val:4d} state a period")

    # index each non-pivot country's period-articles
    idx = {}
    for jur in OTHERS:
        arts = [a for a in parsed[jur] if values(a["text"])]
        idx[jur] = (arts, *build_index(arts))

    pivot_arts = [a for a in parsed[PIVOT] if values(a["text"])]
    items = []
    for pa in pivot_arts:
        cluster = {PIVOT: {"art": pa["art"], "ctx": pa["context"],
                           "text": pa["text"], "vals": sorted(values(pa["text"])),
                           "score": 1.0}}
        for jur in OTHERS:
            arts, vecs, inv, vec = idx[jur]
            hit = best_match(pa, vecs, inv, vec, min_overlap, min_shared)
            if hit:
                sc, b = hit
                cluster[jur] = {"art": b["art"], "ctx": b["context"],
                                "text": b["text"], "vals": sorted(values(b["text"])),
                                "score": sc}
        if len(cluster) < 2:
            continue
        # discriminative on a shared unit?
        shared_units = set.intersection(*[units_of(set(map(tuple, c["vals"]))) for c in cluster.values()]) \
            if len(cluster) > 1 else set()
        disc_units = [u for u in shared_units
                      if len({nums_for_unit(set(map(tuple, c["vals"])), u) for c in cluster.values()}) > 1]
        if not disc_units:
            continue
        items.append({
            "n_countries": len(cluster),
            "disc_units": sorted(disc_units),
            "pivot_score": cluster[PIVOT]["score"],
            "align_score": round(min(c["score"] for k, c in cluster.items() if k != PIVOT), 3),
            "countries": cluster,
        })

    # rank: more countries first, then alignment quality
    items.sort(key=lambda it: (-it["n_countries"], -it["align_score"]))
    json.dump(items, open(Path(__file__).parent / "items_candidates.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # compact review CSV
    rev = Path(__file__).parent / "items_candidates.csv"
    with rev.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "n_countries", "disc_units", "align_score",
                    "ES_art", "ES_vals", "CL_art", "CL_vals",
                    "CO_art", "CO_vals", "AR_art", "AR_vals", "ES_context"])
        for k, it in enumerate(items):
            c = it["countries"]

            def cell(j):
                if j not in c:
                    return "", ""
                vs = ",".join(f"{n}{u[0]}" for n, u in c[j]["vals"])
                return c[j]["art"], vs
            row = [k, it["n_countries"], "/".join(it["disc_units"]), it["align_score"]]
            for j in ["ES", "CL", "CO", "AR"]:
                row += list(cell(j))
            row.append(c["ES"]["ctx"])
            w.writerow(row)

    print(f"\nDISCRIMINATIVE candidate items: {len(items)}")
    by_n = defaultdict(int)
    for it in items:
        by_n[it["n_countries"]] += 1
    for n in sorted(by_n, reverse=True):
        print(f"  {n}-country items: {by_n[n]}")
    print(f"Wrote items_candidates.json and items_candidates.csv")


if __name__ == "__main__":
    main()
