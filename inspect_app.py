#!/usr/bin/env python3
"""
Local inspection app to manually verify the whole conflation pipeline, end to end.

It imports and calls the SAME functions the experiment uses, so what you see is
exactly what was computed --- not a reimplementation that could drift:
  - article  -> numbers   uses build_items.parse / .values
  - response -> numbers -> matched -> verdict   uses run_probe.extract / .classify

Views
  /            integrity dashboard (item-value checks + live-vs-stored verdict checks)
  /items       for every item x country: source article text, extracted numbers,
               recorded value, and whether they agree (catches wrong articles / misses)
  /responses   for every LLM call: prompt, full response, extracted numbers, matched
               countries, and verdict (stored vs recomputed). Filter by model/item/
               condition/verdict; toggle "only mismatches".

Stdlib only. Run:
  python inspect_app.py            # then open http://127.0.0.1:8000
  python inspect_app.py --port 8010 --no-open
Reads: items.json, results_raw.jsonl, and the four civil codes.
"""
import argparse
import html
import json
import re
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_items as bim   # noqa: E402
import run_probe as rob  # noqa: E402

COUNTRIES = ["ES", "CL", "CO", "AR"]

# ---- load data once ---------------------------------------------------------
ITEMS = json.loads((HERE / "items.json").read_text(encoding="utf-8"))
ITEM_BY_ID = {it["id"]: it for it in ITEMS}
ROWS = [json.loads(l) for l in (HERE / "results_raw.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]
MODELS = sorted({r["model"] for r in ROWS})

# Robust per-code article index keyed by the article NUMBER. Handles both heading
# forms: 'Articulo 537' (ES/CL/AR) and Colombia's inline '**art. 1374.**' label,
# which the miner's parser (keyed on markdown position) does not resolve by number.
# Article start: a '#'-heading OR a bold inline marker '**Articulo. N.**' (Colombia
# stacks several inline markers under one markdown heading).
_HEAD2 = re.compile(r"^(?:#{1,6}\s*\**|\*\*)\s*(?:art[ií]culo|art)\.?\s*(\d{1,4})\b", re.I)
# Context heading only when the keyword is followed by a numbering token (roman,
# arabic, or an ordinal word) -- so a mid-sentence '### titulo o buena fe, el plazo
# es de veinte anos' fragment (a source artifact) is NOT treated as a heading and its
# text stays in the article body.
_CTX2 = re.compile(
    r"^#{1,6}\s*\**\s*(?:LIBRO|T[ÍI]TULO|CAP[ÍI]TULO|SECCI[ÓO]N|P[ÁA]RRAFO)"
    r"\b[\s.:)\-–—º°]*"
    r"(?:[0-9]+|[IVXLCDM]+\b|(?:primer|segund|tercer|cuart|quint|sext|s[ée]ptim|octav|"
    r"noven|d[ée]cim|[úu]nic|prelimin)\w*)", re.I)


def index_code(path):
    arts, cur, buf, curctx, ctx = {}, None, [], "", {}

    def flush():
        if cur is not None and cur not in arts:
            txt = re.sub(r"\s+", " ", " ".join(x for x in buf if x)).strip()
            arts[cur] = {"art": cur, "text": txt[:1600], "context": curctx}

    for raw in open(path, encoding="utf-8"):
        s = raw.rstrip("\n").strip()
        m = _HEAD2.match(s)
        if m:
            flush()
            cur = m.group(1)
            rest = s[m.end():].strip(" .-–—°º)*")
            buf = [rest] if rest else []
            curctx = " > ".join(v for _, v in sorted(ctx.items())[-2:])
            continue
        if _CTX2.match(s):
            level = len(s) - len(s.lstrip("#"))
            ctx = {k: v for k, v in ctx.items() if k < level}
            ctx[level] = re.sub(r"^#{1,6}\s*\**\s*", "", s).strip()
            continue
        if cur is not None:
            buf.append(s)
    flush()
    return arts


CODE_ARTS = {jur: index_code(path) for jur, path in bim.CODES.items()}


def find_article(jur, art):
    """Locate an article record by its recorded number, with a numeric fallback."""
    d = CODE_ARTS.get(jur, {})
    if str(art) in d:
        return d[str(art)]
    want = re.match(r"\d+", str(art))
    if want and want.group() in d:
        return d[want.group()]
    return None


# ---- integrity checks (computed once) ---------------------------------------
def item_cell_ok(item, c):
    """True/False/None: does the recorded value appear in its source article?"""
    art = item.get("arts", {}).get(c)
    rec = find_article(c, art) if art else None
    if rec is None:
        return None, rec
    year_nums = {n for n, u in bim.values(rec["text"]) if u == item["unit"]}
    return (item["values"][c] in year_nums), rec


ITEM_ISSUES = []
for it in ITEMS:
    for c in [x for x in COUNTRIES if x in it["values"]]:
        ok, _ = item_cell_ok(it, c)
        if ok is not True:
            ITEM_ISSUES.append((it["id"], c, "article not found" if ok is None
                                else "value not extracted"))


def live_verdict(row):
    """Recompute verdict from the stored response text with the real classifier."""
    if row["verdict"] in ("blank", "err"):
        return row["verdict"]
    asked = None if row["condition"] == "unspec" else row["condition"]
    return rob.classify(row["text"], ITEM_BY_ID[row["item"]], asked)["verdict"]


VERDICT_MISMATCHES = [r for r in ROWS
                      if r["verdict"] not in ("blank", "err")
                      and live_verdict(r) != r["verdict"]]

# ---- rendering helpers ------------------------------------------------------
CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e2e2e2;--card:#f7f7f8;
--ok:#137333;--bad:#c5221f;--mark:#fff2a8;--accent:#1a56db;}
@media (prefers-color-scheme:dark){:root{--bg:#15171a;--fg:#e6e6e6;--mut:#9aa0a6;
--line:#2c2f34;--card:#1e2126;--ok:#5bb974;--bad:#f28b82;--mark:#5c5320;--accent:#8ab4f8;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
padding:10px 18px;display:flex;gap:16px;align-items:center}
nav a{color:var(--accent);text-decoration:none;font-weight:600}
nav .sp{flex:1}main{padding:18px;max-width:1100px;margin:0 auto}
h1{font-size:20px}h2{font-size:16px;margin-top:26px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:var(--card)}
.mut{color:var(--mut)}.ok{color:var(--ok);font-weight:700}.bad{color:var(--bad);font-weight:700}
.pill{display:inline-block;padding:1px 7px;border-radius:10px;background:var(--card);
border:1px solid var(--line);font-size:12px}
mark{background:var(--mark);color:inherit;padding:0 2px;border-radius:3px}
.art{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px;
white-space:pre-wrap;max-height:220px;overflow:auto;font-size:13px}
.resp{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px;
white-space:pre-wrap;font-size:13px}
form{display:flex;gap:10px;flex-wrap:wrap;align-items:end;margin:8px 0 16px}
label{display:flex;flex-direction:column;font-size:12px;color:var(--mut);gap:3px}
select,button{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:6px;
background:var(--bg);color:var(--fg)}
button{background:var(--accent);color:#fff;border:none;cursor:pointer;font-weight:600}
.big{font-size:22px;font-weight:700}.card{background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:12px 16px;display:inline-block;margin-right:12px}
code{background:var(--card);padding:1px 5px;border-radius:4px}
"""

# highlight digit/word numbers followed by a time unit, on already-escaped text
_UNIT = r"(?:a[nñ]os?|meses|mes|d[ií]as?)"
_WORDS = "|".join(sorted(rob.WORDS, key=len, reverse=True))
_HL = re.compile(
    rf"(\b\d{{1,3}}\)?\s*{_UNIT}\b|\b(?:{_WORDS})\s+(?:\(\d{{1,3}}\)\s*)?{_UNIT}\b)",
    re.IGNORECASE)


def hl(text):
    return _HL.sub(r"<mark>\1</mark>", html.escape(text or ""))


def page(title, body):
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<nav><b>Conflation inspector</b>
<a href="/">Dashboard</a><a href="/items">Items (article→number)</a>
<a href="/responses">Responses (LLM→number)</a><span class="sp"></span>
<span class="mut">{len(ITEMS)} items · {len(ROWS)} calls · {len(MODELS)} models</span></nav>
<main>{body}</main></body></html>"""


def esc(x):
    return html.escape(str(x))


# ---- views ------------------------------------------------------------------
def view_dashboard():
    ni, nv = len(ITEM_ISSUES), len(VERDICT_MISMATCHES)
    b = ["<h1>Integrity dashboard</h1>",
         f'<div class="card">Item value checks<br><span class="big '
         f'{"bad" if ni else "ok"}">{"OK" if not ni else str(ni)+" issue(s)"}</span>'
         f'<br><span class="mut">recorded value found in its source article</span></div>',
         f'<div class="card">Verdict checks<br><span class="big '
         f'{"bad" if nv else "ok"}">{"OK" if not nv else str(nv)+" mismatch(es)"}</span>'
         f'<br><span class="mut">stored verdict == live re-classification</span></div>']
    if ITEM_ISSUES:
        b.append("<h2>Item issues</h2><table><tr><th>item</th><th>country</th>"
                 "<th>problem</th></tr>")
        for iid, c, why in ITEM_ISSUES:
            b.append(f"<tr><td>{esc(iid)}</td><td>{c}</td><td class='bad'>{esc(why)}"
                     f"</td></tr>")
        b.append("</table>")
    if VERDICT_MISMATCHES:
        q = urlencode({"only": "mismatch"})
        b.append(f'<h2>Verdict mismatches</h2><p><a href="/responses?{q}">'
                 f"view {nv} row(s)</a></p>")
    b.append('<h2>How to read this</h2><ul>'
             '<li><b>Items</b>: each rule\'s per-country number is checked against the '
             'actual article text using the same extractor the miner uses.</li>'
             '<li><b>Responses</b>: each model answer is re-scored live with the same '
             'classifier used in the run; the stored and live verdicts should match.</li></ul>')
    return page("Dashboard", "".join(b))


def view_items():
    b = ["<h1>Items: article → numbers → recorded value</h1>",
         '<p class="mut">Extractor: <code>build_items.values()</code>. '
         'A row is <span class="ok">OK</span> when the recorded value appears among the '
         'numbers extracted from its source article.</p>']
    for it in ITEMS:
        b.append(f'<h2>{esc(it["id"])} — {esc(it["topic"])}</h2>')
        if it.get("note"):
            b.append(f'<p class="mut">{esc(it["note"])}</p>')
        b.append(f'<p><span class="pill">unit: {it["unit"]}</span> '
                 f'<span class="pill">Q: {esc(it["q"])}</span></p>')
        b.append("<table><tr><th>country</th><th>art.</th><th>recorded</th>"
                 "<th>extracted (num,unit)</th><th>check</th><th>article text</th></tr>")
        for c in [x for x in COUNTRIES if x in it["values"]]:
            ok, rec = item_cell_ok(it, c)
            vals = sorted(bim.values(rec["text"])) if rec else []
            valstr = ", ".join(f"{n}{u[0]}" for n, u in vals) or "—"
            if ok is True:
                chk = '<span class="ok">✓ found</span>'
            elif ok is None:
                chk = '<span class="bad">✗ article not found</span>'
            else:
                chk = '<span class="bad">✗ not extracted</span>'
            arttext = hl(rec["text"]) if rec else "<span class='bad'>—</span>"
            ctx = f'<div class="mut">{esc(rec["context"])}</div>' if rec and rec.get("context") else ""
            b.append(f'<tr><td><b>{c}</b></td><td>{esc(it["arts"].get(c,"—"))}</td>'
                     f'<td class="big">{it["values"][c]}</td><td>{esc(valstr)}</td>'
                     f'<td>{chk}</td><td><div class="art">{arttext}</div>{ctx}</td></tr>')
        b.append("</table>")
    return page("Items", "".join(b))


def opt(name, values, cur):
    o = [f'<option value="">{name}</option>']
    for v in values:
        s = " selected" if v == cur else ""
        o.append(f'<option value="{esc(v)}"{s}>{esc(v)}</option>')
    return "".join(o)


def view_responses(q):
    fm = q.get("model", [""])[0]
    fi = q.get("item", [""])[0]
    fc = q.get("condition", [""])[0]
    fv = q.get("verdict", [""])[0]
    only = q.get("only", [""])[0]
    try:
        limit = max(1, min(1000, int(q.get("limit", ["300"])[0])))
    except ValueError:
        limit = 300
    try:
        offset = max(0, int(q.get("offset", ["0"])[0]))
    except ValueError:
        offset = 0

    rows = ROWS
    if fm:
        rows = [r for r in rows if r["model"] == fm]
    if fi:
        rows = [r for r in rows if r["item"] == fi]
    if fc:
        rows = [r for r in rows if r["condition"] == fc]
    if fv:
        rows = [r for r in rows if r["verdict"] == fv]
    if only == "mismatch":
        rows = [r for r in rows if r["verdict"] not in ("blank", "err")
                and live_verdict(r) != r["verdict"]]
    total = len(rows)
    shown = rows[offset:offset + limit]

    conds = ["unspec"] + COUNTRIES
    verds = ["correct", "conflation", "other", "matched", "aclara", "blank", "err"]
    form = (f'<form method="get" action="/responses">'
            f'<label>model<select name="model">{opt("(all)", MODELS, fm)}</select></label>'
            f'<label>item<select name="item">{opt("(all)", list(ITEM_BY_ID), fi)}</select></label>'
            f'<label>condition<select name="condition">{opt("(all)", conds, fc)}</select></label>'
            f'<label>verdict<select name="verdict">{opt("(all)", verds, fv)}</select></label>'
            f'<label>only<select name="only">'
            f'<option value="">all</option>'
            f'<option value="mismatch"{" selected" if only=="mismatch" else ""}>mismatches</option>'
            f'</select></label>'
            f'<button type="submit">filter</button></form>')

    b = ["<h1>Responses: LLM text → numbers → matched → verdict</h1>",
         '<p class="mut">Extractor/classifier: <code>run_probe.extract()/'
         'classify()</code>, recomputed live from the stored text.</p>', form,
         f'<p class="mut">{total} matching · showing {offset+1}–'
         f'{min(offset+limit,total) if total else 0}</p>']

    b.append("<table><tr><th>model</th><th>item / cond</th><th>values</th>"
             "<th>extracted</th><th>matched</th><th>verdict</th><th>prompt & response</th></tr>")
    for r in shown:
        it = ITEM_BY_ID[r["item"]]
        asked = None if r["condition"] == "unspec" else r["condition"]
        if r["verdict"] in ("blank", "err"):
            ext, matched, lv = "—", "—", r["verdict"]
        else:
            res = rob.classify(r["text"], it, asked)
            ext = ", ".join(map(str, res["vals"])) or "—"
            matched = "+".join(res["matched"]) or "—"
            lv = res["verdict"]
        vmark = ("" if lv == r["verdict"]
                 else f' <span class="bad">≠ live:{esc(lv)}</span>')
        valstr = " ".join(f"{k}:{v}" for k, v in it["values"].items())
        model_short = r["model"].split("/")[-1]
        b.append(
            f'<tr><td>{esc(model_short)}</td>'
            f'<td>{esc(r["item"])}<br><span class="pill">{esc(r["condition"])}</span>'
            f' <span class="mut">r{r.get("run","")}</span></td>'
            f'<td class="mut">{esc(valstr)}</td>'
            f'<td>{esc(ext)}</td><td>{esc(matched)}</td>'
            f'<td><b>{esc(r["verdict"])}</b>{vmark}</td>'
            f'<td><div class="mut">{esc(r.get("prompt",""))}</div>'
            f'<div class="resp">{hl(r.get("text",""))}</div></td></tr>')
    b.append("</table>")

    # pagination
    def link(newoff):
        params = {k: v for k, v in
                  (("model", fm), ("item", fi), ("condition", fc), ("verdict", fv),
                   ("only", only), ("limit", limit), ("offset", newoff)) if v}
        return "/responses?" + urlencode(params)
    nav = []
    if offset > 0:
        nav.append(f'<a href="{link(max(0,offset-limit))}">← prev</a>')
    if offset + limit < total:
        nav.append(f'<a href="{link(offset+limit)}">next →</a>')
    if nav:
        b.append('<p>' + " · ".join(nav) + '</p>')
    return page("Responses", "".join(b))


# ---- server -----------------------------------------------------------------
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                body = view_dashboard()
            elif u.path == "/items":
                body = view_items()
            elif u.path == "/responses":
                body = view_responses(q)
            else:
                self.send_error(404)
                return
        except Exception as e:  # noqa: BLE001
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"error: {e}".encode("utf-8"))
            return
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    port = args.port
    for _ in range(10):
        try:
            srv = HTTPServer(("127.0.0.1", port), H)
            break
        except OSError:
            port += 1
    else:
        sys.exit("no free port")
    url = f"http://127.0.0.1:{port}"
    print(f"Inspecting {len(ITEMS)} items and {len(ROWS)} responses.")
    print(f"Item issues: {len(ITEM_ISSUES)}  |  verdict mismatches: {len(VERDICT_MISMATCHES)}")
    print(f"Serving at {url}  (Ctrl+C to stop)")
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
