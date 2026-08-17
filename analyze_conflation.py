#!/usr/bin/env python3
"""
Analysis + subanalysis for the multi-country conflation probe.

Reads a curated item file and a raw results file (one JSON line per API call,
as written by run_probe.py) and produces:

  1. ITEM STRUCTURE   per item: values per country, how many DISTINCT values,
                      and which countries hold a UNIQUE value (no collision).
  2. OVERALL table    per (model, asked): correct / conflation / other + the
                      conflation-destination matrix (conflation rows only).
  3. CLEAN subanalysis  --- the "cases where the countries disagree" cut ---
                      Attribution is only unambiguous when values do not collide,
                      so this restricts to cells where the ASKED country's value
                      is UNIQUE among the countries present in that item (a matched
                      number then can only be that country's), and counts a
                      conflation DESTINATION only when the emitted wrong value maps
                      to exactly ONE other country. This removes the value-collision
                      confound from every rate and every arrow.
  4. FULLY-DISTINCT items   the strictest cut: items where every present country
                      states a different value (all cells unambiguous at once).

Why this matters: CL and CO (both Bello-lineage) share many values, so a bare
conflation count is inflated by coincidences. The clean subanalysis reports only
what the data can actually attribute.

Usage:
  python analyze_conflation.py                              # items.json + results_raw.jsonl
  python analyze_conflation.py --items items.json --raw results_raw.jsonl --out-prefix analysis

Outputs (next to this script, default prefix 'analysis'):
  analysis_item_structure.csv
  analysis_overall.csv
  analysis_clean.csv          per (model, asked) restricted to unique-value cells
Nothing is sent anywhere.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
COUNTRIES = ["ES", "CL", "CO", "AR"]


def load_items(path):
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    by_id = {}
    for it in items:
        vals = it["values"]                              # {country: int}
        present = [c for c in COUNTRIES if c in vals]
        # count how many present countries share each value
        vcount = defaultdict(int)
        for c in present:
            vcount[vals[c]] += 1
        unique = {c: (vcount[vals[c]] == 1) for c in present}
        # map a value -> the single country holding it, else None (collision)
        owner = {}
        for c in present:
            owner.setdefault(vals[c], []).append(c)
        sole = {v: cs[0] for v, cs in owner.items() if len(cs) == 1}
        by_id[it["id"]] = {
            "item": it, "present": present, "unique": unique,
            "sole_owner": sole, "n_distinct": len(set(vals[c] for c in present)),
            "all_distinct": len(set(vals[c] for c in present)) == len(present),
        }
    return items, by_id


def load_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="items.json")
    ap.add_argument("--raw", default="results_raw.jsonl")
    ap.add_argument("--out-prefix", default="analysis")
    args = ap.parse_args()

    items, meta = load_items(HERE / args.items)
    rows = load_rows(HERE / args.raw)
    prefix = args.out_prefix or "analysis"
    models = sorted({r["model"] for r in rows})

    # ---------------- 1. item structure ----------------
    print("=" * 70)
    print("ITEM STRUCTURE  (values per country; * = unique value, no collision)")
    print("=" * 70)
    struct_csv = HERE / f"{prefix}_item_structure.csv"
    with struct_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "unit", *COUNTRIES, "n_present", "n_distinct",
                    "all_distinct", "unique_countries"])
        for it in items:
            m = meta[it["id"]]
            cells = []
            for c in COUNTRIES:
                if c in m["present"]:
                    star = "*" if m["unique"][c] else " "
                    cells.append(f"{c}={it['values'][c]}{star}")
                else:
                    cells.append(f"{c}=-")
            uq = "".join(c for c in m["present"] if m["unique"][c]) or "-"
            tag = "ALL-DISTINCT" if m["all_distinct"] else ""
            print(f"  {it['id']:<15}{'  '.join(cells):<34} distinct={m['n_distinct']}"
                  f"/{len(m['present'])}  unique[{uq}] {tag}")
            w.writerow([it["id"], it["unit"],
                        *[it["values"].get(c, "") for c in COUNTRIES],
                        len(m["present"]), m["n_distinct"], m["all_distinct"], uq])
    fully = [it["id"] for it in items if meta[it["id"]]["all_distinct"]]
    print(f"\n  Fully-distinct items (every present country differs): "
          f"{fully if fully else 'NONE'}")

    # ---------------- helpers over rows ----------------
    def verdict_of(r, asked):
        matched = r.get("matched", [])
        if r["verdict"] in ("blank", "err"):
            return r["verdict"], matched
        if asked in matched:
            return "correct", matched
        if matched:
            return "conflation", matched
        return "other", matched

    def summarize(subset, clean):
        """Return per-(model,asked) stats. clean=True restricts to unique-value
        cells (asked value unique) and unambiguous conflation destinations."""
        out = {}
        for m in models:
            for asked in COUNTRIES:
                rs = [r for r in subset if r["model"] == m
                      and r["condition"] == asked
                      and (not clean or meta[r["item"]]["unique"].get(asked, False))]
                if not rs:
                    continue
                blank = sum(1 for r in rs if r["verdict"] == "blank")
                err = sum(1 for r in rs if r["verdict"] == "err")
                answered = len(rs) - blank - err
                correct = conf = other = 0
                dest = defaultdict(int)
                for r in rs:
                    v, matched = verdict_of(r, asked)
                    if v == "correct":
                        correct += 1
                    elif v == "conflation":
                        conf += 1
                        wrong = [c for c in matched if c != asked]
                        if clean:
                            # count destination only if the emitted value maps to
                            # exactly one country (unambiguous arrow)
                            sole = meta[r["item"]]["sole_owner"]
                            attributable = {sole[r["values"][c]] for c in wrong
                                            if r["values"][c] in sole}
                            if len(attributable) == 1:
                                dest[next(iter(attributable))] += 1
                        else:
                            for c in wrong:
                                dest[c] += 1
                    elif v == "other":
                        other += 1
                out[(m, asked)] = {
                    "n": len(rs), "answered": answered, "correct": correct,
                    "conf": conf, "other": other, "blank": blank, "err": err,
                    "rate": round(conf / answered, 3) if answered else "",
                    "dest": dest,
                }
        return out

    def print_table(title, stats):
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        for m in models:
            print(f"\n=== {m} ===")
            for asked in COUNTRIES:
                s = stats.get((m, asked))
                if not s:
                    continue
                tostr = " ".join(f"{c}:{s['dest'][c]}" for c in COUNTRIES if s["dest"][c])
                print(f"  asked {asked}: correct {s['correct']}/{s['answered']}  "
                      f"conflation {s['conf']} (rate {s['rate']}) -> [{tostr}]  "
                      f"other {s['other']}"
                      + (f"  blank {s['blank']}" if s['blank'] else ""))

    def write_csv(name, stats):
        p = HERE / name
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "asked", "n", "answered", "correct", "conflation",
                        "conflation_rate", "other", "blank", "err",
                        *[f"to_{c}" for c in COUNTRIES]])
            for m in models:
                for asked in COUNTRIES:
                    s = stats.get((m, asked))
                    if not s:
                        continue
                    w.writerow([m, asked, s["n"], s["answered"], s["correct"],
                                s["conf"], s["rate"], s["other"], s["blank"], s["err"],
                                *[s["dest"][c] for c in COUNTRIES]])
        return p

    # ---------------- 2. unspec default lean ----------------
    print("\n" + "=" * 70)
    print("UNSPEC DEFAULT LEAN  (country not named -> whose value appears)")
    print("=" * 70)
    for m in models:
        us = [r for r in rows if r["model"] == m and r["condition"] == "unspec"]
        lean = defaultdict(int)
        for r in us:
            for c in r.get("matched", []):
                lean[c] += 1
        none = sum(1 for r in us if not r.get("matched"))
        print(f"  {m:<36} "
              + "  ".join(f"{c}:{lean[c]}" for c in COUNTRIES) + f"   none:{none}")

    # ---------------- 3. overall + clean tables ----------------
    overall = summarize(rows, clean=False)
    clean = summarize(rows, clean=True)
    print_table("OVERALL  (all named-country rows; destinations count every match)",
                overall)
    print_table("CLEAN SUBANALYSIS  (only cells where asked value is UNIQUE; "
                "destinations counted only when unambiguous)", clean)

    op = write_csv(f"{prefix}_overall.csv", overall)
    cp = write_csv(f"{prefix}_clean.csv", clean)

    # ---------------- 4. aggregate clean headline ----------------
    print("\n" + "=" * 70)
    print("CLEAN AGGREGATE  (all models pooled, unique-value cells only)")
    print("=" * 70)
    for asked in COUNTRIES:
        ans = corr = conf = 0
        dest = defaultdict(int)
        for m in models:
            s = clean.get((m, asked))
            if not s:
                continue
            ans += s["answered"]; corr += s["correct"]; conf += s["conf"]
            for c in COUNTRIES:
                dest[c] += s["dest"][c]
        if not ans:
            continue
        tostr = " ".join(f"{c}:{dest[c]}" for c in COUNTRIES if dest[c])
        rate = round(conf / ans, 3) if ans else ""
        print(f"  asked {asked}: correct {corr}/{ans}  conflation {conf} "
              f"(rate {rate}) -> [{tostr}]")
    # sink totals (who absorbs conflation, clean arrows only)
    sink = defaultdict(int)
    for (m, asked), s in clean.items():
        for c in COUNTRIES:
            sink[c] += s["dest"][c]
    tot = sum(sink.values()) or 1
    print("\n  Clean conflation SINKS (share of all unambiguous wrong arrows):")
    for c in sorted(COUNTRIES, key=lambda c: -sink[c]):
        print(f"    {c}: {sink[c]:3d}  ({sink[c]/tot:.0%})")

    print(f"\nWrote {struct_csv.name}, {op.name}, {cp.name}")


if __name__ == "__main__":
    main()
