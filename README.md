# probing-civil-codes

**Which country's code?** A small, reproducible probe for *jurisdictional
conflation* in large language models: does a model confuse the civil law of one
Spanish-speaking country with another?

Many civil codes state the *same rule* with a *different number*. The period to
acquire a servitude by prescription, for example, is 20 years in Spain, 5 in
Chile, and 8 in Colombia — same legal concept, three different answers. If you
ask a model the Colombian rule and it returns Spain's 20, that is not a
hallucination of a fact that doesn't exist; it is a **cross-jurisdiction
mix-up**, and it is invisible unless you check the number against the *right*
country's code.

This repo builds a benchmark of such provisions across **ES, CL, CO, AR**, runs
each model once per country (plus an unspecified-country baseline), and scores
the answers deterministically by extracting the number.

Companion short paper (NLLP 2026): *"Which Country's Code? Probing
Jurisdictional Conflation across Spanish-Language Civil Codes in Large Language
Models."*

## Headline finding

Conflation is **capability-dependent**, not a fixed property of "LLMs":

- **Pooled clean accuracy** falls in a clear jurisdictional hierarchy —
  ES 0.90 > AR 0.77 > CL 0.67 ≫ **CO 0.29**. Colombia is the residual hard case
  for *every* model tested (none exceeds 0.50 clean on CO).
- **Spain is the default attractor.** Under unspecified prompts the Spanish
  value appears in 72% of answers, and 52% of all unambiguous wrong answers land
  on Spain's number; Colombia is almost never a wrong destination (2%).
- **The strongest models fail safe.** On clean Colombian cells the two best
  models abstain rather than emit another country's number; a weak model stamps
  Spain's value everywhere.

See the paper and `analysis_clean.csv` for the full breakdown. The "clean"
qualifier matters — read *Why the "clean" subanalysis* below.

## The method in one paragraph

Pick civil-code provisions where the rule is shared but the time period differs
across countries. For each item, prompt every model with the rule and ask for
the period, once with **no country named** (baseline) and once **per country**
that has a value. Repeat a few times per condition. Extract the number and unit
from each answer and compare it to the code of the *asked* country. An answer
that matches the asked country is **correct**; one that matches a *different*
country's value is **conflation**; anything else is **other** (or `blank`/`err`).

## Repository layout

| File | What it is |
|------|------------|
| `items.json` | **13 source-verified items.** Each has the shared rule, the article number per country, and the value (in years/months/days). This is the ground truth. |
| `run_probe.py` | Sends the prompts to OpenRouter and writes `results_raw.jsonl` + `results_summary.csv`. |
| `analyze_conflation.py` | Scores the raw results and writes the `analysis_*.csv` tables, including the clean subanalysis. |
| `inspect_app.py` | Local web app to **manually verify the whole pipeline** — article→number and LLM-response→number→verdict — so you can eyeball that nothing is misparsed. |
| `build_items.py` | The candidate miner (TF-IDF, Spain-pivot alignment) used to *find* candidate provisions. Its output (`items_candidates.*`) is intermediate; every item in `items.json` was hand-verified against the source. |
| `results_raw.jsonl` | One JSON line per API call from the reported run (1,302 calls, 7 models). |
| `results_summary.csv`, `analysis_*.csv` | Per-(model, country) scored tables. |
| `items_candidates.{json,csv}` | Raw miner output; not the benchmark. |

## The 13 items

Shared rule → period per country (`-` = not in that code). Full article numbers
live in `items.json`.

| id | rule (short) | ES | CL | CO | AR |
|----|--------------|----|----|----|----|
| serv-adq | acquire servitude by prescription | 20 | 5 | 8 | – |
| serv-ext | extinguish servitude by non-use | 20 | 3 | 20 | 10 |
| donacion | revoke donation (ingratitude) | 1 | 4 | 4 | 1 |
| muebles | prescription of movables | 3 | 2 | 3 | – |
| inmuebles | prescription of immovables | 10 | 5 | 5 | 10 |
| extraordinaria | extraordinary prescription | 30 | 10 | 10 | 20 |
| ruina | liability for building ruin | 10 | 5 | 10 | 10 |
| fiador | guarantor's release period | 10 | 5 | 10 | 5 |
| indivision | forced-indivision pact limit | 10 | 5 | 5 | 10 |
| nulidad | action to annul a contract | 4 | 4 | 4 | 2 |
| retroventa | right of repurchase | 4 | 4 | 4 | 5 |
| muerte-presunta | presumption of death | 10 | 5 | 2 | 3 |
| honorarios | prescription of professional fees | 3 | 2 | 3 | – |

`muerte-presunta` is the only item where all four countries differ; `serv-adq`
is the only other fully-distinct (three-country) item. Distinct-across-all-four
rules are rare because Chile (1855) and Colombia (1873) both descend from
Bello's code and share many numbers — which is exactly why the analysis works at
the *cell* level, not the item level (below).

## Reproduce it

Pure Python standard library — **no pip install**. Python 3.9+.

```bash
# 1. run the probe (needs an OpenRouter API key; this is the only billed step)
export OPENROUTER_API_KEY=...          # or put it in a .env file next to run_probe.py
python run_probe.py                     # all 7 default models, 3 runs/condition
#   options: --models slug ...   --items items.json   --runs N   --workers N

# 2. score and build the tables (reads local files only, nothing is sent)
python analyze_conflation.py            # writes analysis_overall/clean/item_structure.csv

# 3. manually verify the pipeline in a browser
python inspect_app.py                   # then open the printed localhost URL
```

`run_probe.py` is the *only* step that makes network calls. `analyze_*` and
`inspect_app` are fully offline.

## Why the "clean" subanalysis

Because Chile and Colombia share many values, a naive count of "wrong-country
answers" is inflated by **coincidences**: if the CL and CO values are both 5,
you can't tell a CL→CO mix-up from a right answer, or attribute a "5" to either
country. The clean pass fixes this by restricting to the cells where attribution
is unambiguous:

- score a query only when the **asked** country's value is *unique* within that
  item (so a matched number can only be that country's), and
- count a conflation *destination* only when the emitted wrong value maps to
  **exactly one** other country.

Every rate and every arrow in `analysis_clean.csv` is collision-free. Report
those numbers, not the raw counts.

## Verdict vocabulary

`correct` (matches asked country) · `conflation` (matches a different country) ·
`other` (a number, but nobody's) · `matched` on the unspecified baseline (whose
value leaked in) · `aclara` (model asks which country) · `blank` · `err`.

## Data provenance & scope

Article texts come from the `legalize-*` civil-code corpora (ES/CL/CO/AR;
Uruguay was dropped for coverage gaps). Colombian articles use inline
`**Artículo. N.**` markers rather than Markdown headings, and one Argentine
article carries a stray heading fragment — `inspect_app.py` has a dedicated
indexer that handles both, and the dashboard reports **0 item issues / 0 verdict
mismatches** when the data and scorer agree.

This is a targeted diagnostic on 13 provisions and four jurisdictions, not a
comprehensive legal benchmark. It is meant to make one failure mode — silently
answering with the wrong country's law — measurable.
