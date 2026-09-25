import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HYBRID_ROOT = ROOT / "experiments" / "results" / "hybrid"
OVERNIGHT_CHECKPOINTS = ROOT / "experiments" / "results" / "checkpoints"
BSON_PATH = ROOT / "bson_files" / "aviation-preprocessed-with-docsent-fixed.bson"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import importlib.util
from wannadb.data.data import DocumentBase
from util import consider_overlap_as_match, get_document_by_name

def load_run(run_tag: str) -> Dict[str, Any]:
    root = HYBRID_ROOT / run_tag
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    jobs = json.loads((root / "jobs.json").read_text(encoding="utf-8"))
    units = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "checkpoints").glob("*.json"))]
    return {"config": config, "jobs": jobs, "units": units, "root": root,
            "done_marker": (root / "GRID_DONE").exists()}


def complete_combinations(run: Dict[str, Any]) -> set:

    need = run["config"]["num_seeds"] * len(run["config"]["attributes"])
    counts = Counter((u["variant"], u["k"]) for u in run["units"])
    return {key for key, n in counts.items() if n >= need}


def progress_table(run: Dict[str, Any]) -> pd.DataFrame:
    done = {u["job"] for u in run["units"]}
    rows = [{**j, "done": j["job"] in done} for j in run["jobs"]]
    df = pd.DataFrame(rows)
    return (df.groupby("variant")["done"].agg(["sum", "count"])
              .rename(columns={"sum": "finished units", "count": "planned units"}))


def per_round_frame(run: Dict[str, Any], complete_only: bool = True) -> pd.DataFrame:

    max_rounds = run["config"]["max_rounds"]
    start = {}
    for u in run["units"]:
        if u["variant"] == "A_no_feedback":
            for row in u["final_scores"]:
                start[(u["seed_idx"], row["attribute"])] = row

    complete = complete_combinations(run)
    rows = []
    for u in run["units"]:
        if u["variant"] == "A_no_feedback":
            continue
        if complete_only and (u["variant"], u["k"]) not in complete:
            continue
        for attr, history in u["per_round"].items():
            base = start.get((u["seed_idx"], attr))
            by_round = {0: base} if base is not None else {}
            by_round.update({r["round"]: r for r in history})
            last = None
            for rnd in range(0, max_rounds + 1):
                last = by_round.get(rnd, last)
                if last is None:
                    continue
                rows.append({"variant": u["variant"], "h": u["k"], "seed": u["seed_idx"], "attribute": attr,
                             "round": rnd, "answerer": ("none" if rnd == 0 else "human" if rnd <= u["k"] else "llm"),
                             "precision": last["precision"], "recall": last["recall"], "f1": last["f1_score"]})
    return pd.DataFrame(rows)


def llm_round_frame(run: Dict[str, Any], complete_only: bool = True) -> pd.DataFrame:
    complete = complete_combinations(run)
    rows = []
    for u in run["units"]:
        if complete_only and (u["variant"], u["k"]) not in complete:
            continue
        for attr, recs in (u.get("llm_rounds") or {}).items():
            for rec in recs:
                rows.append({"variant": u["variant"], "h": u["k"], "seed": u["seed_idx"], "attribute": attr, **rec})
    return pd.DataFrame(rows)


def unit_frame(run: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for u in run["units"]:
        rows.append({"job": u["job"], "variant": u["variant"], "h": u["k"], "seed": u["seed_idx"],
                     "t_total_h": u["t_total_seconds"] / 3600, "t_llm_h": u["t_llm_inference_seconds"] / 3600,
                     "t_matching_min": u["t_wannadb_matching_seconds"] / 60,
                     "snapshots_consistent": all(u["snapshot_matches_final"].values()) if u["snapshot_matches_final"] else None,
                     "mean_final_f1": np.mean([r["f1_score"] for r in u["final_scores"]])})
    return pd.DataFrame(rows)


# aggregation
def macro_curve(per_round: pd.DataFrame, variant: str, h: int) -> pd.DataFrame:
    """Macro F1 per round for one (variant, h): mean over attributes per seed, then mean and std over seeds"""
    sub = per_round[(per_round.variant == variant) & (per_round.h == h)]
    per_seed = sub.groupby(["seed", "round"])["f1"].mean().reset_index()
    out = per_seed.groupby("round")["f1"].agg(["mean", "std", "count"]).reset_index()
    return out.rename(columns={"count": "seeds"})


def budget_matrix(per_round: pd.DataFrame, variant: str, metric: str = "f1") -> pd.DataFrame:

    sub = per_round[per_round.variant == variant]
    if variant == "examples":
        max_rounds = int(per_round["round"].max())
        edges = per_round[(per_round.variant == "handoff") & (per_round.h.isin([0, max_rounds]))]
        sub = pd.concat([sub, edges])
    mat = sub.groupby(["h", "round", "seed"])[metric].mean().groupby(["h", "round"]).mean().unstack("round")
    for h in mat.index:
        mat.loc[h, [c for c in mat.columns if c < h]] = np.nan
    return mat.sort_index()


def human_rounds_needed(matrix: pd.DataFrame, reference: pd.Series, tolerance: float) -> pd.Series:
    needed = {}
    for r in matrix.columns:
        col = matrix[r].dropna()
        ok = col[col >= reference.get(r, np.nan) - tolerance]
        needed[r] = ok.index.min() if len(ok) else np.nan
    return pd.Series(needed, name=f"h needed (within {tolerance:.2f})")


def equivalent_human_rounds(per_round: pd.DataFrame, variant: str, budget: int,
                            attributes: Optional[List[str]] = None) -> pd.DataFrame:
    """How much human work does the LLM replace? """
    df = per_round if attributes is None else per_round[per_round.attribute.isin(attributes)]
    max_rounds = int(df["round"].max())
    human = df[(df.variant == "handoff") & (df.h == max_rounds)]
    human_curve = human.groupby(["seed", "round"])["f1"].mean().groupby("round").mean()

    monotone = np.maximum.accumulate(human_curve.values)
    mat = budget_matrix(df, variant)
    rows = []
    for h in mat.index:
        if h > budget or budget not in mat.columns or np.isnan(mat.loc[h, budget]):
            continue
        f1 = mat.loc[h, budget]
        eq = float(np.interp(f1, monotone, human_curve.index.values)) if f1 <= monotone[-1] else np.nan
        rows.append({"human rounds h": h, "LLM rounds": budget - h, "macro F1": f1,
                     "equivalent human-only rounds": eq,
                     "human rounds saved": (eq - h) if not np.isnan(eq) else np.nan})
    return pd.DataFrame(rows).set_index("human rounds h")


def cumulative_llm_cost(llm_rounds: pd.DataFrame, variant: str, h: int, max_rounds: int) -> pd.DataFrame:
    sub = llm_rounds[(llm_rounds.variant == variant) & (llm_rounds.h == h)].copy()
    if sub.empty:
        return pd.DataFrame({"round": range(max_rounds + 1), "tokens": 0.0, "seconds": 0.0})
    sub["tokens"] = sub["prompt_tokens"] + sub["completion_tokens"]
    per_seed_round = sub.groupby(["seed", "round"])[["tokens", "seconds"]].sum()
    seeds = sub["seed"].unique()
    full = pd.MultiIndex.from_product([seeds, range(max_rounds + 1)], names=["seed", "round"])
    cum = per_seed_round.reindex(full, fill_value=0).groupby(level="seed").cumsum()
    return cum.groupby(level="round").mean().reset_index()


# proposal 3: how high can F1 get with the nuggets that exist?
def nugget_ceiling(bson_path: Path = BSON_PATH) -> pd.DataFrame:

    spec = importlib.util.spec_from_file_location(
        "aviation_dataset", str(ROOT / "wannadb_datsets" / "datasets" / "aviation" / "aviation.py"))
    aviation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aviation)
    ground_truth = aviation.load_dataset()
    db = DocumentBase.from_bson(Path(bson_path).read_bytes())

    rows = []
    for attr in aviation.ATTRIBUTES:
        with_value = covered = 0
        for doc in db.documents:
            gt = get_document_by_name(ground_truth, doc.name)
            mentions = gt["mentions"].get(attr, []) if gt else []
            if not mentions:
                continue
            with_value += 1
            if any(consider_overlap_as_match(m["start_char"], m["end_char"], n.start_char, n.end_char)
                   for m in mentions for n in doc.nuggets):
                covered += 1
        coverage = covered / with_value if with_value else 1.0
        rows.append({"attribute": attr, "documents with a value": with_value,
                     "value covered by a nugget": covered, "coverage": coverage,
                     "F1 ceiling": 2 * coverage / (1 + coverage) if coverage > 0 else 0.0})
    return pd.DataFrame(rows).set_index("attribute")

def overnight_final_scores(run_tag: str = "overnight_20260908-0918") -> pd.DataFrame:
    rows = []
    for path in sorted(OVERNIGHT_CHECKPOINTS.glob(f"{run_tag}_*_seed*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        for row in rec["scores"]:
            rows.append({"baseline": rec["baseline"], "seed": rec["seed_idx"], "attribute": row["attribute"],
                         "f1": row["f1_score"], **{k: row[k] for k in row if k.startswith("num_")}})
    return pd.DataFrame(rows)


def compare_with_overnight(run: Dict[str, Any], h: int, overnight_baseline: str, round_no: int = 25) -> pd.DataFrame:

    overnight = overnight_final_scores()
    overnight = overnight[overnight.baseline == overnight_baseline].set_index(["seed", "attribute"])
    rows = []
    for u in run["units"]:
        if u["variant"] != "handoff" or u["k"] != h:
            continue
        for attr, history in u["per_round"].items():
            snap = next((r for r in history if r["round"] == round_no), None)
            key = (u["seed_idx"], attr)
            if snap is None or key not in overnight.index:
                continue
            old = overnight.loc[key]
            rows.append({"seed": u["seed_idx"], "attribute": attr, "f1 now (round %d)" % round_no: snap["f1_score"],
                         "f1 overnight": old["f1"],
                         "identical counts": all(snap[c] == old[c] for c in snap if c.startswith("num_"))})
    return pd.DataFrame(rows)


def compare_prefix(short_tag: str, long_tag: str) -> pd.DataFrame:

    short, long_ = load_run(short_tag), load_run(long_tag)
    r = short["config"]["max_rounds"]
    long_units = {}
    for u in long_["units"]:
        if u["variant"] == "handoff" and u["k"] == long_["config"]["max_rounds"]:
            for attr in u["per_round"]:
                long_units[(u["seed_idx"], attr)] = u
    rows = []
    for u in short["units"]:
        if u["variant"] != "handoff" or u["k"] != r:
            continue
        for row in u["final_scores"]:
            attr = row["attribute"]
            other = long_units.get((u["seed_idx"], attr))
            if other is None:
                continue
            snap = next((s for s in other["per_round"][attr] if s["round"] == r), None)
            if snap is None:
                continue
            rows.append({"seed": u["seed_idx"], "attribute": attr,
                         f"f1, run with budget {r}": row["f1_score"],
                         f"f1 after round {r} of the budget-{long_['config']['max_rounds']} run": snap["f1_score"],
                         "identical counts": all(row[c] == snap[c] for c in row if c.startswith("num_"))})
    return pd.DataFrame(rows)
