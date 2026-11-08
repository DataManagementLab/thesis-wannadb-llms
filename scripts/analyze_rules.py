import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))
from analyze_attributes import md
from analyze_round_effect import document_id, is_right, load_gold, unit_rounds
from analyze_strategies import HYBRID, f1_at, load
from stored_feedback import load_llm_answers, read_documents
from util import consider_overlap_as_match

HUMAN_ONLY = "ds_base_20261024"


def answered_documents(unit: Dict[str, Any]) -> set:
    """Documents an answer of the table was used for in this column, the human rounds cost no LLM"""
    return {document_id(name) for a in unit["answers"].get(unit["attribute"], []) if a[1] == "llm" and a[3]
            for name in a[3].split(",")}


def rule_rows(tag: str, gold: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for unit in load(tag):
        if unit["variant"] == "A_no_feedback":
            continue
        rounds = pd.DataFrame(unit_rounds(unit, gold, tag))
        llm = rounds[rounds["answerer"] == "llm"]
        rows.append({"grid": tag, "variant": unit["variant"], "seed": unit["seed_idx"], "attribute": unit["attribute"],
                     "f1": f1_at(unit, unit["budget"]), "f1 after 10": f1_at(unit, 10), "f1 after 20": f1_at(unit, 20),
                     "rows already right": (llm["kind"] == "row was right").mean(),
                     "own row fixed": (llm["kind"] == "row fixed").mean(), "other rows net": llm["other rows net"].sum(),
                     "llm rounds": len(llm), "rows asked": llm["rows asked"].sum(), "rows fixed": llm["rows fixed"].sum(),
                     "documents": len(answered_documents(unit))})
    return pd.DataFrame(rows)


def documents_per_table(tag: str) -> pd.DataFrame:
    """Distinct documents over all columns of a seed: what a cache with these rows would have to read."""
    rows = []
    for variant_seed, units in pd.DataFrame(
            [{"variant": u["variant"], "seed": u["seed_idx"], "unit": u} for u in load(tag)
             if u["variant"] != "A_no_feedback"]).groupby(["variant", "seed"]):
        asked = set().union(*(answered_documents(u) for u in units["unit"]))
        rows.append({"variant": variant_seed[0], "seed": variant_seed[1], "documents read": len(asked)})
    return pd.DataFrame(rows).groupby("variant")[["documents read"]].mean().round(1)


def rest_empty_f1(answers: Dict[str, Dict[str, List[List[int]]]], gold: Dict[str, Dict[str, Any]],
                  attribute: str, read: set) -> float:
    right = wrong = filled_gold = 0
    for document, mentions in gold.items():
        spans = answers.get(document, {}).get(attribute, []) if document in read else []
        wanted = mentions.get(attribute, [])
        filled_gold += bool(wanted)
        if spans and any(consider_overlap_as_match(m["start_char"], m["end_char"], s, e)
                         for m in wanted for s, e in spans):
            right += 1
        elif spans:
            wrong += 1
    precision = right / (right + wrong) if right + wrong else 1.0
    recall = right / filled_gold if filled_gold else 1.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def read_curve(tag: str) -> pd.DataFrame:
    config = json.loads((HYBRID / tag / "config.json").read_text(encoding="utf-8"))
    dataset = config.get("dataset", "aviation")
    gold = load_gold(dataset)
    ground_truth = [json.loads(p.read_text(encoding="utf-8"))
                    for p in (ROOT / "wannadb_datsets" / "datasets" / dataset / "documents").glob("*.json")]
    order = config.get("read_order", "random")
    rows = []
    for unit in load(tag):
        variant, seed, attribute = unit["variant"], unit["seed_idx"], unit["attribute"]
        count = (0 if variant == "A_no_feedback" else
                 int(variant.split("_")[1]) if variant.startswith("reads_") else len(ground_truth))
        answers = load_llm_answers(config["stored_answers"].replace(
            "/home/ahouari/thesis-wannadb", str(ROOT)), seed, ground_truth)
        read = read_documents(ground_truth, count, config["seed_values"][seed], order) if count else set()
        rows.append({"documents read": count, "seed": seed, "attribute": attribute,
                     "with WannaDB": unit["final_scores"][0]["f1_score"],
                     "rest empty": rest_empty_f1(answers, gold, attribute, read)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", nargs="*", default=["rule_llm10_20261106", "rule_llmall_20261106"])
    parser.add_argument("--reads", nargs="*", default=["reads_llm_20261106"])
    parser.add_argument("--name", default="rules")
    parser.add_argument("--out", type=Path, default=ROOT / "info" / "analysis")
    args = parser.parse_args()

    gold = load_gold("aviation")
    with open(args.out / f"{args.name}.md", "w", encoding="utf-8") as fh:
        fh.write("Generated by scripts/analyze_rules.py. Answers from the LLM-only run (one repeat per seed), no LLM "
                 "request, seeds 0-2, WannaDB's match rule, c02 only.\n")
        if args.rules:
            human = [u for u in load(HUMAN_ONLY) if u["k"] == 40 and u["variant"] == "handoff"]
            llm = [u for u in load(HUMAN_ONLY) if u["k"] == 0 and u["variant"] == "handoff"]
            reference = pd.DataFrame([
                {"reference": "human only, 40 rounds", "macro-F1": sum(f1_at(u, 40) for u in human) / len(human)},
                {"reference": "real LLM, 40 rounds", "macro-F1": sum(f1_at(u, 40) for u in llm) / len(llm)}])
            fh.write(f"\n## References from {HUMAN_ONLY}\n\n{md(reference, index=False)}\n")
            print(md(reference, index=False))

        for tag in args.rules:
            rows = rule_rows(tag, gold)
            per_seed = rows.groupby(["variant", "seed"]).agg(
                f1=("f1", "mean"), f10=("f1 after 10", "mean"), f20=("f1 after 20", "mean"),
                right=("rows already right", "mean"), fixed=("own row fixed", "mean"),
                other=("other rows net", "mean"), rounds=("llm rounds", "mean"), asked=("rows asked", "mean"),
                rows_fixed=("rows fixed", "mean")).reset_index()
            table = per_seed.groupby("variant").agg(
                macro_f1=("f1", "mean"), std_over_seeds=("f1", "std"), f1_after_10=("f10", "mean"),
                f1_after_20=("f20", "mean"), rounds_per_column=("rounds", "mean"), rows_per_column=("asked", "mean"),
                rows_fixed_per_column=("rows_fixed", "mean"), rounds_already_right=("right", "mean"),
                other_rows_net_per_column=("other", "mean")).round(3)
            table = table.join(documents_per_table(tag)).sort_values("macro_f1", ascending=False)
            columns = rows.groupby(["attribute", "variant"])["f1"].mean().unstack().round(3)
            fh.write(f"\n## Pick rules, {tag}\n\n{md(table)}\n\n### Per column\n\n{md(columns)}\n")
            print(tag), print(md(table))
            rows.to_csv(args.out / f"{args.name}_{tag}.csv", index=False)

        compared = {}
        for tag in args.reads:
            curve = read_curve(tag)
            curve.to_csv(args.out / f"{args.name}_{tag}.csv", index=False)
            per_seed = curve.groupby(["documents read", "seed"])[["with WannaDB", "rest empty"]].mean().reset_index()
            table = per_seed.groupby("documents read").agg(
                with_wannadb=("with WannaDB", "mean"), std_over_seeds=("with WannaDB", "std"),
                rest_empty=("rest empty", "mean")).round(3)
            table["gain"] = (table["with_wannadb"] - table["rest_empty"]).round(3)
            columns = curve.groupby(["attribute", "documents read"])["with WannaDB"].mean().unstack().round(3)
            fh.write(f"\n## Read curve, {tag}\n\nThe LLM reads N documents (the same N in every column of a seed) and "
                     f"its answers go to WannaDB as feedback.\n\n{md(table)}\n\n### With WannaDB, per column\n\n"
                     f"{md(columns)}\n")
            compared[tag] = table["with_wannadb"]
            print(tag), print(md(table))
        if len(compared) > 1:
            table = pd.DataFrame(compared)
            fh.write(f"\n## Read curves compared, macro-F1 with WannaDB\n\n{md(table)}\n")
            print(md(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
