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
from analyze_strategies import HYBRID, load
from util import consider_overlap_as_match

DATASETS = ROOT / "wannadb_datsets" / "datasets"
PHASES = {"2-5": (2, 5), "6-20": (6, 20), "21-40": (21, 40), "41+": (41, 10_000)}


def document_id(name: str) -> str:
    return Path(name.replace("\\", "/")).stem


def load_gold(dataset: str) -> Dict[str, Dict[str, Any]]:
    gold = {}
    for path in (DATASETS / dataset / "documents").glob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        gold[document["id"]] = document["mentions"]
    return gold


def is_right(mentions: List[Dict[str, int]], span: Any) -> bool:
    if not mentions:
        return span is None
    return span is not None and any(consider_overlap_as_match(m["start_char"], m["end_char"], span[0], span[1])
                                    for m in mentions)


def phase_of(round_no: int) -> str:
    return next(name for name, (low, high) in PHASES.items() if low <= round_no <= high)


def unit_rounds(unit: Dict[str, Any], gold: Dict[str, Dict[str, Any]], tag: str) -> List[Dict[str, Any]]:
    attribute = unit["attribute"]
    spans = unit["guess_spans"][attribute]
    documents = [document_id(name) for name in spans["docs"]]
    counts = {row["round"]: row for row in unit["per_round"][attribute]}
    answers = {answer[0]: answer for answer in unit["answers"].get(attribute, [])}
    picks = {pick["round"]: pick for pick in unit.get("pick_log") or []}

    rows, before = [], None
    for round_no, cells in spans["rounds"]:
        now = {d: is_right(gold[d].get(attribute, []), span) for d, span in zip(documents, cells)}
        count = counts[round_no]
        expected = count["num_should_be_filled_is_correct"] + count["num_should_be_empty_is_empty"]
        assert sum(now.values()) == expected, (unit["job"], round_no, sum(now.values()), expected)
        answer = answers.get(round_no)
        if before is not None and answer is not None and answer[3]:
            asked = {document_id(name) for name in answer[3].split(",")}  # several for a multi-match
            others = [d for d in documents if d not in asked]
            fixed = sum(now[d] and not before[d] for d in asked)
            kind = "row fixed" if fixed else ("row was right" if all(before[d] for d in asked) else "row still wrong")
            gained = sum(now[d] and not before[d] for d in others)
            lost = sum(before[d] and not now[d] for d in others)
            pick = picks.get(round_no, {})
            rows.append({"grid": tag, "variant": unit["variant"], "k": unit["k"], "seed": unit["seed_idx"],
                         "attribute": attribute, "round": round_no, "phase": phase_of(round_no), "answerer": answer[1],
                         "message": answer[2], "kind": kind, "rows asked": len(asked), "rows fixed": fixed,
                         "other rows gained": gained, "other rows lost": lost,
                         "other rows net": gained - lost, "changed nothing else": gained + lost == 0,
                         "position": pick.get("position"), "cell filled": pick.get("cell_filled")})
        before = now
    return rows


def summarize(rounds: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    table = rounds.groupby(keys, observed=True).agg(
        rounds=("kind", "size"), rows_asked=("rows asked", "mean"), rows_fixed=("rows fixed", "mean"),
        row_was_right=("kind", lambda k: (k == "row was right").mean()),
        row_fixed=("kind", lambda k: (k == "row fixed").mean()), gained=("other rows gained", "mean"),
        lost=("other rows lost", "mean"), net=("other rows net", "mean"),
        nothing_else=("changed nothing else", "mean"))
    return table.round(3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="+")
    parser.add_argument("--name", default="round_effect")
    parser.add_argument("--out", type=Path, default=ROOT / "info" / "analysis")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for tag in args.tags:
        config = json.loads((HYBRID / tag / "config.json").read_text(encoding="utf-8"))
        gold = load_gold(config.get("dataset", "aviation"))
        for unit in load(tag):
            if unit["variant"] != "A_no_feedback" and unit["guess_spans"].get(unit["attribute"]):
                rows.extend(unit_rounds(unit, gold, tag))
    rounds = pd.DataFrame(rows)
    rounds["phase"] = pd.Categorical(rounds["phase"], list(PHASES), ordered=True)

    keys = ["grid", "variant", "k", "answerer"]
    overall = summarize(rounds, keys)
    by_phase = summarize(rounds, keys + ["phase"])
    by_kind = summarize(rounds, keys + ["phase", "kind"])
    by_message = summarize(rounds, keys + ["message"])
    by_attribute = summarize(rounds, keys + ["attribute"])

    args.out.mkdir(parents=True, exist_ok=True)
    rounds.to_csv(args.out / f"{args.name}_rounds.csv", index=False)
    with open(args.out / f"{args.name}.md", "w", encoding="utf-8") as fh:
        fh.write(f"Generated by scripts/analyze_round_effect.py from {', '.join(args.tags)}. Per answer from round 2 on "
                 "(WannaDB's match rule): row_was_right and row_fixed are shares of the rounds, gained and lost count "
                 "the OTHER rows that got right or wrong in the same round, net is gained minus lost, nothing_else is "
                 "the share of the rounds that changed no other row.\n")
        for title, table in (("Overall", overall), ("By phase", by_phase), ("By phase and kind", by_kind),
                             ("By answer type", by_message), ("By column", by_attribute)):
            fh.write(f"\n## {title}\n\n{md(table)}\n")
    print(md(overall))
    print(md(by_phase))
    return 0


if __name__ == "__main__":
    sys.exit(main())
