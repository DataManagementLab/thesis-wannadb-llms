import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

BASELINE_ORDER = ["A_no_feedback", "B_gold_oracle", "C_llm"]
BASELINE_LABEL = {
    "A_no_feedback": "A - kein Feedback",
    "B_gold_oracle": "B - Gold-Orakel",
    "C_llm": "C - LLM (DeepSeek)",
}


def per_attribute_failure_rates(record):
    log_file = record.get("llm_log_file")
    if not log_file or not Path(log_file).exists():
        return {}
    with open(log_file, encoding="utf-8") as f:
        lines = f.readlines()
    expected = len(record["config"]["attributes"]) * record["config"]["max_num_feedback"]
    # lines from a killed earlier attempt come first
    if len(lines) > expected:
        lines = lines[-expected:]
    counts = {}
    for line in lines:
        rec = json.loads(line)
        attr = rec["attribute"]
        bad = (rec["raw_response"] == "") or (rec.get("stage2") and rec["stage2"]["raw_response"] == "")
        total, bad_n = counts.get(attr, (0, 0))
        counts[attr] = (total + 1, bad_n + (1 if bad else 0))
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--save", action="store_true", help="also write a parquet summary")
    args = parser.parse_args()

    files = sorted(CHECKPOINT_DIR.glob(f"{args.run_tag}_*_seed*.json"))
    if not files:
        print(f"No checkpoints found for run_tag={args.run_tag} under {CHECKPOINT_DIR}")
        return 1

    records = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    expected_seeds = max(r["seed_idx"] for r in records) + 1
    print(f"{len(records)} checkpoints found for run_tag={args.run_tag}")
    for baseline in BASELINE_ORDER:
        have = sorted(r["seed_idx"] for r in records if r["baseline"] == baseline)
        missing = [s for s in range(expected_seeds) if s not in have]
        status = "vollstaendig" if not missing else f"FEHLT: seeds {missing}"
        print(f"  {BASELINE_LABEL.get(baseline, baseline):20} seeds present: {have}  ({status})")
    print()

    attributes = sorted({row["attribute"] for r in records for row in r["scores"]})

    def mean_f1(baseline: str, attribute: str):
        vals = [row["f1_score"] for r in records if r["baseline"] == baseline
                for row in r["scores"] if row["attribute"] == attribute]
        return sum(vals) / len(vals) if vals else None

    header = f"{'Attribut':30}" + "".join(f"{BASELINE_LABEL.get(b, b):>22}" for b in BASELINE_ORDER)
    print(header)
    print("-" * len(header))
    for attr in attributes:
        row = f"{attr:30}"
        for baseline in BASELINE_ORDER:
            v = mean_f1(baseline, attr)
            row += f"{v:>22.3f}" if v is not None else f"{'--':>22}"
        print(row)

    overall = {}
    for baseline in BASELINE_ORDER:
        vals = [v for a in attributes if (v := mean_f1(baseline, a)) is not None]
        overall[baseline] = sum(vals) / len(vals) if vals else None
    print("-" * len(header))
    row = f"{'MITTEL (alle Attribute)':30}"
    for baseline in BASELINE_ORDER:
        v = overall[baseline]
        row += f"{v:>22.3f}" if v is not None else f"{'--':>22}"
    print(row)
    print()

    print(f"{'Baseline':22}{'t_wannadb_matching':>20}{'t_llm_inference':>18}{'t_total':>12}")
    for baseline in BASELINE_ORDER:
        rs = [r for r in records if r["baseline"] == baseline]
        if not rs:
            continue
        tm = sum(r["t_wannadb_matching_seconds"] for r in rs) / len(rs)
        tl = sum(r["t_llm_inference_seconds"] for r in rs) / len(rs)
        tt = sum(r["t_total_seconds"] for r in rs) / len(rs)
        print(f"{BASELINE_LABEL.get(baseline, baseline):22}{tm:>18.1f}s{tl:>16.1f}s{tt:>10.1f}s")
    print()

    llm_records = [r for r in records if r["baseline"] == "C_llm"]
    if llm_records:
        agg = {}
        for r in llm_records:
            for attr, (total, bad_n) in per_attribute_failure_rates(r).items():
                t, b = agg.get(attr, (0, 0))
                agg[attr] = (t + total, b + bad_n)

        print("Leere/unbrauchbare LLM-Antworten pro Attribut (ueber alle vorhandenen Seeds):")
        if not agg:
            print("Keine LLM-Logs gefunden (z. B. "
                  f"{sorted({r.get('llm_log_file') for r in llm_records})[:1]}), Rate nicht berechenbar.")
        flagged = False
        for attr in sorted(agg):
            total, bad_n = agg[attr]
            rate = bad_n / total if total else 0.0
            marker = "  <-- ueber 5%, mit Vorsicht lesen" if rate > 0.05 else ""
            if marker:
                flagged = True
            print(f"    {attr:30} {bad_n:>3}/{total:<4} ({rate:>5.1%}){marker}")
        if agg and not flagged:
            print("Alle Attribute unter 5%, die F1-Werte oben sind durchgehend interpretierbar.")
        elif flagged:
            print("Die markierten Attribute mit Vorsicht lesen, der Rest ist regulaer nutzbar.")

    if args.save:
        import pandas as pd
        rows = []
        for r in records:
            for row in r["scores"]:
                rows.append({
                    "run_tag": r["run_tag"], "baseline": r["baseline"], "seed_idx": r["seed_idx"],
                    "attribute": row["attribute"], "precision": row["precision"],
                    "recall": row["recall"], "f1_score": row["f1_score"],
                    "t_wannadb_matching_seconds": r["t_wannadb_matching_seconds"],
                    "t_llm_inference_seconds": r["t_llm_inference_seconds"],
                })
        df = pd.DataFrame(rows)
        out = RESULTS_DIR / f"{args.run_tag}_summary.parquet"
        df.to_parquet(out)
        print(f"\nSaved: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
