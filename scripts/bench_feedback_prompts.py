
import argparse
import json
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import run_grid as rg

from wannadb.data.data import Attribute, DocumentBase
from llm_feedback import DEFAULT_BASE_URL
from llm_humanlike_feedback import HumanLikeLLMCallback

DEFAULT_SOURCE = rg.ROOT / "info" / "llm_logs" / "hybrid40_20260914"


def load_cases(source: Path, per_attribute: int, seed: int):
    by_attribute = defaultdict(list)
    for path in sorted(source.glob("handoff_k00_*.jsonl")):  # LLM-only units: guesses from all 40 rounds
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            pick = rec.get("rejection_pick")
            if pick:
                by_attribute[rec["attribute"]].append({"attribute": rec["attribute"], "round": rec["num_feedback"],
                                                       "source": path.name, **pick})
    rng = random.Random(seed)
    cases = []
    for attribute in sorted(by_attribute):
        cases += rng.sample(by_attribute[attribute], min(per_attribute, len(by_attribute[attribute])))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--modes", nargs="+", default=["humanlike-nuggets", "humanlike-custom"],
                        choices=["humanlike-nuggets", "humanlike-custom", "humanlike-direct"])
    parser.add_argument("--include-description", action="store_true",
                        help="add the attribute description of the dataset to the prompts")
    parser.add_argument("--per-attribute", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=6, help="concurrent requests to the shared server")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=None, help="jsonl with one line per answered case")
    args = parser.parse_args()

    base = DocumentBase.from_bson(rg.BSON_PATH.read_bytes())
    ground_truth = rg.load_aviation_module().load_dataset()
    documents = {d.name: d for d in base.documents}
    descriptions = rg.attribute_descriptions()
    cases = load_cases(args.source, args.per_attribute, args.seed)
    out_path = args.out or rg.ROOT / "experiments" / "results" / "bench" / f"bench_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="bench_"))

    def run(job):
        mode, i, case = job
        document = documents[case["document"]]
        nugget = next(n for n in document.nuggets
                      if n.start_char == case["start_char"] and n.end_char == case["end_char"])
        log = tmp_dir / f"{mode}_{i}.jsonl"
        callback = HumanLikeLLMCallback(ground_truth, {case["attribute"]: case["attribute"]}, base_url=args.base_url,
                                        model=args.model, log_path=str(log), max_total_tokens=None,
                                        allow_custom_text=mode in ("humanlike-custom", "humanlike-direct"),
                                        check_in_full_text=mode == "humanlike-direct",
                                        include_description=args.include_description,
                                        attribute_descriptions=descriptions)
        callback._give_feedback({"nuggets": (nugget,), "attribute": Attribute(case["attribute"]),
                                 "num-feedback": case["round"]})
        rec = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
        rec["case"] = case
        return mode, rec

    jobs = [(mode, i, case) for mode in args.modes for i, case in enumerate(cases)]
    print(f"{len(cases)} cases x {len(args.modes)} modes, {args.parallel} parallel requests")
    results = defaultdict(list)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool, open(out_path, "w", encoding="utf-8") as out:
        for n, (mode, rec) in enumerate(pool.map(run, jobs), 1):
            results[mode].append(rec)
            label = mode + ("+desc" if args.include_description else "")
            out.write(json.dumps({"bench_mode": label, "bench_model": args.model, **rec}, ensure_ascii=False) + "\n")
            if n % 20 == 0:
                print(f"  {n}/{len(jobs)} done, {time.time() - t0:.0f}s")

    for mode, recs in results.items():
        right = sum(r["agrees_with_gold"] is True for r in recs)
        print(f"\n=== {mode}: {right}/{len(recs)} answers agree with gold ({right / len(recs):.0%})")
        by_case = defaultdict(Counter)
        for r in recs:
            by_case[r["oracle_answer"]][(r["result_message"], r["agrees_with_gold"])] += 1
        for case, counts in sorted(by_case.items()):
            total = sum(counts.values())
            ok = sum(v for (msg, agrees), v in counts.items() if agrees)
            print(f"  human would answer {case:12s}: {ok}/{total} right   {dict(counts)}")
        by_attr = defaultdict(lambda: [0, 0])
        for r in recs:
            by_attr[r["attribute"]][0] += r["agrees_with_gold"] is True
            by_attr[r["attribute"]][1] += 1
        print("  per attribute:", {a: f"{ok}/{n}" for a, (ok, n) in sorted(by_attr.items())})
        tokens = [(r["usage"] or {}).get("total_tokens", 0) for r in recs]
        secs = [r["duration_seconds"] for r in recs]
        print(f"  tokens per decision {sum(tokens) / len(recs):.0f}, seconds per decision {sum(secs) / len(recs):.1f}, "
              f"edit step used in {sum(r['step_b'] is not None for r in recs)} cases")
    print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
