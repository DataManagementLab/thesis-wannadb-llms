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

METHODS = {
    "base": {},
    "vote3": {"vote_samples": 3},
    "vote5": {"vote_samples": 5},
    "check1": {"self_check_rounds": 1},
    "check2": {"self_check_rounds": 2},
    "check1c": {"self_check_rounds": 1, "cautious_review": True},
    "check2c": {"self_check_rounds": 2, "cautious_review": True},
}


def load_cases(log_dir: Path, right_cases: int, seed: int):
    wrong, right = [], []
    for path in sorted(log_dir.glob("handoff_k00_*.jsonl")):  # LLM-only units
        accepted = []
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            case = {"attribute": rec["attribute"], "round": rec["num_feedback"], "source": path.name,
                    "pick": rec["pick"], "accepted_before": list(accepted), "logged_agrees": rec["agrees_with_gold"],
                    "include_description": rec.get("include_description", False)}
            (right if rec["agrees_with_gold"] else wrong).append(case)
            if rec["result_message"] != "no-match-in-document":
                accepted.append(rec["result_span"]["text"])
    return wrong + random.Random(seed).sample(right, min(right_cases, len(right))), len(right)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--methods", nargs="+", default=["base", "vote3", "check2"], choices=list(METHODS))
    parser.add_argument("--right-cases", type=int, default=153)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    base = DocumentBase.from_bson(rg.BSON_PATH.read_bytes())
    ground_truth = rg.load_aviation_module().load_dataset()
    documents = {d.name: d for d in base.documents}
    descriptions = rg.attribute_descriptions()
    cases, right_in_run = load_cases(args.log_dir, args.right_cases, args.seed)
    out_path = args.out or rg.ROOT / "experiments" / "results" / "bench" / f"hard_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="hard_"))

    def run(job):
        method, i, case = job
        pick = case["pick"]
        nugget = next(n for n in documents[pick["document"]].nuggets
                      if n.start_char == pick["start_char"] and n.end_char == pick["end_char"])
        log = tmp_dir / f"{method}_{i}.jsonl"
        callback = HumanLikeLLMCallback(ground_truth, {case["attribute"]: case["attribute"]}, base_url=args.base_url,
                                        model=args.model, log_path=str(log), max_total_tokens=None,
                                        allow_custom_text=True, check_in_full_text=True,
                                        include_description=case["include_description"],
                                        attribute_descriptions=descriptions, **METHODS[method])
        callback.seed_accepted_values(case["attribute"], case["accepted_before"])
        callback._give_feedback({"nuggets": (nugget,), "attribute": Attribute(case["attribute"]),
                                 "num-feedback": case["round"]})
        rec = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
        return method, case, rec

    jobs = [(m, i, c) for m in args.methods for i, c in enumerate(cases)]
    n_wrong = sum(not c["logged_agrees"] for c in cases)
    print(f"{len(cases)} cases ({n_wrong} answered wrong in the run), methods {args.methods}, {args.parallel} parallel")
    results = defaultdict(list)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool, open(out_path, "w", encoding="utf-8") as out:
        for n, (method, case, rec) in enumerate(pool.map(run, jobs), 1):
            results[method].append((case, rec))
            out.write(json.dumps({"method": method, "bench_model": args.model, "case": case, **rec},
                                 ensure_ascii=False) + "\n")
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)} done, {time.time() - t0:.0f}s")

    for method in args.methods:
        pairs = results[method]
        fixed = sum((not c["logged_agrees"]) and r["agrees_with_gold"] for c, r in pairs)
        broken = sum(c["logged_agrees"] and not r["agrees_with_gold"] for c, r in pairs)
        still = sum((not c["logged_agrees"]) and not r["agrees_with_gold"] for c, r in pairs)
        calls = sum(r["llm_calls"] for _, r in pairs) / len(pairs)
        tokens = sum((r["usage"] or {}).get("total_tokens", 0) for _, r in pairs) / len(pairs)
        print(f"\n=== {method}: of {n_wrong} wrong answers {fixed} fixed, {still} still wrong; "
              f"of {len(pairs) - n_wrong} right answers {broken} broken; net {fixed - broken:+d}")
        sampled_right = len(pairs) - n_wrong
        print(f"  over the whole run ({right_in_run} right answers): about {fixed - broken * right_in_run / sampled_right:+.0f}"
              f" answers net; {calls:.2f} LLM calls and {tokens:.0f} tokens per decision")
        by_attr = Counter(c["attribute"] for c, r in pairs if (not c["logged_agrees"]) and r["agrees_with_gold"])
        print(f"  fixed by attribute: {dict(by_attr)}")
    print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
