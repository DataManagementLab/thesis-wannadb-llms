
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# the same environment as the grid workers (CPU, 2 threads): already the embedding of the attribute name differs
# slightly on a GPU or with another number of threads, and then the shortlists differ
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = os.environ.get("REPLAY_THREADS", "2")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parent))
sys.path.insert(0, str(SCRIPTS_DIR.parent / "experiments"))
sys.path.insert(0, str(SCRIPTS_DIR))
import run_grid as rg
import run_hybrid_grid as g

from wannadb.configuration import Pipeline
from wannadb.data.data import DocumentBase
from wannadb.interaction import BaseInteractionCallback
from wannadb.resources import ResourceManager
from wannadb.statistics import Statistics
from experiment_runner import ExperimentRunner
from automatic_feedback import AutomaticCustomMatchesRandomRankingBasedMatchingFeedback
from hybrid_feedback import HybridInteractionCallback, human_round_numbers
from util import get_document_by_name

LLM_LOGS = SCRIPTS_DIR.parent / "info" / "llm_logs"


class ReplayMismatch(Exception):
    pass


class ReplayLLM(BaseInteractionCallback):
    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._records = list(records)
        self.last_round = None

    def _call(self, pipeline_element_identifier: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if "do-attribute-request" in data.keys():
            return {"do-attribute": True}
        if not self._records:
            raise ReplayMismatch(f"no logged answer left for round {data['num-feedback']}")
        rec = self._records.pop(0)
        pick = random.choice(list(data["nuggets"]))
        logged = rec["pick"]
        if (rec.get("num_feedback") != data["num-feedback"] or pick.document.name != logged["document"]
                or (pick.start_char, pick.end_char) != (logged["start_char"], logged["end_char"])):
            raise ReplayMismatch(f"round {data['num-feedback']}: drew {pick.document.name} {pick.text!r}, "
                                 f"logged {logged['document']} {logged['text']!r}")
        self.last_round = rec
        span = rec["result_span"]
        if rec["result_message"] == "no-match-in-document":
            return {"message": "no-match-in-document", "nugget": pick, "not-a-match": pick}
        if rec["result_message"] == "custom-match":
            return {"message": "custom-match", "document": pick.document, "start": span["start_char"],
                    "end": span["end_char"]}
        nugget = next(n for n in pick.document.nuggets
                      if (n.start_char, n.end_char) == (span["start_char"], span["end_char"]))
        return {"message": "is-match", "nugget": nugget, "not-a-match": None if nugget is pick else pick}

    def add_confirmed_example(self, attribute_name: str, value: str) -> None:
        pass  # the logged answers already saw whatever examples the run showed


def replay_unit(tag: str, unit: Dict[str, Any], cfg: Dict[str, Any], bson: bytes, gt: List[Dict[str, Any]],
                rm: ResourceManager) -> Dict[str, Any]:
    attr, variant, k = unit["attribute"], unit["variant"], unit["k"]
    budget = unit["budget"]
    # on c02 the logs are next to the checkpoints, locally they were moved to info/llm_logs
    log_path = g.HYBRID_ROOT / tag / "llm_logs" / f"{unit['job']}.jsonl"
    if not log_path.exists():
        log_path = LLM_LOGS / tag / f"{unit['job']}.jsonl"
    records = ([json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
               if log_path.exists() else [])
    holder: Dict[str, Any] = {}

    def oracle_factory(documents, mapping):
        human = AutomaticCustomMatchesRandomRankingBasedMatchingFeedback(documents, mapping)
        holder["llm"] = ReplayLLM(records) if records else None
        return HybridInteractionCallback(
            human, holder["llm"], human_rounds=k,
            human_round_numbers=human_round_numbers(variant, k, cfg["max_rounds"], seed=unit["seed_value"]))

    stats = Statistics(do_collect=True)
    runner = ExperimentRunner(DocumentBase.from_bson(bson), gt, user_attribute_name2attribute_name={attr: attr},
                              preprocessing_pipeline=Pipeline([]),
                              matching_pipeline_settings={"max_num_feedback": budget, "store_best_guesses": True},
                              resource_manager=rm, statistics=stats)
    runner.random_seeds = [unit["seed_value"]]
    runner.fill(raw_attributes=[attr], num_runs=1, feedback_oracle=oracle_factory)

    def gold_of(name):
        return get_document_by_name(gt, name)

    docs_order = [d.name for d in runner.document_base.documents]
    rows, span_rounds = [], []
    for round_no, guesses in (g.best_guess_history(stats, attr) if budget > 0 else []):
        rows.append({"round": round_no, **g.count_snapshot(guesses, attr, gold_of)})
        by_doc = {name: (None if guess is None else [guess[2], guess[3]]) for name, guess in guesses}
        span_rounds.append([round_no, [by_doc.get(name) for name in docs_order]])
    final = [None if not d.attribute_mappings.get(attr) else
             [d.attribute_mappings[attr][0].start_char, d.attribute_mappings[attr][0].end_char]
             for d in runner.document_base.documents]
    stored = {r["round"]: r for r in unit["per_round"].get(attr, [])}
    identical = (len(rows) == len(stored) and all(all(r[key] == stored[r["round"]][key] for key in rg.COUNT_KEYS)
                                                    for r in rows))
    if budget == 0:
        identical = all(unit["final_scores"][0][key] == v for key, v in
                        rg.score_state(runner.document_base, gt, [attr])[0].items() if key in rg.COUNT_KEYS)
    return {"job": unit["job"], "variant": variant, "k": k, "seed_idx": unit["seed_idx"], "attribute": attr,
            "seed_value": unit["seed_value"], "identical_to_checkpoint": identical,
            "guess_spans": {attr: {"docs": docs_order, "rounds": span_rounds, "final": final}}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_tag")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--k", type=int, nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--attributes", nargs="*", default=None)
    args = parser.parse_args()

    import torch
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

    root = g.HYBRID_ROOT / args.run_tag
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    out = root / "replayed"
    out.mkdir(exist_ok=True)
    bson = g.document_base_bytes(cfg)
    gt = g.dataset_module(cfg.get("dataset", "aviation")).load_dataset()

    units = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "checkpoints").glob("*.json"))]
    units = [u for u in units if (args.variants is None or u["variant"] in args.variants)
             and (args.k is None or u["k"] in args.k) and (args.seeds is None or u["seed_idx"] in args.seeds)
             and (args.attributes is None or u["attribute"] in args.attributes)
             and not (out / f"{u['job']}.json").exists()]
    print(f"{args.run_tag}: replaying {len(units)} units", flush=True)
    with ResourceManager() as rm:
        for i, unit in enumerate(units):
            t0 = time.time()
            try:
                rec = replay_unit(args.run_tag, unit, cfg, bson, gt, rm)
            except ReplayMismatch as e:
                print(f"[{i + 1}/{len(units)}] {unit['job']}: MISMATCH {e}", flush=True)
                (out / f"{unit['job']}.mismatch.txt").write_text(str(e), encoding="utf-8")
                continue
            (out / f"{unit['job']}.json").write_text(json.dumps(rec), encoding="utf-8")
            print(f"[{i + 1}/{len(units)}] {unit['job']}: identical={rec['identical_to_checkpoint']} "
                  f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
