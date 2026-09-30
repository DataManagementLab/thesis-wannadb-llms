import argparse
import importlib.util
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("run_grid")
logger.setLevel(logging.INFO)

from wannadb.configuration import Pipeline
from wannadb.data.data import DocumentBase
from wannadb.resources import ResourceManager
from wannadb.statistics import Statistics
from experiment_runner import ExperimentRunner
from automatic_feedback import AutomaticCustomMatchesRandomRankingBasedMatchingFeedback
from llm_feedback import LLMInteractionCallback
from util import consider_overlap_as_match, get_document_by_name

BSON_PATH = ROOT / "bson_files" / "aviation-preprocessed-with-docsent-fixed.bson"
RESULTS_DIR = ROOT / "experiments" / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
LOG_DIR = ROOT / "experiments" / "llm_feedback_logs"

COUNT_KEYS = [
    "num_should_be_filled_is_correct",
    "num_should_be_filled_is_incorrect",
    "num_should_be_filled_is_empty",
    "num_should_be_empty_is_empty",
    "num_should_be_empty_is_full",
]


def load_aviation_module():
    spec = importlib.util.spec_from_file_location(
        "aviation_dataset", str(ROOT / "wannadb_datsets" / "datasets" / "aviation" / "aviation.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_state(document_base: DocumentBase, ground_truth_docs: List[Dict[str, Any]],
                 attribute_names: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for attr in attribute_names:
        r = {k: 0 for k in COUNT_KEYS}
        for document in document_base.documents:
            gt_doc = get_document_by_name(ground_truth_docs, document.name)
            gold = gt_doc["mentions"].get(attr, []) if gt_doc else []
            guess = document.attribute_mappings.get(attr, [])

            if gold:
                if guess:
                    g = guess[0]
                    hit = any(consider_overlap_as_match(m["start_char"], m["end_char"],
                                                        g.start_char, g.end_char) for m in gold)
                    r["num_should_be_filled_is_correct" if hit else "num_should_be_filled_is_incorrect"] += 1
                else:
                    r["num_should_be_filled_is_empty"] += 1
            else:
                r["num_should_be_empty_is_full" if guess else "num_should_be_empty_is_empty"] += 1

        total_filled = r["num_should_be_filled_is_correct"] + r["num_should_be_filled_is_incorrect"] + r["num_should_be_filled_is_empty"]
        total_pred_filled = r["num_should_be_filled_is_correct"] + r["num_should_be_filled_is_incorrect"] + r["num_should_be_empty_is_full"]
        recall = 1.0 if total_filled == 0 else r["num_should_be_filled_is_correct"] / total_filled
        precision = 1.0 if total_pred_filled == 0 else r["num_should_be_filled_is_correct"] / total_pred_filled
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

        rows.append({"attribute": attr, "precision": precision, "recall": recall, "f1_score": f1, **r})
    return rows


def checkpoint_path(run_tag: str, baseline_key: str, seed_idx: int) -> Path:
    return CHECKPOINT_DIR / f"{run_tag}_{baseline_key}_seed{seed_idx}.json"


def sum_llm_duration(log_path: Optional[str], expected_rounds: Optional[int] = None) -> float:
    if not log_path or not Path(log_path).exists():
        return 0.0
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    # a killed earlier attempt leaves its lines at the start of the log
    if expected_rounds is not None and len(lines) > expected_rounds:
        logger.warning(f"{log_path}: {len(lines)} lines, expected {expected_rounds}, "
                       f"using only the last {expected_rounds}")
        lines = lines[-expected_rounds:]
    return sum(json.loads(line)["duration_seconds"] for line in lines)


def run_one(
        run_tag: str,
        baseline_key: str,
        seed_idx: int,
        seed_value: int,
        bson_bytes: bytes,
        ground_truth: List[Dict[str, Any]],
        attributes: List[str],
        max_num_feedback: int,
        feedback_oracle,
        resource_manager: ResourceManager,
        llm_log_path: Optional[str],
        config: Dict[str, Any],
) -> Dict[str, Any]:
    ckpt = checkpoint_path(run_tag, baseline_key, seed_idx)
    if ckpt.exists():
        logger.info(f"SKIP  {baseline_key} seed{seed_idx} (checkpoint already exists)")
        return json.loads(ckpt.read_text(encoding="utf-8"))

    logger.info(f"START {baseline_key} seed{seed_idx} (seed={seed_value}, "
               f"{len(attributes)} attributes, max_num_feedback={max_num_feedback})")

    document_base = DocumentBase.from_bson(bson_bytes)
    mapping = {a: a for a in attributes}
    stats = Statistics(do_collect=True)
    runner = ExperimentRunner(
        document_base, ground_truth,
        user_attribute_name2attribute_name=mapping,
        preprocessing_pipeline=Pipeline([]),
        matching_pipeline_settings={"max_num_feedback": max_num_feedback, "store_best_guesses": True},
        resource_manager=resource_manager, statistics=stats,
    )

    runner.random_seeds = [seed_value]

    t0 = time.perf_counter()
    runner.fill(raw_attributes=attributes, num_runs=1, feedback_oracle=feedback_oracle)
    t_total = time.perf_counter() - t0

    t_llm = sum_llm_duration(llm_log_path, expected_rounds=len(attributes) * max_num_feedback)
    scores = score_state(runner.document_base, ground_truth, attributes)

    record = {
        "run_tag": run_tag,
        "baseline": baseline_key,
        "seed_idx": seed_idx,
        "seed_value": seed_value,
        "config": config,
        "t_total_seconds": t_total,
        "t_llm_inference_seconds": t_llm,
        "t_wannadb_matching_seconds": t_total - t_llm,
        "scores": scores,
        "llm_log_file": llm_log_path,
        "llm_failure_counts": feedback_oracle.failure_counts if hasattr(feedback_oracle, "failure_counts") else None,
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        f"DONE  {baseline_key} seed{seed_idx}: t_total={t_total:.1f}s "
        f"t_llm={t_llm:.1f}s t_matching={t_total - t_llm:.1f}s "
        f"mean_f1={sum(r['f1_score'] for r in scores) / len(scores):.3f}"
    )
    return record


class LLMOracleFactory:
    """ExperimentRunner creates the oracle itself, this keeps a handle on it for the failure counts."""

    def __init__(self, log_path: str, seed_value: int, config: Dict[str, Any]):
        self.log_path = log_path
        self._seed_value = seed_value
        self._config = config
        self.instance: Optional[LLMInteractionCallback] = None

    def __call__(self, documents, mapping):
        self.instance = LLMInteractionCallback(
            documents, mapping,
            log_path=self.log_path,
            include_description=self._config["include_description"],
            max_response_tokens=self._config["max_response_tokens"],
            max_total_tokens=self._config["max_total_tokens"],
            timeout_seconds=self._config["timeout_seconds"],
            random_seed=self._seed_value,
        )
        return self.instance

    @property
    def failure_counts(self):
        return self.instance.failure_counts if self.instance is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-tag", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--max-num-feedback", type=int, default=25)
    parser.add_argument("--attributes", nargs="*", default=None, help="default: all 12 aviation attributes")
    parser.add_argument("--include-description", action="store_true")
    parser.add_argument("--max-response-tokens", type=int, default=16384)
    parser.add_argument("--max-total-tokens", type=int, default=20_000_000,
                        help="safety ceiling per (seed) LLM callback instance, not a real budget")
    parser.add_argument("--timeout-seconds", type=float, default=300.0,
                        help="per request; a long reasoning answer can take more than 100 s")
    parser.add_argument("--skip-baselines", nargs="*", default=[],
                        choices=["A_no_feedback", "B_gold_oracle", "C_llm"])
    args = parser.parse_args()

    dataset = load_aviation_module()
    attributes = args.attributes or list(dataset.ATTRIBUTES)

    config = {
        "attributes": attributes,
        "max_num_feedback": args.max_num_feedback,
        "num_seeds": args.num_seeds,
        "include_description": args.include_description,
        "max_response_tokens": args.max_response_tokens,
        "max_total_tokens": args.max_total_tokens,
        "timeout_seconds": args.timeout_seconds,
    }

    logger.info(f"run_tag={args.run_tag}")
    logger.info(f"config={json.dumps(config, ensure_ascii=False)}")

    if not BSON_PATH.exists():
        logger.error(f"Missing {BSON_PATH}. Create it with the cell 'Dokument-Satz-Embeddings einmalig "
                     "vorberechnen' in experiments/llm_feedback_baseline.ipynb.")
        return 1

    with open(BSON_PATH, "rb") as f:
        bson_bytes = f.read()
    ground_truth = dataset.load_dataset()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = RESULTS_DIR / f"{args.run_tag}_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    plans = [
        ("A_no_feedback", 0),
        ("B_gold_oracle", args.max_num_feedback),
        ("C_llm", args.max_num_feedback),
    ]

    failures = 0
    with ResourceManager() as rm:
        seed_values = None
        for baseline_key, max_fb in plans:
            if baseline_key in args.skip_baselines:
                logger.info(f"SKIP baseline {baseline_key} (--skip-baselines)")
                continue

            for seed_idx in range(args.num_seeds):
                if seed_values is None:
                    # the seeds ExperimentRunner would choose itself
                    probe = ExperimentRunner(DocumentBase.from_bson(bson_bytes), ground_truth,
                                             resource_manager=rm, statistics=Statistics(do_collect=False),
                                             preprocessing_pipeline=Pipeline([]))
                    seed_values = probe.random_seeds[:args.num_seeds]
                    del probe
                    logger.info(f"seeds={seed_values}")

                seed_value = seed_values[seed_idx]

                try:
                    if baseline_key == "C_llm":
                        log_path = str(LOG_DIR / f"{args.run_tag}_{baseline_key}_seed{seed_idx}.jsonl")
                        oracle = LLMOracleFactory(log_path, seed_value, config)
                        llm_log_path = log_path
                    else:
                        oracle = AutomaticCustomMatchesRandomRankingBasedMatchingFeedback
                        llm_log_path = None

                    run_one(args.run_tag, baseline_key, seed_idx, seed_value, bson_bytes,
                           ground_truth, attributes, max_fb, oracle, rm, llm_log_path, config)

                except Exception:
                    failures += 1
                    logger.error(f"FAILED {baseline_key} seed{seed_idx}, no checkpoint written, "
                                 "rerun the script to retry")
                    logger.error(traceback.format_exc())
                    continue

    logger.info(f"GRID DONE. run_tag={args.run_tag}  failures={failures}")
    logger.info(f"Summarize with: python scripts/summarize_grid.py --run-tag {args.run_tag}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
