import argparse
import hashlib
import importlib.util
import json
import logging
import os
import random
import re
import socket
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import run_grid as rg
import wandb_log

from wannadb.configuration import Pipeline
from wannadb.data.data import DocumentBase
from wannadb.resources import ResourceManager
from wannadb.statistics import Statistics
from experiment_runner import ExperimentRunner
from automatic_feedback import AutomaticCustomMatchesRandomRankingBasedMatchingFeedback
from llm_feedback import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMInteractionCallback
from llm_humanlike_feedback import PROMPT_VERSION, HumanLikeLLMCallback
from hybrid_feedback import ESCALATIONS, SCHEDULES, HybridInteractionCallback, human_round_numbers
from multi_feedback import MULTI_PROMPT_VERSIONS, GoldMultiFeedback, MultiFeedbackLLMCallback
from stored_feedback import (READ_ORDERS, RULE_VARIANTS, StoredAnswerFeedback, gold_answers, load_llm_answers,
                             read_documents, stored_setting)
from util import consider_overlap_as_match, get_document_by_name

from openai import (APIConnectionError, APITimeoutError, AuthenticationError, InternalServerError, NotFoundError,
                    OpenAI, PermissionDeniedError, RateLimitError)

logger = logging.getLogger("hybrid_grid")
logger.setLevel(logging.INFO)

HYBRID_ROOT = rg.RESULTS_DIR / "hybrid"
DATASETS_DIR = rg.ROOT / "wannadb_datsets" / "datasets"
POLL_SECONDS = 60

# an unreachable LLM server does not count as a failed attempt of the unit
INFRASTRUCTURE_ERRORS = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
INFRASTRUCTURE_WAIT_SECONDS = 600
# wrong model or key: every further unit would fail the same way, so the grid stops
FATAL_LLM_ERRORS = (NotFoundError, AuthenticationError, PermissionDeniedError)


class Dirs:
    def __init__(self, run_tag: str):
        self.root = HYBRID_ROOT / run_tag
        self.checkpoints = self.root / "checkpoints"
        self.claims = self.root / "claims"
        self.failures = self.root / "failures"
        self.llm_logs = self.root / "llm_logs"
        self.worker_logs = self.root / "worker_logs"
        self.config = self.root / "config.json"
        self.jobs = self.root / "jobs.json"
        self.done_marker = self.root / "GRID_DONE"
        self.stop = self.root / "STOP"
        self.coordinator_pid = self.root / "coordinator.pid"

    def create(self) -> None:
        for d in (self.checkpoints, self.claims, self.failures, self.llm_logs, self.worker_logs):
            d.mkdir(parents=True, exist_ok=True)

    def checkpoint(self, job_id: str) -> Path:
        return self.checkpoints / f"{job_id}.json"


def dataset_module(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_dataset", str(DATASETS_DIR / name / f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset_bson(name: str) -> Path:
    if name == "aviation":
        return rg.BSON_PATH
    fixed = rg.ROOT / "bson_files" / f"{name}-preprocessed-fixed.bson"
    return fixed if fixed.exists() else rg.ROOT / "bson_files" / f"{name}-preprocessed.bson"


def dataset_descriptions(name: str) -> Dict[str, str]:
    source = (DATASETS_DIR / name / f"{name}.py").read_text(encoding="utf-8")
    return dict(re.findall(r'^\s*"(\w+)",?\s*#\s*(.+?)\s*$', source, re.MULTILINE))


def document_base_bytes(cfg: Dict[str, Any]) -> bytes:
    raw = dataset_bson(cfg.get("dataset", "aviation")).read_bytes()
    if not cfg.get("num_docs"):
        return raw
    db = DocumentBase.from_bson(raw)
    names = sorted(d.name for d in db.documents)
    # the same shuffle for every size, so a smaller collection is part of every larger one
    random.Random(cfg.get("subset_seed", 0)).shuffle(names)
    keep = set(names[:cfg["num_docs"]])
    return DocumentBase([d for d in db.documents if d.name in keep], db.attributes).to_bson()


def coarse_to_fine(values: List[int]) -> List[int]:
    """Endpoints first, then halving the step (40, 20, 10, 5, ...), so a half finished grid
    already covers the whole range."""
    values = sorted(set(values))
    lo, hi = values[0], values[-1]
    order, seen = [], set()
    step = max(hi - lo, 1)
    while step >= 1:
        for v in values:
            if (v - lo) % step == 0 and v not in seen:
                seen.add(v)
                order.append(v)
        step //= 2
    order += [v for v in values if v not in seen]
    return order


def build_jobs(k_values: List[int], max_rounds: int, seed_indices: List[int], variants: List[str],
               attributes: List[str]) -> List[Dict[str, Any]]:
    jobs = [{"job": f"A_no_feedback_seed{s}_{a}", "variant": "A_no_feedback", "k": 0, "seed_idx": s, "attribute": a}
            for s in seed_indices for a in attributes]
    for k in coarse_to_fine(k_values):
        for variant in variants:
            if variant in SCHEDULES and variant != "handoff" and k in (0, max_rounds) and "handoff" in variants:
                continue  # identical to handoff there
            for s in seed_indices:
                for a in attributes:
                    jobs.append({"job": f"{variant}_k{k:02d}_seed{s}_{a}", "variant": variant, "k": k,
                                 "seed_idx": s, "attribute": a})
    return jobs


def llm_rounds_of(job: Dict[str, Any], cfg: Dict[str, Any]) -> int:
    if job["variant"] == "A_no_feedback":
        return 0
    return max(cfg["max_rounds"] - job["k"], 0)


def attribute_seed(cfg: Dict[str, Any], seed_idx: int, attribute: str) -> int:
    # otherwise the oracle would pick the same shortlist positions in every attribute
    return cfg["seed_values"][seed_idx] + cfg["attributes"].index(attribute)


def llm_connection(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # the key itself is never stored, only the name of the environment variable holding it
    api_key = os.environ.get(cfg["api_key_env"]) if cfg.get("api_key_env") else None
    # OpenRouter would otherwise spread the requests over several providers
    extra_body = {"provider": {"order": [cfg["provider"]], "allow_fallbacks": False}} if cfg.get("provider") else None
    return {"base_url": cfg.get("base_url", DEFAULT_BASE_URL), "model": cfg.get("model", DEFAULT_MODEL),
            "api_key": api_key, "extra_body": extra_body}


def prf(counts: Dict[str, int]) -> Dict[str, float]:
    total_filled = (counts["num_should_be_filled_is_correct"] + counts["num_should_be_filled_is_incorrect"]
                    + counts["num_should_be_filled_is_empty"])
    total_pred_filled = (counts["num_should_be_filled_is_correct"] + counts["num_should_be_filled_is_incorrect"]
                         + counts["num_should_be_empty_is_full"])
    recall = 1.0 if total_filled == 0 else counts["num_should_be_filled_is_correct"] / total_filled
    precision = 1.0 if total_pred_filled == 0 else counts["num_should_be_filled_is_correct"] / total_pred_filled
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1_score": f1}


def count_snapshot(guesses, attribute: str, gold_of) -> Dict[str, int]:
    counts = {key: 0 for key in rg.COUNT_KEYS}
    for doc_name, guess in guesses:
        gt_doc = gold_of(doc_name)
        gold = gt_doc["mentions"].get(attribute, []) if gt_doc else []
        if gold:
            if guess is None:
                counts["num_should_be_filled_is_empty"] += 1
            else:
                _, _, start, end, _ = guess
                hit = any(consider_overlap_as_match(m["start_char"], m["end_char"], start, end) for m in gold)
                counts["num_should_be_filled_is_correct" if hit else "num_should_be_filled_is_incorrect"] += 1
        else:
            counts["num_should_be_empty_is_full" if guess is not None else "num_should_be_empty_is_empty"] += 1
    return counts


def best_guess_history(stats: Statistics, attribute: str) -> list:
    # searched instead of hardcoding the index of the matcher in the pipeline
    run_stats = stats["matching"]["runs"]["0"]
    for element in run_stats._entries.values():
        if isinstance(element, Statistics) and attribute in element._entries:
            attr_stats = element._entries[attribute]
            if isinstance(attr_stats, Statistics) and "best_guesses" in attr_stats._entries:
                return attr_stats._entries["best_guesses"]
    return []


def run_unit(job: Dict[str, Any], cfg: Dict[str, Any], dirs: Dirs, bson_bytes: bytes,
             ground_truth: List[Dict[str, Any]], rm: ResourceManager, worker_id: int) -> Dict[str, Any]:
    variant, k, seed_idx = job["variant"], job["k"], job["seed_idx"]
    attributes = [job["attribute"]]
    seed_value = attribute_seed(cfg, seed_idx, job["attribute"])
    budget = 0 if variant == "A_no_feedback" else cfg["max_rounds"]
    needs_llm = variant != "A_no_feedback" and budget > k

    llm_log = dirs.llm_logs / f"{job['job']}.jsonl"
    if llm_log.exists():  # left over from a killed attempt
        llm_log.unlink()

    holder: Dict[str, Any] = {}

    dataset = cfg.get("dataset", "aviation")
    escalation = variant if variant in ESCALATIONS else None

    def oracle_factory(documents, mapping):
        multi = cfg.get("multi_feedback", False)
        human = (GoldMultiFeedback if multi else AutomaticCustomMatchesRandomRankingBasedMatchingFeedback)(
            documents, mapping)
        llm = None
        stored = stored_setting(variant)
        if stored is not None:
            rule, count = stored
            answers = (gold_answers(ground_truth) if cfg["stored_answers"] == "gold"
                       else load_llm_answers(cfg["stored_answers"], seed_idx, ground_truth))
            # one read set per seed, shared by all columns: a read document answers every column
            read = None if count is None else read_documents(ground_truth, count, cfg["seed_values"][seed_idx],
                                                              cfg.get("read_order", "random"))
            llm = StoredAnswerFeedback(answers, pick_rule=rule, read_documents=read,
                                       multi_min=cfg.get("multi_min", 3))
        elif needs_llm:
            mode = cfg.get("llm_mode", "list")
            llm_class = LLMInteractionCallback if mode == "list" else HumanLikeLLMCallback
            if multi:
                llm_class = MultiFeedbackLLMCallback
            extra = {} if mode == "list" else {"allow_custom_text": mode in ("humanlike-custom", "humanlike-direct"),
                                               "check_in_full_text": mode == "humanlike-direct",
                                               "vote_samples": cfg.get("vote_samples", 1),
                                               "self_check_rounds": cfg.get("self_check_rounds", 0),
                                               "cautious_review": cfg.get("cautious_review", False),
                                               "show_accepted_in_prompt": cfg.get("show_accepted_in_prompt", False),
                                               "probe_samples": cfg.get("probe_samples", 2) if variant == "esc_vote" else 0,
                                               "review_as_signal": variant == "esc_review",
                                               "request_logprobs": cfg.get("request_logprobs", False)}
            if multi:
                extra["multi_context"] = cfg.get("multi_context", "sentence")
            llm = llm_class(
                documents, mapping,
                log_path=str(llm_log),
                include_description=cfg["include_description"],
                attribute_descriptions=dataset_descriptions(dataset) if cfg["include_description"] else None,
                max_response_tokens=cfg["max_response_tokens"],
                max_total_tokens=cfg["max_total_tokens"],
                timeout_seconds=cfg["timeout_seconds"],
                max_retries=cfg["llm_max_retries"],
                random_seed=seed_value,
                show_confirmed_examples=(variant == "examples"),
                **llm_connection(cfg),
                **extra,
            )
        holder["callback"] = HybridInteractionCallback(
            human, llm, human_rounds=k,
            human_round_numbers=human_round_numbers(variant, k, cfg["max_rounds"], seed=seed_value),
            escalation=escalation)
        return holder["callback"]

    logger.info(f"[w{worker_id}] START {job['job']} (seed={seed_value}, budget={budget}, "
                f"human rounds={min(k, budget)}, LLM rounds per attribute={max(budget - k, 0)})")

    stats = Statistics(do_collect=True)
    runner = ExperimentRunner(
        DocumentBase.from_bson(bson_bytes), ground_truth,
        user_attribute_name2attribute_name={a: a for a in attributes},
        preprocessing_pipeline=Pipeline([]),
        matching_pipeline_settings={"max_num_feedback": budget, "store_best_guesses": True,
                                    "len_ranked_list": cfg.get("len_ranked_list", 10)},
        resource_manager=rm, statistics=stats,
    )
    runner.random_seeds = [seed_value]

    t0 = time.perf_counter()
    runner.fill(raw_attributes=attributes, num_runs=1, feedback_oracle=oracle_factory)
    t_total = time.perf_counter() - t0

    final_scores = rg.score_state(runner.document_base, ground_truth, attributes)
    final_by_attr = {row["attribute"]: row for row in final_scores}

    gold_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def gold_of(doc_name: str):
        if doc_name not in gold_cache:
            gold_cache[doc_name] = get_document_by_name(ground_truth, doc_name)
        return gold_cache[doc_name]

    per_round: Dict[str, List[Dict[str, Any]]] = {}
    snapshot_matches_final: Dict[str, bool] = {}
    # the cells themselves, so that other match rules (exact, any overlap) can be scored later
    docs_order = [d.name for d in runner.document_base.documents]
    guess_spans: Dict[str, Dict[str, Any]] = {}
    for attr in attributes:
        rows = []
        span_rounds = []
        history = best_guess_history(stats, attr) if budget > 0 else []
        for i, (round_no, guesses) in enumerate(history):
            counts = count_snapshot(guesses, attr, gold_of)
            rows.append({"round": round_no, **counts, **prf(counts)})
            if round_no % cfg.get("spans_every", 1) == 0 or i == len(history) - 1:
                by_doc = {doc_name: (None if g is None else [g[2], g[3]]) for doc_name, g in guesses}
                span_rounds.append([round_no, [by_doc.get(name) for name in docs_order]])
        final = [None if not d.attribute_mappings.get(attr) else
                 [d.attribute_mappings[attr][0].start_char, d.attribute_mappings[attr][0].end_char]
                 for d in runner.document_base.documents]
        guess_spans[attr] = {"docs": docs_order, "rounds": span_rounds, "final": final}
        per_round[attr] = rows
        if rows:
            snapshot_matches_final[attr] = all(rows[-1][key] == final_by_attr[attr][key] for key in rg.COUNT_KEYS)

    llm_rounds: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    t_llm = 0.0
    if llm_log.exists():
        for line in llm_log.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            usage = rec.get("usage") or {}
            stage2 = rec.get("stage2")
            shown = rec.get("confirmed_examples_shown")
            llm_rounds[rec["attribute"]].append({
                "round": rec.get("num_feedback"),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "seconds": rec["duration_seconds"],
                "stage2": stage2 is not None,
                "message": rec["result_message"],
                "agrees_with_gold": rec.get("agrees_with_gold"),
                "empty_response": rec["raw_response"] == "" or bool(stage2 and stage2["raw_response"] == ""),
                "num_examples_shown": None if shown is None else len(shown),
                "oracle_answer": rec.get("oracle_answer"),
                "decision": rec.get("decision"),
                "llm_calls": rec.get("llm_calls"),
                "signals": rec.get("signals"),
                "multi": rec.get("multi"),
            })
            t_llm += rec["duration_seconds"]

    answers: Dict[str, List[List[Any]]] = defaultdict(list)
    for attr, round_no, answerer, message, document in holder["callback"].answers:
        answers[attr].append([round_no, answerer, message, document])

    record = {
        "job": job["job"], "variant": variant, "k": k, "seed_idx": seed_idx, "attribute": job["attribute"],
        "seed_value": seed_value,
        "budget": budget, "worker_id": worker_id, "host": socket.gethostname(),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "t_total_seconds": t_total, "t_llm_inference_seconds": t_llm, "t_wannadb_matching_seconds": t_total - t_llm,
        "final_scores": final_scores,
        "per_round": per_round,
        "rounds_used": {attr: (rows[-1]["round"] if rows else 0) for attr, rows in per_round.items()},
        "snapshot_matches_final": snapshot_matches_final,
        "llm_rounds": dict(llm_rounds),
        "answers": dict(answers),
        "guess_spans": guess_spans,
        "pick_log": holder["callback"].pick_log,
        "llm_failure_counts": holder["callback"].failure_counts,
    }

    tmp = dirs.checkpoint(job["job"]).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    tmp.replace(dirs.checkpoint(job["job"]))  # atomic: a checkpoint is either complete or absent

    mean_f1 = sum(r["f1_score"] for r in final_scores) / len(final_scores)
    mismatch = [a for a, ok in snapshot_matches_final.items() if not ok]
    logger.info(f"[w{worker_id}] DONE  {job['job']}: t_total={t_total:.0f}s t_llm={t_llm:.0f}s "
                f"mean_f1={mean_f1:.3f}" + (f"  SNAPSHOT MISMATCH {mismatch}" if mismatch else ""))
    return record


def failure_count(dirs: Dirs, job_id: str) -> int:
    path = dirs.failures / f"{job_id}.txt"
    return path.read_text(encoding="utf-8").count("=== attempt") if path.exists() else 0


def try_claim(dirs: Dirs, job: Dict[str, Any], worker_id: int, max_attempts: int) -> bool:
    if dirs.checkpoint(job["job"]).exists() or failure_count(dirs, job["job"]) >= max_attempts:
        return False
    try:
        fd = os.open(dirs.claims / f"{job['job']}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps({"worker_id": worker_id, "pid": os.getpid(), "host": socket.gethostname(),
                            "claimed_at": time.strftime("%Y-%m-%d %H:%M:%S")}))
    if dirs.checkpoint(job["job"]).exists():  # finished between the check and the claim
        (dirs.claims / f"{job['job']}.claim").unlink(missing_ok=True)
        return False
    return True


def worker_main(args) -> int:
    dirs = Dirs(args.run_tag)
    cfg = json.loads(dirs.config.read_text(encoding="utf-8"))
    jobs = json.loads(dirs.jobs.read_text(encoding="utf-8"))

    import torch
    torch.set_num_threads(cfg["threads_per_worker"])

    bson_bytes = document_base_bytes(cfg)
    ground_truth = dataset_module(cfg.get("dataset", "aviation")).load_dataset()

    units_done = 0
    consecutive_failures = 0
    with ResourceManager() as rm:
        while units_done < cfg["units_per_worker"]:
            if dirs.stop.exists() or (dirs.root / f"STOP_{args.worker_id}").exists():
                logger.info(f"[w{args.worker_id}] STOP file found, exiting")
                return 0
            job = next((j for j in jobs if try_claim(dirs, j, args.worker_id, cfg["max_attempts"])), None)
            if job is None:
                logger.info(f"[w{args.worker_id}] nothing left to claim, exiting")
                return 0
            wait = 0
            try:
                run_unit(job, cfg, dirs, bson_bytes, ground_truth, rm, args.worker_id)
                consecutive_failures = 0
                units_done += 1
            except FATAL_LLM_ERRORS as e:
                logger.error(f"[w{args.worker_id}] LLM request rejected during {job['job']} ({type(e).__name__}: {e}), "
                             f"stopping the grid")
                dirs.stop.touch()
                return 1
            except INFRASTRUCTURE_ERRORS as e:
                logger.error(f"[w{args.worker_id}] LLM SERVER UNAVAILABLE during {job['job']} ({type(e).__name__}: {e}); "
                             f"unit released without counting an attempt, waiting {INFRASTRUCTURE_WAIT_SECONDS}s")
                wait = INFRASTRUCTURE_WAIT_SECONDS
            except Exception:
                consecutive_failures += 1
                logger.error(f"[w{args.worker_id}] FAILED {job['job']}")
                logger.error(traceback.format_exc())
                with open(dirs.failures / f"{job['job']}.txt", "a", encoding="utf-8") as f:
                    f.write(f"=== attempt at {time.strftime('%Y-%m-%d %H:%M:%S')} worker {args.worker_id}\n")
                    f.write(traceback.format_exc() + "\n")
                # a systematic problem would otherwise fail every unit within minutes
                wait = min(60 * 2 ** (consecutive_failures - 1), 1800)
            finally:
                (dirs.claims / f"{job['job']}.claim").unlink(missing_ok=True)
            if wait:
                time.sleep(wait)
    logger.info(f"[w{args.worker_id}] finished {units_done} units, exiting so the coordinator starts a fresh process")
    return 0


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, SystemError):
        return False


def progress(dirs: Dirs, cfg: Dict[str, Any], jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    done = [j for j in jobs if dirs.checkpoint(j["job"]).exists()]
    failed = [j for j in jobs if not dirs.checkpoint(j["job"]).exists()
              and failure_count(dirs, j["job"]) >= cfg["max_attempts"]]
    running = sorted(p.stem for p in dirs.claims.glob("*.claim"))
    llm_total = sum(llm_rounds_of(j, cfg) for j in jobs)
    llm_done = sum(llm_rounds_of(j, cfg) for j in done)
    elapsed = time.time() - cfg.get("started_at_epoch", time.time())
    eta_h = None
    if llm_done > 0 and elapsed > 0:
        eta_h = (llm_total - llm_done) / (llm_done / elapsed) / 3600
    return {"done": len(done), "total": len(jobs), "failed": [j["job"] for j in failed], "running": running,
            "llm_rounds_done": llm_done, "llm_rounds_total": llm_total, "elapsed_h": elapsed / 3600, "eta_h": eta_h}


def release_claims_of_dead_worker(dirs: Dirs, worker_id: int, pid: int, exit_code: int) -> None:
    for claim in dirs.claims.glob("*.claim"):
        try:
            info = json.loads(claim.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # by worker id, not pid: on Windows the venv python is a launcher with a different pid
        if info.get("worker_id") != worker_id:
            continue
        job_id = claim.stem
        claim.unlink(missing_ok=True)
        if not dirs.checkpoint(job_id).exists():
            logger.error(f"worker {worker_id} (pid {pid}) died with exit code {exit_code} while running {job_id}")
            with open(dirs.failures / f"{job_id}.txt", "a", encoding="utf-8") as f:
                f.write(f"=== attempt at {time.strftime('%Y-%m-%d %H:%M:%S')} worker {worker_id}\n"
                        f"worker process died with exit code {exit_code}, no Python traceback\n")


def claimable_left(dirs: Dirs, cfg: Dict[str, Any], jobs: List[Dict[str, Any]]) -> bool:
    return any(not dirs.checkpoint(j["job"]).exists() and failure_count(dirs, j["job"]) < cfg["max_attempts"]
               and not (dirs.claims / f"{j['job']}.claim").exists() for j in jobs)


def check_llm(cfg: Dict[str, Any]) -> Optional[str]:
    """One small request, so that a wrong model name, key or URL shows up before the grid starts."""
    conn = llm_connection(cfg)
    client = OpenAI(base_url=conn["base_url"], api_key=conn["api_key"] or "not-needed", timeout=60, max_retries=1)
    try:
        client.chat.completions.create(model=conn["model"], messages=[{"role": "user", "content": "Reply with: ok"}],
                                       max_tokens=16, temperature=0, extra_body=conn["extra_body"])
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


def tokens_used(dirs: Dirs, cache: Dict[str, int]) -> int:
    for path in dirs.checkpoints.glob("*.json"):
        if path.stem not in cache:
            unit = json.loads(path.read_text(encoding="utf-8"))
            cache[path.stem] = sum(r["prompt_tokens"] + r["completion_tokens"]
                                   for rounds in (unit.get("llm_rounds") or {}).values() for r in rounds)
    return sum(cache.values())


def coordinator_main(args) -> int:
    if args.wandb:
        if not wandb_log.api_key_configured():
            logger.error("--wandb given, but this machine is not logged in to W&B: run `wandb login` first")
            return 1

    dirs = Dirs(args.run_tag)
    stored = json.loads(dirs.config.read_text(encoding="utf-8")) if dirs.config.exists() else None
    k_values = args.k_values if args.k_values is not None else list(range(0, args.max_rounds + 1))
    grid = stored or {"k_values": k_values, "max_rounds": args.max_rounds, "model": args.model,
                      "base_url": args.base_url, "api_key_env": args.api_key_env, "provider": args.provider,
                      "stored_answers": args.stored_answers}
    if any(k < grid["max_rounds"] for k in grid["k_values"]) and not grid.get("stored_answers"):
        if stored is None and args.model is None:
            logger.error("choose the LLM with --model, it is stored with the grid")
            return 1
        problem = check_llm(grid)
        if problem:
            logger.error(f"LLM check failed, grid not started: {problem}")
            return 1

    dirs.create()

    if dirs.coordinator_pid.exists():
        old = int(dirs.coordinator_pid.read_text().strip() or 0)
        if old and old != os.getpid() and pid_alive(old):
            logger.error(f"another coordinator (pid {old}) is already running for {args.run_tag}")
            return 1
    dirs.coordinator_pid.write_text(str(os.getpid()))
    dirs.stop.unlink(missing_ok=True)
    dirs.done_marker.unlink(missing_ok=True)

    dataset = dataset_module(args.dataset)
    attributes = args.attributes or list(dataset.ATTRIBUTES)

    if dirs.config.exists():
        cfg = json.loads(dirs.config.read_text(encoding="utf-8"))
        logger.info(f"resuming {args.run_tag} with its stored config (command line grid options ignored)")
    else:
        with ResourceManager() as rm:
            # same seeds as run_grid.py; they do not depend on the dataset
            probe = ExperimentRunner(DocumentBase.from_bson(rg.BSON_PATH.read_bytes()), rg.load_aviation_module().load_dataset(),
                                     resource_manager=rm, statistics=Statistics(do_collect=False),
                                     preprocessing_pipeline=Pipeline([]))
            # held-out seeds keep their index, so seed 3 is the same seed in every grid
            seed_indices = args.seed_indices if args.seed_indices else list(range(args.num_seeds))
            seed_values = probe.random_seeds[:max(seed_indices) + 1]
        if args.multi_feedback and any(k < args.max_rounds for k in k_values) and args.llm_mode != "humanlike-direct":
            raise SystemExit("--multi-feedback with LLM rounds needs --llm-mode humanlike-direct")
        variants = args.variants + [f"reads_{count}" for count in args.read_documents]
        if any(stored_setting(v) for v in variants) != bool(args.stored_answers):
            raise SystemExit("--stored-answers belongs to the rule_* variants and --read-documents, and they need it")
        cfg = {
            "run_tag": args.run_tag, "attributes": attributes, "max_rounds": args.max_rounds,
            "k_values": sorted(set(k_values)), "variants": variants, "num_seeds": len(seed_indices),
            "seed_indices": seed_indices, "seed_values": seed_values, "workers": args.workers, "threads_per_worker": args.threads_per_worker,
            "units_per_worker": args.units_per_worker, "max_attempts": args.max_attempts,
            "include_description": args.include_description, "max_response_tokens": args.max_response_tokens,
            "max_total_tokens": args.max_total_tokens, "timeout_seconds": args.timeout_seconds,
            "llm_max_retries": args.llm_max_retries, "device": args.device, "llm_mode": args.llm_mode,
            "prompt_version": "list-v1" if args.llm_mode == "list" else PROMPT_VERSION,
            "multi_feedback": args.multi_feedback,
            "multi_context": args.multi_context if args.multi_feedback else None,
            "multi_prompt_version": MULTI_PROMPT_VERSIONS[args.multi_context] if args.multi_feedback else None,
            # which matcher ran, e.g. before or after the multi-match change
            "matching_py_md5": hashlib.md5((rg.ROOT / "wannadb" / "matching" / "matching.py").read_bytes()).hexdigest(),
            "vote_samples": args.vote_samples, "self_check_rounds": args.self_check_rounds,
            "cautious_review": args.cautious_review, "show_accepted_in_prompt": args.show_accepted_in_prompt,
            "model": args.model, "base_url": args.base_url, "api_key_env": args.api_key_env,
            "provider": args.provider, "max_grid_tokens": args.max_grid_tokens,
            "dataset": args.dataset, "num_docs": args.num_docs, "subset_seed": args.subset_seed,
            "spans_every": args.spans_every, "probe_samples": args.probe_samples,
            "request_logprobs": args.request_logprobs,
            "stored_answers": args.stored_answers, "len_ranked_list": args.len_ranked_list,
            "read_order": args.read_order, "multi_min": args.multi_min,
            "bson": dataset_bson(args.dataset).name, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "started_at_epoch": time.time(),
        }
        dirs.config.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        jobs = build_jobs(cfg["k_values"], cfg["max_rounds"], cfg["seed_indices"], cfg["variants"], cfg["attributes"])
        dirs.jobs.write_text(json.dumps(jobs, indent=1), encoding="utf-8")

    jobs = json.loads(dirs.jobs.read_text(encoding="utf-8"))

    # no worker runs yet, so every claim is left over from a killed run
    for claim in dirs.claims.glob("*.claim"):
        claim.unlink()

    logger.info(f"run_tag={args.run_tag} units={len(jobs)} workers={args.workers} model={llm_connection(cfg)['model']} "
                f"seeds={cfg['seed_values']} k_values={len(cfg['k_values'])} variants={cfg['variants']}")

    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = str(cfg["threads_per_worker"])
    env["TOKENIZERS_PARALLELISM"] = "false"
    if cfg["device"] == "cpu":
        # the GPUs are shared with other users and can be full at any time
        env["CUDA_VISIBLE_DEVICES"] = ""

    def spawn(worker_id: int) -> subprocess.Popen:
        log = open(dirs.worker_logs / f"worker{worker_id:02d}.log", "a", encoding="utf-8")
        return subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run-tag", args.run_tag,
                                 "--worker-id", str(worker_id)], stdout=log, stderr=subprocess.STDOUT, env=env)

    workers: Dict[int, subprocess.Popen] = {}
    quick_failures: Dict[int, int] = defaultdict(int)
    started: Dict[int, float] = {}
    for wid in range(args.workers):
        workers[wid] = spawn(wid)
        started[wid] = time.time()
        time.sleep(2)  # stagger model loading

    last_report = 0.0
    token_cache: Dict[str, int] = {}
    while True:
        for wid, proc in list(workers.items()):
            code = proc.poll()
            if code is None:
                continue
            del workers[wid]
            release_claims_of_dead_worker(dirs, wid, proc.pid, code)
            if code != 0:
                logger.error(f"worker {wid} (pid {proc.pid}) exited with code {code}")
            ran_for = time.time() - started[wid]
            quick_failures[wid] = quick_failures[wid] + 1 if (code != 0 and ran_for < 300) else 0
            if dirs.stop.exists() or not claimable_left(dirs, cfg, jobs):
                continue
            if quick_failures[wid] >= 3:
                logger.error(f"worker {wid} crashed 3 times within 5 minutes, not restarting it")
                continue
            workers[wid] = spawn(wid)
            started[wid] = time.time()

        if time.time() - last_report > 600 or not workers:
            p = progress(dirs, cfg, jobs)
            eta = "?" if p["eta_h"] is None else f"{p['eta_h']:.1f}h"
            used = tokens_used(dirs, token_cache)
            logger.info(f"PROGRESS {p['done']}/{p['total']} units, LLM rounds {p['llm_rounds_done']}/"
                        f"{p['llm_rounds_total']}, tokens {used / 1e6:.1f}M, elapsed {p['elapsed_h']:.1f}h, ETA {eta}, "
                        f"running {len(p['running'])}, failed {len(p['failed'])}, workers alive {len(workers)}")
            # checked every 10 minutes, units still running when the budget is reached are finished
            if cfg.get("max_grid_tokens") and used >= cfg["max_grid_tokens"] and not dirs.stop.exists():
                logger.error(f"token budget of {cfg['max_grid_tokens']} used up, stopping after the running units")
                dirs.stop.touch()
            last_report = time.time()

        if not workers:
            break
        time.sleep(POLL_SECONDS)

    p = progress(dirs, cfg, jobs)
    summary = {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), **p, "stopped_by_user": dirs.stop.exists()}
    dirs.done_marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"GRID DONE {json.dumps(summary)}")
    dirs.coordinator_pid.unlink(missing_ok=True)

    if args.wandb:
        try:
            wandb_log.log_hybrid_grid(args.run_tag, wandb_log.Uploader(offline=False, limit=None, same_checkout=True))
        except Exception:
            logger.error(f"logging to W&B failed, the grid results are unaffected. Log them later with: "
                         f"python scripts/wandb_log.py --hybrid {args.run_tag}\n{traceback.format_exc()}")
    return 0 if (p["done"] == p["total"]) else 1


def status_main(args) -> int:
    dirs = Dirs(args.run_tag)
    cfg = json.loads(dirs.config.read_text(encoding="utf-8"))
    jobs = json.loads(dirs.jobs.read_text(encoding="utf-8"))
    p = progress(dirs, cfg, jobs)
    print(json.dumps({**p, "grid_done": dirs.done_marker.exists()}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-tag", default=time.strftime("hybrid_%Y%m%d-%H%M%S"))
    parser.add_argument("--worker-id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--status", action="store_true", help="print progress of --run-tag and exit")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--k-values", type=int, nargs="*", default=None, help="default: 0..max-rounds")
    parser.add_argument("--variants", nargs="*", default=["handoff", "examples"],
                        choices=list(SCHEDULES) + list(ESCALATIONS) + list(RULE_VARIANTS),
                        help="order of the k human rounds: handoff/examples first, llmfirst last, interleave spread "
                             "evenly, sandwich half at the start and half at the end, random at random rounds; "
                             "esc_*: the human answers the first k rounds and every LLM round the signal marks as "
                             "unsure (vote: sampled answers disagree, review: a reviewer says revise, oracle: the "
                             "answer is wrong by gold, an upper bound); rule_*: the human answers the first k rounds, "
                             "then a fixed rule picks the row and --stored-answers answers it, no LLM is asked")
    parser.add_argument("--stored-answers", default=None,
                        help="for the rule_* variants and --read-documents: answers.jsonl of llm_only_baseline.py "
                             "(repeat = seed index), or gold")
    parser.add_argument("--read-documents", type=int, nargs="*", default=[],
                        help="adds a variant reads_N per value: only N documents are answered (the same N in every "
                             "column of a seed), WannaDB fills the rest; needs a long enough --len-ranked-list")
    parser.add_argument("--multi-min", type=int, default=3,
                        help="rule_multi*: fewest right rows for confirming them at once, else one row is answered")
    parser.add_argument("--read-order", choices=READ_ORDERS, default="random",
                        help="how --read-documents chooses the documents, always from the texts alone")
    parser.add_argument("--len-ranked-list", type=int, default=10,
                        help="rows WannaDB shows per round, half on each side of the threshold; more rows than twice "
                             "the documents shows all of them, and a confirmed row moves the threshold further")
    parser.add_argument("--dataset", default="aviation",
                        help="a folder in wannadb_datsets/datasets with a preprocessed bson in bson_files")
    parser.add_argument("--num-docs", type=int, default=None,
                        help="use a random subset of this many documents (nested across sizes), for scaling")
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--spans-every", type=int, default=1,
                        help="store the cells of every n-th round (all rounds for 1), for other match rules later")
    parser.add_argument("--probe-samples", type=int, default=2,
                        help="esc_vote: sampled extra answers (temperature 0.7) that must agree with the answer")
    parser.add_argument("--request-logprobs", action="store_true",
                        help="ask the server for token log probabilities and log the weakest one of the answer")
    parser.add_argument("--llm-mode", default="list",
                        choices=["list", "humanlike-nuggets", "humanlike-custom", "humanlike-direct"],
                        help="list: pick one of the whole shortlist (first version); humanlike: one random document, "
                             "confirm the guess in its sentence or edit on the tagged full text, custom also lets the "
                             "LLM copy its own value; direct checks the guess on the tagged full text in one call")
    parser.add_argument("--multi-feedback", action="store_true",
                        help="human and LLM confirm every right row of the ranked list at once (multi-match, needs "
                             "the multi-match matcher); with no right row they answer one row as before. LLM only "
                             "with --llm-mode humanlike-direct")
    parser.add_argument("--multi-context", choices=["sentence", "paragraph"], default="sentence",
                        help="what each row of the LLM's list question shows around the guess")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed-indices", type=int, nargs="*", default=None,
                        help="run only these seeds (indices into ExperimentRunner.random_seeds), e.g. held-out 3 4")
    parser.add_argument("--attributes", nargs="*", default=None, help="default: all 12 aviation attributes")
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--units-per-worker", type=int, default=150,
                        help="a worker exits after this many units and is replaced, bounding memory growth")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="attempts per unit; LLM server outages do not count as an attempt")
    parser.add_argument("--llm-max-retries", type=int, default=10,
                        help="retries per LLM request on server trouble, waits 1s, 2s, 4s, ... capped at 300s "
                             "(about 13 minutes in total) before the unit is released")
    parser.add_argument("--device", choices=["cpu", "auto"], default="cpu",
                        help="cpu hides the GPUs from the workers; auto lets torch pick")
    parser.add_argument("--include-description", action="store_true")
    parser.add_argument("--vote-samples", type=int, default=1,
                        help="human-like modes: majority over this many sampled answers (temperature 0.7)")
    parser.add_argument("--self-check-rounds", type=int, default=0,
                        help="human-like modes: a reviewer call checks the answer against the values accepted so far, "
                             "up to this many times")
    parser.add_argument("--cautious-review", action="store_true",
                        help="the reviewer asks for a revision only when it is sure the answer is wrong")
    parser.add_argument("--show-accepted-in-prompt", action="store_true",
                        help="human-like modes: the answering prompt lists the values accepted so far (own answers "
                             "and human confirmations), as a control for the self-check")
    parser.add_argument("--max-response-tokens", type=int, default=16384)
    parser.add_argument("--max-total-tokens", type=int, default=20_000_000)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--model", default=None, help="required for a new grid with LLM rounds")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default=None,
                        help="name of the environment variable that holds the API key, e.g. OPENROUTER_API_KEY")
    parser.add_argument("--provider", default=None,
                        help="OpenRouter only: send every request to this provider, without fallbacks")
    parser.add_argument("--max-grid-tokens", type=int, default=None,
                        help="stop the grid once its finished units used this many tokens")
    parser.add_argument("--wandb", action="store_true",
                        help="log the finished grid to Weights & Biases (scripts/wandb_log.py); "
                             "the login is checked before the grid starts")
    args = parser.parse_args()

    if args.status:
        return status_main(args)
    if args.worker_id is not None:
        return worker_main(args)
    return coordinator_main(args)


if __name__ == "__main__":
    sys.exit(main())
