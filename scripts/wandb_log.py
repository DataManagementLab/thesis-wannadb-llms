import argparse
import ast
import inspect
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import hybrid_analysis as ha
import llm_feedback
import wandb

ENTITY = "abdelhamid-elhouari-wissenschaftsstadt-darmstadt"
PROJECT = "wannadb-llm-feedback"
OVERNIGHT_CHECKPOINTS = ROOT / "experiments" / "results" / "checkpoints"
OVERNIGHT_LLM_LOGS = ROOT / "experiments" / "llm_feedback_logs"
HUMAN_ORACLE = "AutomaticCustomMatchesRandomRankingBasedMatchingFeedback"
# grids from before the model was stored in config.json all ran on this model
LEGACY_MODEL = "unsloth/DeepSeek-V4-Flash-0731"
LEGACY_BASE_URL = "http://10.0.21.72:13505/v1"


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def api_key_configured() -> bool:
    try:
        wandb.Api(timeout=30)
        return True
    except wandb.errors.UsageError:
        return False


def matcher_settings() -> Dict[str, Any]:
    """Read from the source, because building the pipeline would load the embedding models."""
    tree = ast.parse((ROOT / "experiments" / "experiment_runner.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        target = getattr(node, "target", None) or (node.targets[0] if isinstance(node, ast.Assign) else None)
        if getattr(target, "id", None) == "predefined_settings" and isinstance(node.value, ast.Dict):
            out = {}
            for key, value in zip(node.value.keys, node.value.values):
                try:
                    out[ast.literal_eval(key)] = ast.literal_eval(value)
                except ValueError:
                    out[ast.literal_eval(key)] = ast.unparse(value)  # e.g. "DummyCustomMatchExtractor()"
            return out
    raise RuntimeError("predefined_settings not found in experiments/experiment_runner.py")


def llm_settings() -> Dict[str, Any]:
    source = inspect.getsource(llm_feedback)
    temperatures = sorted({kw.value.value for node in ast.walk(ast.parse(source))
                           for kw in getattr(node, "keywords", [])
                           if kw.arg == "temperature" and isinstance(kw.value, ast.Constant)})
    params = inspect.signature(llm_feedback.LLMInteractionCallback.__init__).parameters
    defaults = {name: p.default for name, p in params.items() if isinstance(p.default, (bool, int, float, str))}
    return {"temperature": temperatures[0] if len(temperatures) == 1 else temperatures,
            "prompt_format": "a: candidate list; on rejection a second call over all nuggets of the document",
            **defaults}


def llm_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    usage = rec.get("usage") or {}
    stage2 = rec.get("stage2")
    return {"round": rec.get("num_feedback"), "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0), "seconds": rec["duration_seconds"],
            "stage2": stage2 is not None, "message": rec["result_message"],
            "agrees_with_gold": rec.get("agrees_with_gold"),
            "empty_response": rec["raw_response"] == "" or bool(stage2 and stage2["raw_response"] == "")}


def llm_summary(rows: pd.DataFrame, failure_counts: Iterable[Optional[Dict[str, int]]]) -> Dict[str, Any]:
    failures = defaultdict(int)
    for counts in failure_counts:
        for key, value in (counts or {}).items():
            failures[key] += value
    if rows.empty:
        return {"llm/rounds": 0, "llm/total_tokens": 0, "llm/hours": 0.0}
    confirmed = rows[rows["message"] == "is-match"]
    # grids before 2026-09-30 did not check stage 2 confirmations against gold
    checked = confirmed[confirmed["agrees_with_gold"].notna()]
    return {
        "llm/rounds": len(rows),
        "llm/prompt_tokens": int(rows["prompt_tokens"].sum()),
        "llm/completion_tokens": int(rows["completion_tokens"].sum()),
        "llm/total_tokens": int(rows["prompt_tokens"].sum() + rows["completion_tokens"].sum()),
        "llm/hours": float(rows["seconds"].sum() / 3600),
        "llm/confirm_rate": float(len(confirmed) / len(rows)),
        "llm/correct_when_confirmed": float((checked["agrees_with_gold"] == True).mean()) if len(checked) else None,  # noqa: E712
        "llm/confirmations_not_checked": float(1 - len(checked) / len(confirmed)) if len(confirmed) else None,
        "llm/second_stage_rate": float(rows["stage2"].mean()),
        "llm/empty_response_rate": float(rows["empty_response"].mean()),
        "llm/parse_failures": failures["parse_failures"],
        "llm/truncated_responses": failures["truncated_responses"],
    }


def macro(scores: List[Dict[str, Any]]) -> Dict[str, float]:
    return {"macro_f1": float(np.mean([s["f1_score"] for s in scores])),
            "macro_precision": float(np.mean([s["precision"] for s in scores])),
            "macro_recall": float(np.mean([s["recall"] for s in scores]))}


def table(df: pd.DataFrame):
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return wandb.Table(dataframe=df.astype(object).where(df.notna(), None))


class Uploader:
    def __init__(self, offline: bool, limit: Optional[int], same_checkout: bool):
        self.offline = offline
        self.limit = limit
        self.same_checkout = same_checkout
        self.started = 0
        self.skipped: List[str] = []
        self.existing = set() if offline else self._existing_ids()

    @staticmethod
    def _existing_ids() -> set:
        try:
            return {r.id for r in wandb.Api(timeout=60).runs(f"{ENTITY}/{PROJECT}", per_page=500)}
        except (ValueError, wandb.errors.CommError):
            return set()  # project does not exist yet

    def start(self, run_id: str, name: str, group: str, job_type: str, tags: List[str], config: Dict[str, Any]):
        run_id = re.sub(r"[^A-Za-z0-9_-]", "-", run_id)
        if run_id in self.existing:
            self.skipped.append(run_id)
            return None
        if self.limit is not None and self.started >= self.limit:
            return None
        self.started += 1
        # git state and machine info only describe the run when logging right after it
        settings = wandb.Settings(console="off", quiet=True, x_disable_stats=True,
                                  disable_git=not self.same_checkout, x_disable_meta=not self.same_checkout)
        return wandb.init(entity=ENTITY, project=PROJECT, id=run_id, name=name, group=group,
                          job_type=job_type, tags=tags, config=config, dir=str(ROOT), settings=settings,
                          mode="offline" if self.offline else "online")

    def limit_reached(self) -> bool:
        return self.limit is not None and self.started >= self.limit


def log_hybrid_grid(run_tag: str, up: Uploader, figures: Optional[Path] = None) -> None:
    run = ha.load_run(run_tag)
    cfg = run["config"]
    attributes = cfg["attributes"]
    max_rounds = cfg["max_rounds"]

    per_round = ha.per_round_frame(run, complete_only=False)
    llm_rounds = ha.llm_round_frame(run, complete_only=False)
    per_round_path = run["root"] / "per_round.parquet"
    llm_rounds_path = run["root"] / "llm_rounds.parquet"
    per_round.to_parquet(per_round_path, index=False)
    llm_rounds.to_parquet(llm_rounds_path, index=False)
    print(f"wrote {rel(per_round_path)} ({len(per_round)} rows) and {rel(llm_rounds_path)} ({len(llm_rounds)} rows)")

    units = defaultdict(list)
    for u in run["units"]:
        units[(u["variant"], u["k"], u["seed_idx"])].append(u)

    matcher = matcher_settings()
    llm_defaults = llm_settings()
    shared = {
        "grid": run_tag, "grid_type": "hybrid", "runner": "scripts/run_hybrid_grid.py",
        "dataset": "aviation", "bson": cfg["bson"], "attributes": attributes, "device": cfg["device"],
        "each_attribute_run_separately": True, "human_oracle": HUMAN_ORACLE,
        "model": cfg.get("model", LEGACY_MODEL),
        "grid_settings": {k: cfg[k] for k in ("k_values", "variants", "num_seeds", "seed_values", "workers",
                                              "threads_per_worker", "max_attempts", "created_at") if k in cfg},
        "results_dir": rel(run["root"]), "per_round_parquet": rel(per_round_path),
        "llm_rounds_parquet": rel(llm_rounds_path), "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "logged_at_grid_end": up.same_checkout,
    }

    incomplete = []
    for (variant, h, seed), group_units in sorted(units.items(), key=lambda kv: (kv[0][0] != "A_no_feedback",) + kv[0]):
        if up.limit_reached():
            break
        if {u["attribute"] for u in group_units} != set(attributes):
            incomplete.append(f"{variant} h={h} seed={seed}")
            continue
        no_feedback = variant == "A_no_feedback"
        budget = 0 if no_feedback else max_rounds
        name = f"no_feedback_seed{seed}" if no_feedback else f"{variant}_h{h:02d}_seed{seed}"
        group = f"{run_tag}:no_feedback" if no_feedback else f"{run_tag}:{variant}_h{h:02d}"
        llm_cfg = None
        if not no_feedback and h < budget:
            llm_cfg = {**llm_defaults, "model": cfg.get("model", LEGACY_MODEL),
                       "base_url": cfg.get("base_url", LEGACY_BASE_URL), "provider": cfg.get("provider"),
                       "include_description": cfg["include_description"],
                       "max_response_tokens": cfg["max_response_tokens"], "max_total_tokens": cfg["max_total_tokens"],
                       "timeout_seconds": cfg["timeout_seconds"], "max_retries": cfg["llm_max_retries"],
                       "show_confirmed_examples": variant == "examples"}
        config = {**shared, "variant": variant, "human_rounds": 0 if no_feedback else h,
                  "llm_rounds_per_attribute": max(budget - h, 0), "budget_per_attribute": budget,
                  "seed_idx": seed, "seed_value": cfg["seed_values"][seed],
                  "attribute_seeds": {u["attribute"]: u["seed_value"] for u in group_units},
                  "matcher": {**matcher, "max_num_feedback": budget, "store_best_guesses": True},
                  "llm": llm_cfg}
        wb = up.start(f"{run_tag}-{name}", name, group, variant, [run_tag, "hybrid-grid", variant], config)
        if wb is None:
            continue

        if no_feedback:
            scores = [s for u in group_units for s in u["final_scores"]]
            wb.log({**macro(scores), **{f"f1/{s['attribute']}": s["f1_score"] for s in scores}}, step=0)
        else:
            sub = per_round[(per_round.variant == variant) & (per_round.h == h) & (per_round.seed == seed)]
            f1 = sub.pivot(index="round", columns="attribute", values="f1")
            prec = sub.groupby("round")["precision"].mean()
            recall = sub.groupby("round")["recall"].mean()
            llm = llm_rounds[(llm_rounds.variant == variant) & (llm_rounds.h == h) & (llm_rounds.seed == seed)] \
                if not llm_rounds.empty else llm_rounds
            if llm.empty:
                tokens_cum = seconds_cum = pd.Series(0.0, index=f1.index)
            else:
                by_round = llm.assign(tokens=llm.prompt_tokens + llm.completion_tokens).groupby("round")[["tokens", "seconds"]].sum()
                by_round = by_round.reindex(f1.index, fill_value=0).cumsum()
                tokens_cum, seconds_cum = by_round["tokens"], by_round["seconds"]
            for r in f1.index:
                wb.log({"macro_f1": float(f1.loc[r].mean()), "macro_precision": float(prec[r]),
                        "macro_recall": float(recall[r]), "human_rounds_so_far": min(r, h),
                        "llm_rounds_so_far": max(r - h, 0), "llm_tokens_cumulative": float(tokens_cum[r]),
                        "llm_hours_cumulative": float(seconds_cum[r] / 3600),
                        **{f"f1/{a}": float(f1.loc[r, a]) for a in f1.columns}}, step=int(r))
        wb.summary.update({
            **llm_summary(llm_rounds[(llm_rounds.variant == variant) & (llm_rounds.h == h) & (llm_rounds.seed == seed)]
                          if not llm_rounds.empty else llm_rounds, [u["llm_failure_counts"] for u in group_units]),
            "time/total_hours": sum(u["t_total_seconds"] for u in group_units) / 3600,
            "time/matching_minutes": sum(u["t_wannadb_matching_seconds"] for u in group_units) / 60,
            "check/snapshots_match_final": all(all(u["snapshot_matches_final"].values()) for u in group_units),
            "hosts": sorted({u["host"] for u in group_units}),
            "finished_at": max(u["finished_at"] for u in group_units),
        })
        wb.finish()
        print(f"  logged {name}")

    if incomplete:
        print(f"not logged, not every attribute finished: {', '.join(incomplete)}")
    if not up.limit_reached():
        log_hybrid_summary(run_tag, run, per_round, up, shared, figures)


def log_hybrid_summary(run_tag: str, run: Dict[str, Any], per_round: pd.DataFrame, up: Uploader,
                       shared: Dict[str, Any], figures: Optional[Path]) -> None:
    max_rounds = run["config"]["max_rounds"]
    complete = ha.complete_combinations(run)
    per_round = per_round[[(v, h) in complete for v, h in zip(per_round.variant, per_round.h)]]
    wb = up.start(f"{run_tag}-summary", f"{run_tag}_summary", f"{run_tag}:summary", "analysis",
                  [run_tag, "hybrid-grid", "summary"], {**shared, "complete_settings": len(complete)})
    if wb is None:
        return
    curves = []
    for variant, h in sorted(complete):
        if variant == "A_no_feedback":
            continue
        c = ha.macro_curve(per_round, variant, h)
        curves.append(c.assign(variant=variant, h=h)[["variant", "h", "round", "mean", "std", "seeds"]])
    logged = {"macro_curves": table(pd.concat(curves, ignore_index=True))}
    headline = {}
    for variant in sorted({v for v, _ in complete} - {"A_no_feedback"}):
        mat = ha.budget_matrix(per_round, variant)
        mat.columns = [f"r{c}" for c in mat.columns]
        logged[f"budget_matrix/{variant}"] = table(mat.reset_index())
    if ("handoff", 0) in complete and ("handoff", max_rounds) in complete:
        eq = ha.equivalent_human_rounds(per_round, "handoff", max_rounds)
        logged[f"human_rounds_replaced_at_budget_{max_rounds}"] = table(eq.reset_index())
        llm_only = ha.macro_curve(per_round, "handoff", 0).set_index("round")["mean"]
        human_only = ha.macro_curve(per_round, "handoff", max_rounds).set_index("round")["mean"]
        headline.update({"no_feedback_f1": float(human_only[0]),
                         f"human_only_f1_at_{max_rounds}": float(human_only[max_rounds]),
                         f"llm_only_f1_at_{max_rounds}": float(llm_only[max_rounds]),
                         f"human_rounds_replaced_by_{max_rounds}_llm_rounds": float(eq.loc[0, "human rounds saved"])})
    if any(v == "examples" for v, _ in complete):
        diff = (ha.budget_matrix(per_round, "examples")[max_rounds]
                - ha.budget_matrix(per_round, "handoff")[max_rounds]).dropna().drop([0, max_rounds], errors="ignore")
        headline.update({"examples_minus_handoff_mean": float(diff.mean()),
                         "examples_better_share": float((diff > 0).mean())})
    if figures is not None:
        for png in sorted(Path(figures).glob("*.png")):
            logged[f"figures/{png.stem}"] = wandb.Image(str(png))
    wb.log(logged)
    wb.summary.update(headline)
    wb.finish()
    print(f"  logged {run_tag}_summary: {', '.join(f'{k}={v:.3f}' for k, v in headline.items())}")


def log_overnight_grid(run_tag: str, up: Uploader) -> None:
    cfg = json.loads((ROOT / "experiments" / "results" / f"{run_tag}_config.json").read_text(encoding="utf-8"))
    llm = llm_settings()
    for path in sorted(OVERNIGHT_CHECKPOINTS.glob(f"{run_tag}_*_seed*.json")):
        if up.limit_reached():
            break
        rec = json.loads(path.read_text(encoding="utf-8"))
        baseline, seed = rec["baseline"], rec["seed_idx"]
        budget = 0 if baseline == "A_no_feedback" else cfg["max_num_feedback"]
        name = f"{baseline}_seed{seed}"
        rows = pd.DataFrame()
        if rec.get("llm_log_file"):
            log = OVERNIGHT_LLM_LOGS / Path(rec["llm_log_file"]).name
            if log.exists():
                rows = pd.DataFrame([llm_row(json.loads(line)) for line in log.read_text(encoding="utf-8").splitlines()])
        config = {
            "grid": run_tag, "grid_type": "overnight", "runner": "scripts/run_grid.py", "dataset": "aviation",
            "attributes": cfg["attributes"], "each_attribute_run_separately": False,
            "variant": baseline, "budget_per_attribute": budget, "seed_idx": seed, "seed_value": rec["seed_value"],
            "human_oracle": HUMAN_ORACLE if baseline == "B_gold_oracle" else None,
            "matcher": {**matcher_settings(), "max_num_feedback": budget, "store_best_guesses": True},
            # the callback code changed after this grid, so only what the grid recorded itself
            "llm": None if baseline != "C_llm" else {
                "model": LEGACY_MODEL, "base_url": LEGACY_BASE_URL, "temperature": llm["temperature"],
                "prompt_format": llm["prompt_format"], "include_description": cfg["include_description"],
                "max_response_tokens": cfg["max_response_tokens"], "max_total_tokens": cfg["max_total_tokens"],
                "timeout_seconds": cfg["timeout_seconds"]},
            "results": rel(path), "llm_log": rel(OVERNIGHT_LLM_LOGS / Path(rec["llm_log_file"]).name) if rec.get("llm_log_file") else None,
            "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"), "logged_at_grid_end": False,
        }
        wb = up.start(f"{run_tag}-{name}", name, f"{run_tag}:{baseline}", baseline,
                      [run_tag, "overnight-grid", baseline], config)
        if wb is None:
            continue
        wb.log({**macro(rec["scores"]), **{f"f1/{s['attribute']}": s["f1_score"] for s in rec["scores"]}}, step=budget)
        wb.summary.update({**llm_summary(rows, [rec.get("llm_failure_counts")]),
                           "time/total_hours": rec["t_total_seconds"] / 3600,
                           "time/matching_minutes": rec["t_wannadb_matching_seconds"] / 60})
        wb.finish()
        print(f"  logged {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hybrid", nargs="*", default=[], help="run tags of run_hybrid_grid.py grids")
    parser.add_argument("--overnight", nargs="*", default=[], help="run tags of run_grid.py grids")
    parser.add_argument("--figures", type=Path, default=None, help="folder of PNGs to attach to the hybrid summary run")
    parser.add_argument("--offline", action="store_true", help="write W&B files locally only, upload nothing")
    parser.add_argument("--limit", type=int, default=None, help="log at most this many W&B runs (for testing)")
    parser.add_argument("--same-checkout", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.offline and not api_key_configured():
        print("Not logged in to W&B. Run `wandb login` once on this machine, or use --offline.")
        return 1
    up = Uploader(args.offline, args.limit, args.same_checkout)
    for tag in args.overnight:
        log_overnight_grid(tag, up)
    for tag in args.hybrid:
        log_hybrid_grid(tag, up, args.figures)
    print(f"done: {up.started} W&B runs logged, {len(up.skipped)} already existed"
          + ("" if args.offline else f", see https://wandb.ai/{ENTITY}/{PROJECT}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
