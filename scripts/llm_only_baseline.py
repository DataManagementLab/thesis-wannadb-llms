import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

import run_grid as rg
from analyze_attributes import f1_of, md
from analyze_match_rules import RULES, counts
from llm_feedback import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMInteractionCallback
from llm_humanlike_feedback import HumanLikeLLMCallback
from run_hybrid_grid import DATASETS_DIR, dataset_descriptions, dataset_module

OUT_ROOT = ROOT / "experiments" / "results" / "llm_only"
MAX_WORKERS = 12  # the shared server must stay usable for others
MAX_TOTAL_TOKENS = 5_000_000


def column_label(name: str, descriptions: Optional[Dict[str, str]]) -> str:
    description = (descriptions or {}).get(name)
    return f'"{name}" (meaning: {description})' if description else f'"{name}"'


def build_prompt(text: str, attributes: List[str], descriptions: Optional[Dict[str, str]]) -> str:
    answer = ", ".join(f'"{a}": "<exact words from the document>" or null' for a in attributes)
    return "\n".join([
        "An information extraction system fills a table with one row per document. The columns are:",
        *(f"- {column_label(a, descriptions)}" for a in attributes),
        "",
        "Below is one complete document.",
        "",
        "Document:",
        '"""',
        text,
        '"""',
        "",
        "Judge only from the text, not from outside knowledge.",
        "For every column, copy only the value word for word from the document. If the document does not contain a "
        "value for a column, answer null for it.",
        "",
        "Respond with exactly one JSON object and nothing else: {" + answer + "}",
    ])


def ask(client: OpenAI, model: str, prompt: str, max_tokens: int, retries: int = 5):
    last_error: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            return client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                                  max_tokens=max_tokens, temperature=0)
        except (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError) as e:
            last_error = e
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 300))
    raise last_error


def answer_document(client: OpenAI, args, doc: Dict[str, Any], attributes: List[str],
                    descriptions: Optional[Dict[str, str]], repeat: int) -> Dict[str, Any]:
    prompt = build_prompt(doc["text"], attributes, descriptions)
    t0 = time.time()
    response = ask(client, args.model, prompt, args.max_tokens)
    seconds = time.time() - t0
    choice = response.choices[0]
    content = choice.message.content or ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = LLMInteractionCallback._salvage_json(content)
    parsed = parsed if isinstance(parsed, dict) else None
    cells, not_found = {}, []
    for attr in attributes:
        value = (parsed or {}).get(attr)
        span = HumanLikeLLMCallback._locate(SimpleNamespace(text=doc["text"]), str(value)) if value else None
        if value and span is None:
            not_found.append(attr)  # an invented or rewritten value never becomes a cell
        cells[attr] = list(span) if span else None
    usage = response.usage
    return {"document": doc["name"], "repeat": repeat, "seconds": seconds,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "finish_reason": getattr(choice, "finish_reason", None), "parse_failure": parsed is None,
            "values": {a: (parsed or {}).get(a) for a in attributes}, "cells": cells, "not_found": not_found,
            "raw_response": content}


def score(records: List[Dict[str, Any]], docs: List[Dict[str, Any]], attributes: List[str]) -> pd.DataFrame:
    gold = {d["name"]: d for d in docs}
    rows = []
    for repeat in sorted({r["repeat"] for r in records}):
        by_doc = {r["document"]: r["cells"] for r in records if r["repeat"] == repeat}
        names = [d["name"] for d in docs if d["name"] in by_doc]
        for attr in attributes:
            cells = [by_doc[n][attr] for n in names]
            for rule_name, rule in RULES.items():
                c = counts(cells, names, attr, gold.get, rule)
                rows.append({"repeat": repeat, "attribute": attr, "rule": rule_name, "f1": f1_of(c), **c})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--dataset", default="aviation")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-description", action="store_true")
    parser.add_argument("--limit-docs", type=int, default=None, help="first N documents only, for a smoke test")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-tokens", type=int, default=16_384)
    args = parser.parse_args()
    workers = min(args.workers, MAX_WORKERS)

    attributes = list(dataset_module(args.dataset).ATTRIBUTES)
    descriptions = None if args.no_description else dataset_descriptions(args.dataset)
    docs = []
    for path in sorted((DATASETS_DIR / args.dataset / "documents").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        docs.append({"name": path.stem, "text": d["text"], "mentions": d.get("mentions", {})})
    docs = docs[:args.limit_docs] if args.limit_docs else docs

    out = OUT_ROOT / args.run_tag
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items()}
    config.update({"workers": workers, "attributes": attributes, "num_docs": len(docs)})
    (out / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    answers_path = out / "answers.jsonl"
    records = [json.loads(line) for line in answers_path.read_text(encoding="utf-8").splitlines()] \
        if answers_path.exists() else []
    done = {(r["document"], r["repeat"]) for r in records}
    todo = [(d, rep) for rep in range(args.repeats) for d in docs if (d["name"], rep) not in done]
    print(f"{len(done)} calls already done, {len(todo)} to send with {workers} parallel requests", flush=True)

    # a timeout would resend a long answer that the server keeps generating, so wait long and retry only in ask()
    client = OpenAI(base_url=args.base_url, api_key="not-needed", timeout=1800.0, max_retries=0)
    lock = threading.Lock()
    used = sum((r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in records)
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool, open(answers_path, "a", encoding="utf-8") as fh:
        futures = [pool.submit(answer_document, client, args, d, attributes, descriptions, rep) for d, rep in todo]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                rec = future.result()
            except Exception as e:  # the call is not stored, so a restart sends it again
                print(f"{i}/{len(futures)} calls, failed: {type(e).__name__}: {e}", flush=True)
                continue
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                records.append(rec)
                used += (rec["prompt_tokens"] or 0) + (rec["completion_tokens"] or 0)
            print(f"{i}/{len(futures)} calls, {used / 1e6:.2f} M tokens, {time.time() - t_start:.0f} s, last call "
                  f"{rec['seconds']:.0f} s", flush=True)
            if used > MAX_TOTAL_TOKENS:
                print("token budget used up, stopping")
                for f in futures:
                    f.cancel()
                break
    wall = time.time() - t_start

    scores = score(records, docs, attributes)
    scores.to_csv(out / "scores.csv", index=False)
    macro = scores.groupby(["rule", "repeat"])["f1"].mean().groupby("rule").agg(["mean", "std"])
    calls = pd.DataFrame(records)
    summary = {
        "calls": len(calls), "repeats": int(calls["repeat"].nunique()), "documents": len(docs),
        "wall_clock_seconds_this_session": wall, "workers": workers,
        "prompt_tokens_per_call": calls["prompt_tokens"].mean(),
        "completion_tokens_per_call": calls["completion_tokens"].mean(),
        "tokens_per_table": (calls["prompt_tokens"] + calls["completion_tokens"]).sum() / calls["repeat"].nunique(),
        "seconds_per_call_median": calls["seconds"].median(), "seconds_per_call_mean": calls["seconds"].mean(),
        "parse_failures": int(calls["parse_failure"].sum()),
        "truncated": int((calls["finish_reason"] == "length").sum()),
        "values_not_found_in_text": int(calls["not_found"].map(len).sum()),
        "values_given": int(calls["values"].map(lambda v: sum(x is not None for x in v.values())).sum()),
        "macro_f1": {rule: {"mean": row["mean"], "std": row["std"]} for rule, row in macro.iterrows()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    per_attr = scores[scores.rule == "half of longer (WannaDB)"].groupby("attribute")["f1"].mean().round(3)
    print(json.dumps(summary, indent=1))
    print(md(per_attr.reset_index(), index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
