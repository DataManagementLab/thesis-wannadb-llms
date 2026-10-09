import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import run_grid as rg

HYBRID_ROOT = rg.RESULTS_DIR / "hybrid"
ROUNDS_SHOWN = [5, 10, 20, 30, 40]


def load(tag: str):
    root = HYBRID_ROOT / tag
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    mode = cfg.get("llm_mode", "list") + ("+desc" if cfg.get("include_description") else "")
    if cfg.get("vote_samples", 1) > 1:
        mode += f"+vote{cfg['vote_samples']}"
    if cfg.get("self_check_rounds", 0) > 0:
        mode += f"+check{cfg['self_check_rounds']}" + ("c" if cfg.get("cautious_review") else "")
    if cfg.get("show_accepted_in_prompt"):
        mode += "+accepted"
    if cfg.get("max_rounds", 40) != 40:
        mode += f", {cfg['max_rounds']} rounds"
    model = (cfg.get("model") or "DeepSeek-V4-Flash (old)").split("/")[-1]
    rows, llm = [], []
    for path in sorted((root / "checkpoints").glob("*.json")):
        u = json.loads(path.read_text(encoding="utf-8"))
        if u["variant"] == "A_no_feedback":
            continue
        name = f"{u['variant']} h={u['k']}"
        if u["k"] == cfg["max_rounds"]:
            name, label = "human only", "human only"
        elif u["k"] == 0:
            name, label = "LLM only", f"LLM only [{mode}, {model}]"
        else:
            label = f"{name} [{mode}, {model}]"
        for attr, per_round in u["per_round"].items():
            for r in per_round:
                rows.append({"tag": tag, "setting": label, "name": name, "seed": u["seed_idx"], "attribute": attr,
                             "round": r["round"], "f1": r["f1_score"]})
            for r in (u.get("llm_rounds") or {}).get(attr, []):
                llm.append({"tag": tag, "setting": label, "seed": u["seed_idx"], "attribute": attr, **r})
    return pd.DataFrame(rows), pd.DataFrame(llm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tags", nargs="+")
    parser.add_argument("--reference", nargs="*", default=[], help="older grids, only h=0, h=10 and h=40 are used")
    args = parser.parse_args()

    frames, llm_frames = [], []
    for tag in args.tags + args.reference:
        df, llm = load(tag)
        if tag in args.reference:
            df = df[df["name"].isin(["LLM only", "handoff h=10", "human only"])]
            llm = llm[llm["setting"].isin(df["setting"].unique())]
        frames.append(df)
        llm_frames.append(llm)
    df = pd.concat(frames)
    llm = pd.concat(llm_frames, ignore_index=True) if any(len(f) for f in llm_frames) else pd.DataFrame()

    seeds = set.intersection(*(set(df[df.tag == t].seed) for t in df.tag.unique()))
    df = df[df.seed.isin(seeds)]
    # human only is the same oracle in every grid, keep one copy
    human = df[df.name == "human only"]
    if len(human):
        first = human.tag.iloc[0]
        df = df[(df.name != "human only") | (df.tag == first)]
    print(f"seeds used: {sorted(seeds)}")

    # a setting counts only when all attributes of all seeds are there
    counts = df.groupby(["setting", "seed"]).attribute.nunique().unstack()
    complete = counts[(counts.notna().sum(axis=1) == len(seeds)) & (counts.min(axis=1) == 12)].index
    df = df[df.setting.isin(complete)]

    macro = (df.groupby(["setting", "seed", "round"]).f1.mean().reset_index()
             .groupby(["setting", "round"]).f1.agg(["mean", "std"]).reset_index())
    table = macro[macro["round"].isin(ROUNDS_SHOWN)].pivot(index="setting", columns="round", values="mean")
    print("\nMacro-F1 (12 attributes, mean over seeds) after r rounds")
    print(table.sort_values(40, ascending=False).round(3).to_string())

    final = df[df["round"] == 40].groupby(["setting", "attribute"]).f1.mean().unstack(0)
    print("\nF1 per attribute after 40 rounds")
    print(final.round(2).to_string())

    if len(llm):
        llm = llm[llm.seed.isin(seeds)]
        summary = llm.groupby("setting").agg(
            rounds=("message", "size"),
            agree=("agrees_with_gold", lambda s: (s == True).mean()),  # noqa: E712
            checked=("agrees_with_gold", lambda s: s.notna().mean()),
            custom=("message", lambda s: (s == "custom-match").mean()),
            no_match=("message", lambda s: (s == "no-match-in-document").mean()),
            tokens=("completion_tokens", lambda s: (s + llm.loc[s.index, "prompt_tokens"]).mean()),
            seconds=("seconds", "mean"),
        )
        print("\nLLM answers (agree = share of all LLM answers that agree with gold; for the old list prompt "
              "unchecked answers count as not agreeing)")
        print(summary.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
