
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "experiments" / "results" / "checkpoints"
INFO_DIR = ROOT / "info"

BASELINE_ORDER = ["A_no_feedback", "B_gold_oracle", "C_llm"]
BASELINE_LABEL = {
    "A_no_feedback": "A: No feedback (0 rounds)",
    "B_gold_oracle": "B: Gold oracle, 25 rounds (upper bound)",
    "C_llm": "C: LLM DeepSeek, 25 rounds",
}
BASELINE_SHORT = {
    "A_no_feedback": "A: No feedback\n(0 rounds)",
    "B_gold_oracle": "B: Gold oracle\n(25 rounds)",
    "C_llm": "C: LLM DeepSeek\n(25 rounds)",
}
BASELINE_COLOR = {
    "A_no_feedback": "#b0b0b0",
    "B_gold_oracle": "#2e7d32",
    "C_llm": "#1565c0",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    args = ap.parse_args()

    files = sorted(CHECKPOINT_DIR.glob(f"{args.run_tag}_*_seed*.json"))
    records = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    attributes = sorted({row["attribute"] for r in records for row in r["scores"]})

    def mean_f1(baseline, attribute):
        vals = [row["f1_score"] for r in records if r["baseline"] == baseline
                for row in r["scores"] if row["attribute"] == attribute]
        return sum(vals) / len(vals) if vals else 0.0

    #Plot 1: grouped bar chart, F1 per attribute per baseline
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(attributes))
    width = 0.26
    for i, baseline in enumerate(BASELINE_ORDER):
        vals = [mean_f1(baseline, a) for a in attributes]
        ax.bar(x + (i - 1) * width, vals, width, label=BASELINE_LABEL[baseline],
               color=BASELINE_COLOR[baseline])
    ax.set_xticks(x)
    ax.set_xticklabels(attributes, rotation=40, ha="right")
    ax.set_ylabel("Mean F1 over 5 seeds")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out1 = INFO_DIR / "overnight_f1_per_attribute.png"
    fig.savefig(out1, dpi=150)
    print(f"Saved {out1}")

    # Plot 2: mean F1 overview + timing
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    overall = {b: sum(mean_f1(b, a) for a in attributes) / len(attributes) for b in BASELINE_ORDER}
    axes[0].bar([BASELINE_SHORT[b] for b in BASELINE_ORDER],
                [overall[b] for b in BASELINE_ORDER],
                color=[BASELINE_COLOR[b] for b in BASELINE_ORDER])
    for i, b in enumerate(BASELINE_ORDER):
        axes[0].text(i, overall[b] + 0.02, f"{overall[b]:.3f}", ha="center")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Mean F1 across all 12 attributes")
    axes[0].tick_params(axis="x", rotation=0, labelsize=8)


    tm, tl = [], []
    for b in BASELINE_ORDER:
        rs = [r for r in records if r["baseline"] == b]
        tm.append(sum(r["t_wannadb_matching_seconds"] for r in rs) / len(rs))
        tl.append(sum(r["t_llm_inference_seconds"] for r in rs) / len(rs))
    xb = np.arange(len(BASELINE_ORDER))
    axes[1].bar(xb, tm, label="t_wannadb_matching (WannaDB's own computation)", color="#455a64")
    axes[1].bar(xb, tl, bottom=tm, label="t_llm_inference (waiting on DeepSeek)", color="#ef6c00")
    for i in range(len(BASELINE_ORDER)):
        axes[1].text(i, tm[i], f" {tm[i]:.0f}s", va="bottom", ha="center", fontsize=8)
        if tl[i] > 0:
            axes[1].text(i, tm[i] + tl[i], f" {tl[i]:.0f}s", va="bottom", ha="center", fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_ylim(1, 30000)
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels([BASELINE_SHORT[b] for b in BASELINE_ORDER], rotation=0, fontsize=8)
    axes[1].set_ylabel("Mean total time per run, log scale (s)\n(one run = all 12 attributes)")
    axes[1].legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out2 = INFO_DIR / "overnight_overview_and_timing.png"
    fig.savefig(out2, dpi=150)
    print(f"Saved {out2}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
