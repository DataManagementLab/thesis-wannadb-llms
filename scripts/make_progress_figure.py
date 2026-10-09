
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
import hybrid_analysis as ha

OUT = ROOT / "info" / "hybrid_figures"
SEEDS = [0, 1, 2]
HUMAN, LLM, OLD = "#1baf7a", "#2a78d6", "#9aa5b1"


def curve(tag: str, variant: str, h: int):
    run = ha.load_run(tag)
    df = ha.per_round_frame(run, complete_only=False)
    df = df[(df.variant == variant) & (df.h == h) & (df.seed.isin(SEEDS))]
    return df.groupby(["seed", "round"]).f1.mean().groupby("round").mean()


def main() -> int:
    human = curve("glm_direct_desc_r100_20261008", "handoff", 100)
    best = curve("glm_direct_desc_r100_20261008", "handoff", 0)
    old = curve("hybrid40_20260914", "handoff", 0)
    human40 = curve("hybrid40_20260914", "handoff", 40)[40]

    ladder = [
        ("list prompt\n(DeepSeek)", old[40], OLD),
        ("pick a\nnugget", curve("glm_nuggets_20261008", "handoff", 0)[40], LLM),
        ("+ own\ntext", curve("glm_custom_20261008", "handoff", 0)[40], LLM),
        ("+ check on\nfull text", curve("glm_direct_20261008", "handoff", 0)[40], LLM),
        ("+ attribute\ndescription", curve("glm_direct_desc_20261008", "handoff", 0)[40], LLM),
        ("human\nonly", human40, HUMAN),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax1.plot(human.index, human.values, color=HUMAN, lw=2.2, label="human only")
    ax1.plot(best.index, best.values, color=LLM, lw=2.2, label="LLM only, new prompt")
    ax1.plot(old.index, old.values, color=OLD, lw=2.2, ls="--", label="LLM only, first prompt (DeepSeek)")
    ax1.axhline(human40, color=HUMAN, lw=1, ls=":")
    cross = next(r for r, v in best.items() if v >= human40)
    ax1.annotate(f"LLM only passes human@40\nafter {cross} rounds", xy=(cross, best[cross]), xytext=(55, 0.6),
                 arrowprops={"arrowstyle": "->", "color": "#444"}, fontsize=9)
    ax1.annotate("human@40", xy=(2, human40 + 0.012), fontsize=8, color=HUMAN)
    ax1.set_xlabel("feedback rounds per attribute")
    ax1.set_ylabel("macro-F1 (12 attributes)")
    ax1.set_title("(a) Feedback rounds", loc="left", fontsize=11)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1.02)
    ax1.legend(frameon=False, fontsize=9, loc="lower right")

    labels = [l for l, _, _ in ladder]
    values = [v for _, v, _ in ladder]
    ax2.bar(range(len(ladder)), values, color=[c for _, _, c in ladder], width=0.65)
    for i, v in enumerate(values):
        ax2.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
    ax2.set_xticks(range(len(ladder)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylim(0.5, 0.9)
    ax2.set_ylabel("macro-F1 after 40 rounds")
    ax2.set_title("(b) LLM only, prompt steps", loc="left", fontsize=11)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"prompt_progress.{ext}", dpi=200)
    print(f"wrote {OUT / 'prompt_progress.png'}; LLM only passes human@40 ({human40:.3f}) after {cross} rounds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
