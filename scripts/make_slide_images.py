"""Image panels for the presentation slides, drawn from the same data as the report figures.

Each MRI panel is one image without any text, so the slides can set their labels as large native
text: three dataset examples, the post-hoc Grad-CAM test cases (input and one map per model)
and the CBAM spatial weights on the rule-selected validation examples. The comparison panel
redraws the pre-specified paired comparison of report Figure 5 at slide size. No model is run here.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

import make_figures as mf

plt.rcParams["savefig.bbox"] = "standard"  # full-bleed squares, no padding
EXAMPLES = {"glioma": "Meningioma/110", "meningioma": "Glioma/2380", "pituitary": "Pituitary tumor/1285"}


def panel(path, sample_id, **overlay):
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1])
    mf.show_mri(ax, sample_id, **overlay)
    return ax, fig


def save(fig, path):
    fig.savefig(path, dpi=200, facecolor="black")
    plt.close(fig)


def examples(out):
    # Dataset examples: one coronal slice per class (original labels), with the tumor outline.
    for name, sid in EXAMPLES.items():
        _, fig = panel(out, sid)
        save(fig, out / f"example_{name}.png")


def comparison(out):
    """Mean paired difference from the baseline, its 95% patient-group bootstrap interval and the single seeds."""
    test = json.loads((mf.ANALYSIS / "test.json").read_text(encoding="utf-8"))
    comps = test["statistical_comparison"]["comparisons"]
    with plt.rc_context({"font.size": 24, "xtick.labelsize": 24, "ytick.labelsize": 28, "axes.labelsize": 24,
                         "legend.fontsize": 22, "axes.linewidth": 1.6, "xtick.major.width": 1.6}):
        fig = plt.figure(figsize=(15, 4.5))
        ax = fig.add_axes([0.13, 0.25, 0.65, 0.55])
        for row, model in enumerate(mf.MODELS[1:]):
            c, y = comps[model], 1 - row
            ax.plot([c["ci_lower"], c["ci_upper"]], [y, y], color=mf.COLOR[model], lw=7, solid_capstyle="round")
            ax.scatter(list(c["per_seed_macro_f1_difference"].values()), [y - 0.27] * 3, s=170,
                       color=mf.COLOR[model], alpha=0.55, linewidth=0)
            ax.scatter([c["mean_macro_f1_difference"]], [y], s=560, color=mf.COLOR[model], edgecolor="white",
                       linewidth=2.5, zorder=3)
            ax.text(0.0325, y, f"{mf.signed(c['mean_macro_f1_difference'])}\n[{mf.signed(c['ci_lower'])}, "
                    f"{mf.signed(c['ci_upper'])}]", va="center", fontsize=24, color=mf.INK2, linespacing=1.2)
        for tick in (-0.03, -0.015, 0.015, 0.03):  # light grid; the zero line is drawn separately and darker
            ax.axvline(tick, color=mf.GRID, lw=1.2, zorder=0)
        ax.axvline(0, color=mf.INK2, lw=2.4, zorder=1)
        ax.text(0.0007, 1.45, "no difference", ha="left", va="center", fontsize=22, color=mf.INK2, style="italic")
        ax.set_yticks([1, 0], [mf.LABEL[m] for m in mf.MODELS[1:]])
        ax.set_ylim(-0.5, 1.6)
        ax.set_xlim(-0.03, 0.03)
        ax.set_xticks([-0.03, -0.015, 0, 0.015, 0.03], ["−0.030", "−0.015", "0", "+0.015", "+0.030"])
        ax.set_xlabel("Difference from the baseline in test Macro-F1 (paired by seed)", labelpad=10)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=14)
        ax.legend(handles=[Line2D([], [], color=mf.MUTED, marker="o", lw=0, markersize=16, label="mean of 3 seeds"),
                           Line2D([], [], color=mf.MUTED, lw=7, label="95% bootstrap interval"),
                           Line2D([], [], color=mf.MUTED, marker="o", alpha=0.55, lw=0, markersize=11,
                                  label="single seed")],
                  loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=3, handlelength=1.6, columnspacing=1.8)
        fig.savefig(out / "slide_comparison.png", dpi=150, facecolor="white")
        plt.close(fig)


def gradcam_and_spatial(out):
    # Post-hoc Grad-CAM test cases: the input with the tumor outline, then the predicted-class map of each model.
    record = json.loads((mf.GRADCAM / "test_cases.json").read_text(encoding="utf-8"))
    heat = np.load(mf.GRADCAM / "test_cases_heatmaps.npz")
    for case in record["cases"]:
        _, fig = panel(out, case["sample_id"])
        save(fig, out / f"cam_{case['case']}_input.png")
        for model in mf.MODELS:
            _, fig = panel(out, case["sample_id"], heat=heat[f"{case['case']}|{model}|predicted"].astype(float))
            save(fig, out / f"cam_{case['case']}_{mf.SLUG[model]}.png")

    # CBAM spatial weights (seed 0) on the rule-selected validation examples, as in report figure 11.
    attention = json.loads((mf.ATTENTION_DIR / "validation_attention.json").read_text(encoding="utf-8"))
    data = np.load(mf.ATTENTION_DIR / "validation_attention.npz")
    run = f"formal-v2-{mf.SLUG['resnet50_cbam_terminal']}-seed0"
    sample_ids = list(data["sample_ids"])
    edges = np.round(np.linspace(0, 512, 8)).astype(int)
    for sid in attention["examples"]:
        cells = data[f"{run}|spatial"][sample_ids.index(sid)]
        scaled = (cells - cells.min()) / (cells.max() - cells.min())
        grid = np.asarray(Image.fromarray(scaled.astype(np.float32)).resize((512, 512), Image.Resampling.NEAREST))
        ax, fig = panel(out, sid, heat=grid, color=mf.ATTENTION, opacity=0.6)
        row, col = np.unravel_index(cells.argmax(), cells.shape)
        ax.add_patch(mf.Rectangle((edges[col] - 0.5, edges[row] - 0.5), edges[col + 1] - edges[col],
                                  edges[row + 1] - edges[row], fill=False, edgecolor="white", linewidth=2))
        cls = mf.SHORT[int(mf.SPLIT[sid]["class_idx"])].lower()
        save(fig, out / f"spatial_{cls}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=mf.PROJECT_ROOT / "results/figures/slides")
    parser.add_argument("--only", nargs="*", choices=["examples", "gradcam", "comparison"],
                        help="Panel groups to draw (default: all)")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    groups = {"examples": examples, "gradcam": gradcam_and_spatial, "comparison": comparison}
    for name in args.only or groups:
        groups[name](out)
    print("saved", len(list(out.glob("*.png"))), "slide images in", out)


if __name__ == "__main__":
    main()
