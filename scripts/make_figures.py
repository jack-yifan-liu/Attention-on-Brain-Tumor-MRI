"""Draw the final-report and presentation figures from saved data and results.

No model is trained or evaluated here. Inputs are the saved split, raw images and
masks, the formal run histories, the one-time test analysis, the post-hoc Grad-CAM
maps written by scripts/gradcam_test_cases.py and the validation attention weights
written by scripts/attention_weights.py. The final report covers the baseline, SE
and CBAM models; the PSANet adaptation is not drawn.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import NullLocator
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data/raw/sciencedb/data"
ANALYSIS = PROJECT_ROOT / "results/analysis/pid-numeric-v2-final"
GRADCAM = PROJECT_ROOT / "results/figures/gradcam"
ATTENTION_DIR = PROJECT_ROOT / "results/figures/attention"

MODELS = ("resnet50", "resnet50_se_terminal", "resnet50_cbam_terminal")
LABEL = {"resnet50": "Baseline", "resnet50_se_terminal": "+ SE", "resnet50_cbam_terminal": "+ CBAM"}
SLUG = {"resnet50": "resnet50", "resnet50_se_terminal": "resnet50-se-terminal",
        "resnet50_cbam_terminal": "resnet50-cbam-terminal"}
# Categorical slots 1-3 of the reference palette (validated for all pairs, light mode).
COLOR = {"resnet50": "#2a78d6", "resnet50_se_terminal": "#eb6834", "resnet50_cbam_terminal": "#1baf7a"}
SHORT = ("Glioma", "Meningioma", "Pituitary")  # class names in label order
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BOX, BOX_EDGE = "#f3f2ee", "#dddcd5"
ATTENTION, ATTENTION_FILL = "#6250d6", "#ebe8fb"
MASK_LINE, HEAT = "#2fe07a", "#e34948"
WRONG, RIGHT = "#c62f2f", "#006300"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
WIDTH = 6.3  # inches: A4 text width with 2.54 cm margins

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
})


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


SPLIT = {row["sample_id"]: row for row in read_csv(PROJECT_ROOT / "data/metadata_pid_numeric/data_split.csv")}


def load_mri(sample_id):
    with Image.open(DATA_ROOT / SPLIT[sample_id]["mri_path"]) as image:
        return np.asarray(image.convert("L"))


def load_mask(sample_id):
    with Image.open(DATA_ROOT / SPLIT[sample_id]["mask_path"]) as image:
        return np.asarray(image.convert("L")) >= 128


def show_mri(ax, sample_id, outline=True, heat=None, color=HEAT, opacity=0.62):
    ax.imshow(load_mri(sample_id), cmap="gray", vmin=0, vmax=255, interpolation="antialiased")
    if heat is not None:
        rgba = np.zeros((*heat.shape, 4))
        rgba[..., :3] = matplotlib.colors.to_rgb(color)
        rgba[..., 3] = np.clip(heat, 0, 1) * opacity
        ax.imshow(rgba, interpolation="antialiased")
    if outline:
        ax.contour(load_mask(sample_id).astype(float), levels=[0.5], colors=[MASK_LINE], linewidths=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save(fig, out, name):
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print("saved", name)


def box(ax, x, y, w, h, title=None, lines=(), fill=BOX, edge=BOX_EDGE, size=7.2, title_size=7.8, align="left"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=fill, edgecolor=edge, linewidth=0.8))
    tx = x + 0.08 if align == "left" else x + w / 2
    ha = "left" if align == "left" else "center"
    ty = y + 0.08
    if title:
        ax.text(tx, ty, title, ha=ha, va="top", fontsize=title_size, fontweight="bold", color=INK)
        ty += 0.19
    if lines:
        ax.text(tx, ty, "\n".join(lines), ha=ha, va="top", fontsize=size, color=INK2, linespacing=1.32)


def arrow(ax, start, end, color=MUTED):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9,
                                                          mutation_scale=8, shrinkA=0, shrinkB=0))


def signed(value, digits=3):
    return f"{value:+.{digits}f}".replace("-", "−")  # typographic minus, as on the axes


def canvas(width, height):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    return fig, ax


# ---------------------------------------------------------------- Figure 1
def fig_dataset_examples(out):
    # Chosen by eye from training images for clear examples of each plane; labels are the original ones.
    rows = [("Glioma", ["Meningioma/79", "Meningioma/110", "Meningioma/554"]),
            ("Meningioma", ["Pituitary tumor/1812", "Glioma/2380", "Glioma/2631"]),
            ("Pituitary tumor", ["Glioma/2749", "Pituitary tumor/1285", "Pituitary tumor/1621"])]
    fig, axes = plt.subplots(3, 3, figsize=(5.2, 5.45))
    fig.subplots_adjust(left=0.07, right=1, top=0.95, bottom=0, wspace=0.03, hspace=0.03)
    for r, (cls, samples) in enumerate(rows):
        for c, sid in enumerate(samples):
            assert SPLIT[sid]["class_name"] == cls and SPLIT[sid]["split"] == "train"
            show_mri(axes[r, c], sid)
            if r == 0:
                axes[r, c].set_title(("Axial", "Coronal", "Sagittal")[c], fontsize=8.5, pad=4)
        axes[r, 0].text(-0.05, 0.5, cls, transform=axes[r, 0].transAxes, rotation=90, ha="right", va="center",
                        fontsize=8.5, fontweight="bold")
    save(fig, out, "fig01_dataset_examples")


# ---------------------------------------------------------------- Figure 2
def fig_data_flowchart(out):
    fig, ax = canvas(WIDTH, 4.0)
    w, g = 1.95, 0.175
    xs = [0.05, 0.05 + w + g, 0.05 + 2 * (w + g)]
    for y, text in ((0.06, "CLEAN AND RELABEL"), (1.62, "PREVENT LEAKAGE")):
        ax.text(0.05, y, text, fontsize=7.5, fontweight="bold", color=INK2, va="top")
    y1, h1 = 0.30, 1.14
    box(ax, xs[0], y1, w, h1, "Two releases, same 3,064 slices",
        ["ScienceDB (used as input):", "PNG images and tumor masks;", "class given by folder name", "",
         "Figshare v8 (original release):", "label and patient code per slice"])
    box(ax, xs[1], y1, w, h1, "1  Match every image",
        ["Each PNG matched to its original", "record by image content:", "1,555 pixel-identical, 1,509 via an",
         "identical tumor mask", "→ 3,064 one-to-one matches"])
    box(ax, xs[2], y1, w, h1, "2  Relabel",
        ["1,768 folder labels (57.7%)", "contradict the original labels", "→ original labels used throughout",
         "Tumor position agrees with the", "original labels (Figure 4)"])
    for a, b in ((0, 1), (1, 2)):
        arrow(ax, (xs[a] + w, y1 + h1 / 2), (xs[b], y1 + h1 / 2))
    y2, h2 = 1.86, 1.36
    box(ax, xs[0], y2, w, h2, "3  Group patients",
        ["233 patient codes; codes with the", "same number (42 codes, 18 numbers,", "e.g. MR024780B, D, E and G) are",
         "treated as one patient", "→ 209 patient groups"])
    box(ax, xs[1], y2, w, h2, "4  Split by patient group",
        ["Within each class, 70 / 15 / 15%", "of groups (split seed 42):"])
    for k, (name, imgs, grps) in enumerate((("Training", "2,099", "147"), ("Validation", "544", "31"),
                                             ("Test", "421", "31"))):
        yy = y2 + 0.58 + k * 0.235
        ax.add_patch(FancyBboxPatch((xs[1] + 0.08, yy), w - 0.16, 0.2, boxstyle="round,pad=0,rounding_size=0.04",
                                    facecolor="white", edgecolor=BOX_EDGE, linewidth=0.6))
        ax.text(xs[1] + 0.15, yy + 0.1, name, fontsize=7.2, fontweight="bold", va="center", color=INK)
        ax.text(xs[1] + w - 0.15, yy + 0.1, f"{imgs} images · {grps} groups", fontsize=7.2, va="center",
                ha="right", color=INK2)
    box(ax, xs[2], y2, w, h2, "5  Check for leakage",
        ["No image, patient code or group", "appears in two splits", "No two images are pixel-identical",
         "Near-duplicate screen: highest", "cross-split correlation 0.93", "(review threshold 0.99)"])
    for a, b in ((0, 1), (1, 2)):
        arrow(ax, (xs[a] + w, y2 + h2 / 2), (xs[b], y2 + h2 / 2))
    # Elbow connector from step 2 down to step 3; it drops to the right of the row label.
    xc, xd, ye = xs[2] + w / 2, xs[0] + w - 0.25, 1.53
    ax.plot([xc, xc, xd], [y1 + h1, ye, ye], color=MUTED, lw=0.9, solid_capstyle="butt")
    arrow(ax, (xd, ye), (xd, y2))
    yf = y2 + h2 + 0.14
    box(ax, 0.05, yf, 3 * w + 2 * g, 0.6, "6  Fixed for every model",
        ["One saved split, verified by checksum, is shared by all runs; augmentation is applied only after "
         "splitting;", "the test set is used once, after training."])
    arrow(ax, (xs[2] + w / 2, y2 + h2), (xs[2] + w / 2, yf))
    save(fig, out, "fig02_data_flowchart")


# ---------------------------------------------------------------- Figure 3
def fig_label_errors(out):
    examples = ["Glioma/812", "Glioma/915", "Pituitary tumor/1157", "Pituitary tumor/1224"]
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.2))
    fig.subplots_adjust(left=0, right=1, top=0.80, bottom=0.08, wspace=0.04)
    for ax, sid in zip(axes, examples):
        row = SPLIT[sid]
        assert row["directory_class"] != row["class_name"] and row["split"] == "train"
        show_mri(ax, sid)
        folder = row["directory_class"].replace(" tumor", "")
        original = row["class_name"].replace(" tumor", "")
        ax.text(0.5, 1.17, f"ScienceDB folder: {folder}", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.4, color=WRONG)
        ax.text(0.5, 1.03, f"Original label: {original}", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.4, color=RIGHT, fontweight="bold")
        ax.text(0.5, -0.03, f"patient {row['patient_id']}", transform=ax.transAxes, ha="center", va="top",
                fontsize=6.8, color=MUTED)
    save(fig, out, "fig03_label_errors")


# ---------------------------------------------------------------- Figure 4
def fig_architecture(out):
    fig, ax = canvas(WIDTH, 4.6)
    nodes = [("Input MRI", "224×224×3"), ("Stem\nconv + pool", "56×56×64"),
             ("Stage 1", "56×56×256"), ("Stage 2", "28×28×512"),
             ("Stage 3", "14×14×1024"), ("Stage 4", "7×7×2048"),
             ("Attention\nmodule", "7×7×2048"), ("Global\navg. pool", "2048"), ("Linear", "3 classes")]
    nw, ng, y, h = 0.61, 0.083, 0.42, 0.52
    xs = [0.05 + k * (nw + ng) for k in range(len(nodes))]
    for k, (name, shape) in enumerate(nodes):
        attention = k == 6
        ax.add_patch(FancyBboxPatch((xs[k], y), nw, h, boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=ATTENTION_FILL if attention else BOX,
                                    edgecolor=ATTENTION if attention else BOX_EDGE, linewidth=1.2 if attention else 0.8))
        ax.text(xs[k] + nw / 2, y + h / 2, name, ha="center", va="center", fontsize=6.9, fontweight="bold",
                color=ATTENTION if attention else INK, linespacing=1.15)
        ax.text(xs[k] + nw / 2, y + h + 0.07, shape, ha="center", va="top", fontsize=6.3, color=MUTED)
        if k:
            arrow(ax, (xs[k - 1] + nw, y + h / 2), (xs[k], y + h / 2))
    # Bracket over the pretrained backbone.
    bx0, bx1, by = xs[1], xs[5] + nw, 0.30
    ax.plot([bx0, bx0, bx1, bx1], [by + 0.06, by, by, by + 0.06], color=AXIS, lw=0.8)
    ax.text((bx0 + bx1) / 2, by - 0.04, "ImageNet-pretrained ResNet-50 backbone (identical in all models)",
            ha="center", va="bottom", fontsize=7, color=INK2)
    ax.text(xs[6] + nw / 2, by - 0.04, "none (baseline),\nSE or CBAM", ha="center", va="bottom", fontsize=6.6,
            color=ATTENTION, linespacing=1.1)
    # Expansion lines from the attention node to the two detail panels.
    py, ph = 1.52, 2.98
    left, right = (0.05, 3.08), (3.22, 6.25)
    for (x0, x1) in (left, right):
        ax.plot([xs[6] + nw / 2, (x0 + x1) / 2], [y + h + 0.24, py - 0.02], color=ATTENTION, lw=0.6, alpha=0.6)
        ax.add_patch(FancyBboxPatch((x0, py), x1 - x0, ph, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor="white", edgecolor=ATTENTION, linewidth=0.8))

    def flow(x0, x1, title, steps, skips):
        ax.text(x0 + 0.1, py + 0.1, title, fontsize=7.8, fontweight="bold", color=ATTENTION, va="top")
        cx, bw = (x0 + x1) / 2 + 0.12, (x1 - x0) - 0.62
        heights = [0.40 if kind == "big" else 0.22 for _, kind in steps]
        ys = []
        # Centre the stack in the panel area below the title.
        yy = py + 0.3 + (ph - 0.3 - sum(heights) - 0.13 * (len(steps) - 1)) / 2
        for (text, kind), bh in zip(steps, heights):
            fill = "white" if kind == "io" else (ATTENTION_FILL if kind == "op" else BOX)
            ax.add_patch(FancyBboxPatch((cx - bw / 2, yy), bw, bh, boxstyle="round,pad=0,rounding_size=0.04",
                                        facecolor=fill, edgecolor=BOX_EDGE if kind != "op" else ATTENTION,
                                        linewidth=0.7))
            ax.text(cx, yy + bh / 2, text, ha="center", va="center", fontsize=6.6, color=INK, linespacing=1.2)
            ys.append((yy, bh))
            yy += bh + 0.13
        for (ya, ha_), (yb, _) in zip(ys, ys[1:]):
            arrow(ax, (cx, ya + ha_), (cx, yb))
        # Skip path: the feature map entering each attention step is the one that gets re-weighted.
        sx = cx - bw / 2 - 0.16
        for source, target in skips:
            src_y = ys[source][0] + ys[source][1] / 2
            dst_y = ys[target][0] + ys[target][1] / 2
            ax.plot([cx - bw / 2, sx, sx], [src_y, src_y, dst_y], color=MUTED, lw=0.7)
            arrow(ax, (sx, dst_y), (cx - bw / 2, dst_y))

    flow(*left, "SE block  (+0.53 M parameters)", [
        ("Feature map 2048×7×7", "io"),
        ("Squeeze: global average pool → 2048", "step"),
        ("Excitation: Linear 2048→128, ReLU\nLinear 128→2048, sigmoid", "big"),
        ("Multiply each channel by its weight", "op"),
        ("Output 2048×7×7", "io")], skips=[(0, 3)])
    flow(*right, "CBAM block  (+0.53 M parameters)", [
        ("Feature map 2048×7×7", "io"),
        ("Channel attention: average and max pool\n→ shared MLP (2048→128→2048) → sigmoid", "big"),
        ("Multiply each channel by its weight", "op"),
        ("Spatial attention: channel mean and max\n→ 7×7 convolution, BN → sigmoid", "big"),
        ("Multiply each position by its weight", "op"),
        ("Output 2048×7×7", "io")], skips=[(0, 2), (2, 4)])
    save(fig, out, "fig04_architecture")


# ---------------------------------------------------------------- Figure 5
def fig_training_curves(out):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.25), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    alphas = (1.0, 0.62, 0.34)
    for ax, model in zip(axes, MODELS):
        for seed in range(3):
            run = PROJECT_ROOT / f"results/formal/formal-v2-{SLUG[model]}-seed{seed}"
            history = json.loads((run / "history.json").read_text(encoding="utf-8"))
            best = json.loads((run / "summary.json").read_text(encoding="utf-8"))["best_epoch"]
            f1 = [e["validation"]["macro_f1"] for e in history]
            epochs = range(1, len(f1) + 1)
            ax.plot(epochs, f1, color=COLOR[model], alpha=alphas[seed], lw=1.1, solid_joinstyle="round")
            ax.scatter([best], [f1[best - 1]], s=22, color=COLOR[model], alpha=alphas[seed], edgecolor="white",
                       linewidth=1, zorder=3)
        ax.set_title(LABEL[model], color=INK)
        ax.set_xlim(0.5, 30.5)
        ax.set_ylim(0.80, 1.0)
        ax.set_xticks([1, 10, 20, 30])
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Macro-F1")
    handles = [Line2D([], [], color=INK2, alpha=a, lw=1.1, label=f"seed {s}") for s, a in enumerate(alphas)]
    handles.append(Line2D([], [], color=INK2, marker="o", lw=0, markersize=4.5, markeredgecolor="white",
                          label="selected epoch"))
    axes[2].legend(handles=handles, loc="lower right", handlelength=1.6, borderaxespad=0.2)
    save(fig, out, "fig05_training_curves")


# ---------------------------------------------------------------- Figure 6
def fig_test_results(out, test):
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.3), gridspec_kw={"width_ratios": [1, 1.35]})
    fig.subplots_adjust(wspace=0.42)
    for k, model in enumerate(MODELS):
        vals = [test["run_results"][f"formal-v2-{SLUG[model]}-seed{s}"]["conditions"]["clean"]["metrics"]["macro_f1"]
                for s in range(3)]
        a.scatter([k - 0.08, k, k + 0.08], vals, s=26, color=COLOR[model], edgecolor="white", linewidth=1, zorder=3)
        mean = statistics.mean(vals)
        a.plot([k - 0.24, k + 0.24], [mean, mean], color=INK, lw=1.2)
        a.text(k + 0.28, mean, f"{mean:.3f}", va="center", fontsize=7, color=INK2)
    a.set_xticks(range(3), [LABEL[m] for m in MODELS])
    a.set_xlim(-0.5, 2.75)
    a.set_ylim(0.87, 0.93)
    a.set_ylabel("Test Macro-F1")
    a.yaxis.grid(True, color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.set_title("Test performance per seed", loc="left")
    a.legend(handles=[Line2D([], [], color=MUTED, marker="o", lw=0, markersize=4.5, label="one seed"),
                      Line2D([], [], color=INK, lw=1.2, label="mean")], loc="upper left", borderaxespad=0.1,
             handlelength=1.2)
    comps = test["statistical_comparison"]["comparisons"]
    for row, model in enumerate(MODELS[1:]):
        c = comps[model]
        yv = 1 - row
        b.plot([c["ci_lower"], c["ci_upper"]], [yv, yv], color=COLOR[model], lw=2.2, solid_capstyle="round")
        diffs = list(c["per_seed_macro_f1_difference"].values())
        b.scatter(diffs, [yv - 0.2] * 3, s=12, color=COLOR[model], alpha=0.55, linewidth=0)
        b.scatter([c["mean_macro_f1_difference"]], [yv], s=34, color=COLOR[model], edgecolor="white", linewidth=1.2,
                  zorder=3)
        b.text(0.0315, yv, f"{signed(c['mean_macro_f1_difference'])}\n[{signed(c['ci_lower'])}, "
               f"{signed(c['ci_upper'])}]", va="center", fontsize=6.8, color=INK2, linespacing=1.15)
    b.axvline(0, color=AXIS, lw=0.9, zorder=0)
    b.set_yticks([1, 0], [f"{LABEL[m]} − Baseline" for m in MODELS[1:]])
    b.set_ylim(-0.45, 1.75)
    b.set_xlim(-0.03, 0.03)
    b.set_xticks([-0.03, -0.015, 0, 0.015, 0.03])
    b.set_xlabel("Difference in test Macro-F1 (paired by seed)")
    b.xaxis.grid(True, color=GRID, lw=0.6)
    b.set_axisbelow(True)
    b.spines["left"].set_visible(False)
    b.tick_params(axis="y", length=0)
    b.set_title("Pre-specified comparison: no consistent difference", loc="left")
    b.legend(handles=[Line2D([], [], color=MUTED, marker="o", lw=0, markersize=4.5, label="mean of 3 seeds"),
                      Line2D([], [], color=MUTED, lw=2.2, label="95% bootstrap interval"),
                      Line2D([], [], color=MUTED, marker="o", alpha=0.55, lw=0, markersize=3, label="single seed")],
             loc="upper center", ncol=3, fontsize=6.6, handlelength=1.3, columnspacing=1.2, borderaxespad=0.1,
             frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    save(fig, out, "fig06_test_results")


# ---------------------------------------------------------------- Figure 7
def fig_confusion(out, test):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.35))
    fig.subplots_adjust(wspace=0.12, left=0.13, right=0.99, bottom=0.22, top=0.88)
    for ax, model in zip(axes, MODELS):
        mats = [np.array(test["run_results"][f"formal-v2-{SLUG[model]}-seed{s}"]["conditions"]["clean"]["metrics"]
                         ["confusion_matrix"], dtype=float) for s in range(3)]
        mean = np.mean(mats, axis=0)
        share = mean / mean.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=BLUE_RAMP, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                dark = share[i, j] > 0.55
                ax.text(j, i - 0.12, f"{share[i, j] * 100:.0f}%", ha="center", va="center", fontsize=7.6,
                        fontweight="bold", color="white" if dark else INK)
                ax.text(j, i + 0.2, f"({mean[i, j]:.1f})", ha="center", va="center", fontsize=6.3,
                        color="white" if dark else INK2)
        ax.set_xticks(range(3), SHORT, rotation=25, ha="right")
        ax.set_yticks(range(3), SHORT if ax is axes[0] else [""] * 3)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(LABEL[model], color=INK)
        ax.set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    save(fig, out, "fig07_confusion_matrices")


# ---------------------------------------------------------------- Figures 8 and 9
def fig_gradcam_cases(out, kind, name):
    record = json.loads((GRADCAM / "test_cases.json").read_text(encoding="utf-8"))
    heat = np.load(GRADCAM / "test_cases_heatmaps.npz")
    cases = [c for c in record["cases"] if c["kind"] == kind]
    fig, axes = plt.subplots(len(cases), 4, figsize=(WIDTH, 5.35))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.94, bottom=0.075, wspace=0.03, hspace=0.24)
    for c, title in enumerate(("MRI and tumor outline", *(LABEL[m] for m in MODELS))):
        axes[0, c].set_title(title, fontsize=8.2, pad=4, color=INK)
    for r, case in enumerate(cases):
        show_mri(axes[r, 0], case["sample_id"])
        code = "numeric" if case["code_format"] == "numeric" else "MR-prefixed"
        axes[r, 0].text(0.5, -0.03, f"patient {case['patient_id']} ({code} code)", transform=axes[r, 0].transAxes,
                        ha="center", va="top", fontsize=6.4, color=MUTED)
        axes[r, 0].text(-0.06, 0.5, f"True: {SHORT[case['true_label']]}", transform=axes[r, 0].transAxes,
                        rotation=90, ha="right", va="center", fontsize=7.6, fontweight="bold")
        for c, model in enumerate(MODELS, start=1):
            info = case["seed0"][model]
            pred = info["predicted_label"]
            show_mri(axes[r, c], case["sample_id"], outline=True, heat=heat[f"{case['case']}|{model}|predicted"].astype(float))
            ok = pred == case["true_label"]
            axes[r, c].text(0.5, -0.03, f"Predicted: {SHORT[pred]} ({info['probabilities'][pred]:.2f})",
                            transform=axes[r, c].transAxes, ha="center", va="top", fontsize=6.6,
                            color=RIGHT if ok else WRONG, fontweight="bold")
    heat_bar = fig.add_axes([0.40, 0.022, 0.30, 0.014])
    gradient = np.ones((1, 256, 4))
    gradient[..., :3] = matplotlib.colors.to_rgb(HEAT)
    gradient[..., 3] = np.linspace(0, 0.62, 256)
    heat_bar.imshow(np.ones((1, 256)), cmap="gray", vmin=0, vmax=2, aspect="auto")
    heat_bar.imshow(gradient, aspect="auto")
    heat_bar.set_xticks([0, 255], ["low", "high"])
    heat_bar.set_yticks([])
    heat_bar.tick_params(length=0, labelsize=6.5)
    for spine in heat_bar.spines.values():
        spine.set_visible(False)
    heat_bar.text(-0.03, 0.5, "Grad-CAM for the predicted class", transform=heat_bar.transAxes, ha="right",
                  va="center", fontsize=6.6, color=INK2)
    heat_bar.text(1.14, 0.5, "tumor outline", transform=heat_bar.transAxes, ha="left", va="center",
                  fontsize=6.6, color=INK2)
    heat_bar.plot([1.05, 1.12], [0.5, 0.5], transform=heat_bar.transAxes, color=MASK_LINE, lw=1.4, clip_on=False)
    save(fig, out, name)


# ---------------------------------------------------------------- Figure 10
def fig_cam_overlap(out):
    rows = [r for r in read_csv(ANALYSIS / "validation_cam_localization.csv") if r["model"] in MODELS and r["enrichment"]]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.75))
    fig.subplots_adjust(right=0.72)
    data, above = [], []
    for model in MODELS:
        vals = np.array([float(r["enrichment"]) for r in rows if r["model"] == model])
        data.append(np.clip(vals, 1 / 64, None))  # keeps zeros on the log axis; percentiles shown are unaffected
        above.append((vals > 1).mean())
    pos = [2, 1, 0]
    parts = ax.boxplot(data, positions=pos, vert=False, widths=0.5, whis=(5, 95), showfliers=False, patch_artist=True,
                       medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=INK2, lw=0.8),
                       capprops=dict(color=INK2, lw=0.8))
    for patch, model in zip(parts["boxes"], MODELS):
        patch.set_facecolor(matplotlib.colors.to_rgba(COLOR[model], 0.25))
        patch.set_edgecolor(COLOR[model])
    for p, vals, frac in zip(pos, data, above):
        ax.text(1.03, p, f"median {np.median(vals):.1f}×\nabove 1 in {frac * 100:.0f}% of maps",
                transform=ax.get_yaxis_transform(), va="center", fontsize=6.8, color=INK2, linespacing=1.25)
    ax.axvline(1, color=AXIS, lw=0.9)
    ax.text(1.06, 2.42, "1 = uniform map", fontsize=6.6, color=MUTED, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xlim(1 / 8, 32)
    ax.set_xticks([1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16, 32], ["1/8", "1/4", "1/2", "1", "2", "4", "8", "16", "32"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-0.5, 2.75)
    ax.set_yticks(pos, [LABEL[m] for m in MODELS])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Grad-CAM enrichment in the tumor mask (validation, 3 seeds, log scale)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, out, "fig10_gradcam_overlap")


# ---------------------------------------------------------------- Figure 11
def fig_robustness(out, test):
    families = [("rician", "Rician-like noise", "σ 0.02 / 0.05 / 0.10"),
                ("blur", "Gaussian blur", "σ 0.5 / 1 / 2 px"),
                ("intensity", "Intensity scaling", "±10 / 25 / 40%")]
    agg = test["mean_and_sample_std_by_model"]
    test_rows = [r for r in SPLIT.values() if r["split"] == "test"]
    glioma_share = sum(r["class_name"] == "Glioma" for r in test_rows) / len(test_rows)
    # Macro-F1 when every image is called glioma: only the glioma F1, 2p / (1 + p), is non-zero.
    all_glioma = 2 * glioma_share / (1 + glioma_share) / 3
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, wspace=0.1, top=0.78)
    for ax, (fam, title, levels) in zip(axes, families):
        conds = ["clean", f"{fam}_mild", f"{fam}_moderate", f"{fam}_severe"]
        for k, model in enumerate(MODELS):
            m = [agg[model]["conditions"][c]["macro_f1"]["mean"] for c in conds]
            sd = [agg[model]["conditions"][c]["macro_f1"]["std"] for c in conds]
            xs = np.arange(4) + (k - 1) * 0.1
            ax.errorbar(xs, m, yerr=sd, color=COLOR[model], lw=1.4, elinewidth=0.8, capsize=0, marker="o",
                        markersize=3.8, markeredgecolor="white", markeredgewidth=0.8, label=LABEL[model])
        if fam == "rician":
            ax.axhline(all_glioma, color=AXIS, lw=0.9, zorder=0)
            ax.text(-0.25, all_glioma + 0.013, "all predicted glioma", fontsize=6.2, color=MUTED, ha="left",
                    va="bottom")
        ax.set_title(f"{title}\n{levels}", fontsize=8, color=INK, linespacing=1.3)
        ax.set_xticks(range(4), ["Clean", "Mild", "Moderate", "Severe"], rotation=0)
        ax.set_xlim(-0.35, 3.35)
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Test Macro-F1 (mean ± SD, 3 seeds)")
    axes[2].legend(loc="lower center", ncol=1, handlelength=1.4, borderaxespad=0.6)
    save(fig, out, "fig11_robustness")


# ---------------------------------------------------------------- Figure 12
def swarm(values, dot_x, dot_y):
    """Offsets along x that keep dots (dot_x wide and dot_y tall, in data units) from overlapping."""
    placed = []
    for value in values:
        for k in range(200):
            offset = (k + 1) // 2 * dot_x / 2 * (1 if k % 2 else -1)
            if all(((offset - x) / dot_x) ** 2 + ((value - y) / dot_y) ** 2 >= 1 for x, y in placed):
                break
        placed.append((offset, value))
    return [x for x, _ in placed]


def fig_subgroup(out):
    preds = [r for r in read_csv(ANALYSIS / "test_predictions.csv") if r["condition"] == "clean" and r["model"] in MODELS]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.5), gridspec_kw={"width_ratios": [1, 1.2]})
    fig.subplots_adjust(wspace=0.3)
    formats = (("MR-prefixed", lambda pid: pid.startswith("MR")), ("numeric", lambda pid: not pid.startswith("MR")))
    test_gliomas = [s for s, r in SPLIT.items() if r["split"] == "test" and r["class_name"] == "Glioma"]
    tick_labels = []
    for g, (fmt, test_fn) in enumerate(formats):
        ids = {s for s in test_gliomas if test_fn(SPLIT[s]["patient_id"])}
        groups = {SPLIT[s]["patient_group_id"] for s in ids}
        for k, model in enumerate(MODELS):
            rows = [r for r in preds if r["model"] == model and r["sample_id"] in ids]
            rate = sum(r["true_label"] != r["predicted_label"] for r in rows) / len(rows)
            x = g + (k - 1) * 0.24
            a.bar(x, rate * 100, width=0.2, color=COLOR[model], label=LABEL[model] if g == 0 else None)
            a.text(x, rate * 100 + 1, f"{rate * 100:.1f}", ha="center", va="bottom", fontsize=6.4, color=INK2)
        tick_labels.append(f"{'MR-prefixed' if g == 0 else 'Numeric'} codes\n{len(ids)} images\n"
                           f"{len(groups)} patient groups")
    a.set_xticks([0, 1], tick_labels)
    a.set_xlim(-0.5, 1.5)
    a.set_ylim(0, 50)
    a.set_ylabel("Test gliomas misclassified (%)")
    a.yaxis.grid(True, color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.legend(loc="upper left", handlelength=1, borderaxespad=0.1)
    a.set_title("Glioma errors by patient-code format", loc="left")
    # Panel B: one dot per test patient group, errors pooled over the 3 models x 3 seeds.
    per_group = defaultdict(lambda: [0, 0])
    for r in preds:
        g = per_group[SPLIT[r["sample_id"]]["patient_group_id"]]
        g[0] += r["true_label"] != r["predicted_label"]
        g[1] += 1
    columns = defaultdict(list)
    for gid, (errors, total) in per_group.items():
        kinds = {(r["class_name"], r["patient_id"].startswith("MR")) for r in SPLIT.values()
                 if r["patient_group_id"] == gid and r["split"] == "test"}
        assert len(kinds) == 1, gid
        columns[kinds.pop()].append(errors / total * 100)
    order = [(("Glioma", True), "Glioma\nMR-prefixed"), (("Glioma", False), "Glioma\nnumeric"),
             (("Meningioma", False), "Meningioma\nnumeric"), (("Pituitary tumor", False), "Pituitary\nnumeric")]
    assert {key for key, _ in order} == set(columns)
    for x, (key, _) in enumerate(order):
        values = sorted(columns[key])
        offsets = swarm(values, 0.13, 3.4)
        b.scatter([x + o for o in offsets], values, s=18, color=INK2, edgecolor="white", linewidth=0.7, zorder=3,
                  clip_on=False)
    b.set_xticks(range(len(order)), [f"{name}\n{len(columns[key])} groups" for key, name in order], fontsize=7)
    b.tick_params(axis="x", length=0)
    b.spines["bottom"].set_visible(False)
    b.set_xlim(-0.6, len(order) - 0.4)
    b.set_ylim(-3, 70)
    b.set_yticks([0, 20, 40, 60])
    b.set_ylabel("Images misclassified per group (%)")
    b.yaxis.grid(True, color=GRID, lw=0.6)
    b.set_axisbelow(True)
    errors = sorted((e for e, _ in per_group.values()), reverse=True)
    b.set_title(f"{sum(errors[:5]) / sum(errors) * 100:.0f}% of errors in 5 of {len(errors)} patient groups",
                loc="left")
    save(fig, out, "fig12_subgroup_errors")


# ---------------------------------------------------------------- Figure 13
def fig_attention_weights(out):
    record = json.loads((ATTENTION_DIR / "validation_attention.json").read_text(encoding="utf-8"))
    data = np.load(ATTENTION_DIR / "validation_attention.npz")
    se, cbam = "resnet50_se_terminal", "resnet50_cbam_terminal"
    fig = plt.figure(figsize=(WIDTH, 4.3))
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1.3], hspace=0.62, wspace=1.0, left=0.08, right=0.99,
                            top=0.93, bottom=0.09)
    # Mean weight of every channel over the validation images (seed 0), channels sorted per model.
    ax = fig.add_subplot(grid[0, 0:2])
    for model in (se, cbam):
        mean = np.sort(data[f"formal-v2-{SLUG[model]}-seed0|channel_mean"])
        ax.plot(np.arange(1, len(mean) + 1), mean, color=COLOR[model], lw=1.4, label=LABEL[model][2:])
    ax.set_xlim(1, 2048)
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 1024, 2048])
    ax.set_xlabel("Channel, sorted by mean weight")
    ax.set_ylabel("Mean weight (seed 0)")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", handlelength=1.2, borderaxespad=0.2)
    ax.set_title("Channel weights", loc="left")

    def seed_dots(ax, rows, title, xlabel):
        # One row per entry: its seed values as dots and, when there are several, their mean as a bar.
        for y, (label, values, color) in enumerate(rows):
            ax.scatter(values, [y] * len(values), s=24, color=color, edgecolor="white", linewidth=1, zorder=3)
            if len(values) > 1:
                ax.plot([np.mean(values)] * 2, [y - 0.28, y + 0.28], color=INK, lw=1.2)
        ax.set_yticks(range(len(rows)), [label for label, _, _ in rows])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel(xlabel, linespacing=1.15)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, loc="left")

    runs = {model: [record["runs"][f"formal-v2-{SLUG[model]}-seed{s}"] for s in range(3)] for model in (se, cbam)}
    seed_dots(fig.add_subplot(grid[0, 2:4]),
              [(LABEL[model][2:], [r["channel"]["variance_share_channel_identity"] * 100 for r in runs[model]],
                COLOR[model]) for model in (cbam, se)],
              "Fixed part of channel weights", "Variation in channel weights\nexplained by the channel (%)")
    # Pointing test: does the highest of the 49 spatial weights overlap the tumor mask?
    seed_dots(fig.add_subplot(grid[0, 4:6]),
              [("random\nposition", [runs[cbam][0]["spatial"]["pointing_chance_rate"] * 100], MUTED),
               ("CBAM\nspatial", [r["spatial"]["pointing_hit_rate"] * 100 for r in runs[cbam]], COLOR[cbam])],
              "Spatial pointing test", "Images where the highest weight\nfalls on the tumor (%)")
    # Spatial weights of seed 0 on the rule-selected validation examples, drawn as the 7 x 7 cells the module uses and
    # shaded from each map's lowest to highest weight; the white square is the cell used by the pointing test.
    run = f"formal-v2-{SLUG[cbam]}-seed0"
    sample_ids = list(data["sample_ids"])
    edges = np.round(np.linspace(0, 512, 8)).astype(int)
    for k, sid in enumerate(record["examples"]):
        ax = fig.add_subplot(grid[1, 2 * k:2 * k + 2])
        i = sample_ids.index(sid)
        cells = data[f"{run}|spatial"][i]
        scaled = (cells - cells.min()) / (cells.max() - cells.min())
        heat = np.asarray(Image.fromarray(scaled.astype(np.float32)).resize((512, 512), Image.Resampling.NEAREST))
        show_mri(ax, sid, heat=heat, color=ATTENTION, opacity=0.6)
        row, col = np.unravel_index(cells.argmax(), cells.shape)
        ax.add_patch(Rectangle((edges[col] - 0.5, edges[row] - 0.5), edges[col + 1] - edges[col],
                               edges[row + 1] - edges[row], fill=False, edgecolor="white", linewidth=1.1))
        ax.set_title(SHORT[int(SPLIT[sid]["class_idx"])], fontsize=8, color=INK, pad=3)
        where = "on" if data[f"{run}|pointing_hit"][i] else "off"
        ax.text(0.5, -0.03, f"highest-weight cell {where} the tumor", transform=ax.transAxes, ha="center",
                va="top", fontsize=6.6, color=INK2)
    legend = fig.add_axes([0.34, 0.012, 0.2, 0.016])
    gradient = np.ones((1, 256, 4))
    gradient[..., :3] = matplotlib.colors.to_rgb(ATTENTION)
    gradient[..., 3] = np.linspace(0, 0.6, 256)
    legend.imshow(np.ones((1, 256)), cmap="gray", vmin=0, vmax=2, aspect="auto")
    legend.imshow(gradient, aspect="auto")
    legend.set_xticks([0, 255], ["lowest", "highest"])
    legend.set_yticks([])
    legend.tick_params(length=0, labelsize=6.5)
    for spine in legend.spines.values():
        spine.set_visible(False)
    legend.text(-0.04, 0.5, "CBAM spatial weight in each image (seed 0)", transform=legend.transAxes, ha="right",
                va="center", fontsize=6.6, color=INK2)
    legend.plot([1.08, 1.2], [0.5, 0.5], transform=legend.transAxes, color=MASK_LINE, lw=1.4, clip_on=False)
    legend.text(1.23, 0.5, "tumor outline", transform=legend.transAxes, ha="left", va="center", fontsize=6.6,
                color=INK2)
    save(fig, out, "fig13_attention_weights")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/figures/report")
    parser.add_argument("--only", nargs="*", help="Figure numbers to draw, for example 2 4")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    test = json.loads((ANALYSIS / "test.json").read_text(encoding="utf-8"))
    figures = {
        "1": lambda: fig_dataset_examples(out), "2": lambda: fig_data_flowchart(out),
        "3": lambda: fig_label_errors(out), "4": lambda: fig_architecture(out),
        "5": lambda: fig_training_curves(out), "6": lambda: fig_test_results(out, test),
        "7": lambda: fig_confusion(out, test), "8": lambda: fig_gradcam_cases(out, "correct", "fig08_gradcam_correct"),
        "9": lambda: fig_gradcam_cases(out, "error", "fig09_gradcam_errors"), "10": lambda: fig_cam_overlap(out),
        "11": lambda: fig_robustness(out, test), "12": lambda: fig_subgroup(out),
        "13": lambda: fig_attention_weights(out),
    }
    for key in args.only or figures:
        figures[key]()


if __name__ == "__main__":
    main()
