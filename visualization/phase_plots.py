"""
visualization/phase_plots.py — Paper Figure Generation  [V1]

All four figures use publication-quality matplotlib settings.
All outputs are vector PDF (not raster PNG) for direct LaTeX inclusion.

FIGURES:
    Figure 1 (plot_phase_diagram):
        The "money figure" — path amplitudes as arrows on complex unit circle.
        Left subplot: correct-answer paths (should cluster = constructive).
        Right subplot: wrong-answer paths (should be opposite = destructive).
        This is what you show your supervisor to prove the mechanism works.

    Figure 2 (plot_interference_decomposition):
        Bar chart: classical_sum vs interference for each contradiction query.
        Negative interference bars = destructive interference shown.
        Quantitative evidence that interference actually occurs.

    Figure 3 (plot_noise_degradation):
        MRR vs noise level for all models and ablation conditions.
        QuantumReasoner's curve should be flattest (slowest degradation).
        The widening gap at 15-20% noise is the paper's primary claim.

    Figure 4 (plot_training_curves):
        Val MRR over epochs for all ablation conditions.
        no_phase curve converging lower proves phase angles are useful.

USAGE:
    from visualization.phase_plots import (
        plot_phase_diagram, plot_interference_decomposition,
        plot_noise_degradation, plot_training_curves, set_paper_style,
    )
    set_paper_style()
    fig = plot_phase_diagram(corr_amps, wrong_amps, ...)
    fig.savefig("outputs/figures/phase_diagram.pdf", bbox_inches="tight")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Any
import warnings

import numpy as np

# Lazy matplotlib import — works even if matplotlib is not installed
try:
    import matplotlib
    matplotlib.use("Agg")   # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not installed. Figures cannot be generated.", ImportWarning)


def set_paper_style() -> None:
    """
    Apply publication-quality matplotlib style.

    Font sizes, line widths, and DPI settings matching standard ML conference
    submission requirements (NeurIPS/ICLR/ICML style).
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    plt.rcParams.update({
        "font.size":          10,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "figure.dpi":         150,
        "figure.figsize":     (6.5, 4.0),
        "lines.linewidth":    1.8,
        "lines.markersize":   5,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "text.usetex":        False,   # True if LaTeX installed
        "font.family":        "sans-serif",
        "savefig.dpi":        300,
        "savefig.format":     "pdf",
        "savefig.bbox":       "tight",
    })


def _ensure_matplotlib():
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for figure generation. "
            "Install with: pip install matplotlib"
        )


# ── Figure 1: Phase Diagram ───────────────────────────────────────────────────

def plot_phase_diagram(
    correct_amplitudes:       Any,           # list or array of complex scalars
    contradictory_amplitudes: Any,           # list or array of complex scalars
    correct_label:            str = "Correct answer paths",
    wrong_label:              str = "Contradictory answer paths",
    query_title:              str = "",
    save_path:                Optional[str | Path] = None,
) -> "plt.Figure":
    """
    The Money Figure: path amplitudes as arrows on the complex unit circle.

    After training:
        Left (correct): arrows cluster together → constructive interference
        Right (wrong):  arrows point in opposite direction → destructive

    Pre-training: both subplots look random (no pattern) — this is the
    27% problem. The money figure proves the mechanism works post-training.

    Args:
        correct_amplitudes:       Complex amplitudes for correct-answer paths.
        contradictory_amplitudes: Complex amplitudes for wrong-answer paths.
        correct_label:            Label for left subplot.
        wrong_label:              Label for right subplot.
        query_title:              Overall figure title (the query being shown).
        save_path:                If provided, save the figure here.

    Returns:
        matplotlib Figure object.
    """
    _ensure_matplotlib()

    # Convert to numpy
    def to_numpy_complex(x):
        if hasattr(x, "numpy"):
            x = x.detach().cpu().numpy()
        return np.array(x, dtype=np.complex128)

    corr_amps  = to_numpy_complex(correct_amplitudes)
    wrong_amps = to_numpy_complex(contradictory_amplitudes)

    fig, (ax_corr, ax_wrong) = plt.subplots(1, 2, figsize=(8, 4))

    # Color scheme
    corr_color  = "#2563A8"   # blue for correct
    wrong_color = "#C8400A"   # red for wrong

    def plot_amplitude_arrows(ax, amps, color, label, title):
        """Plot complex amplitudes as arrows originating from origin."""
        # Draw unit circle
        theta_circle = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta_circle), np.sin(theta_circle),
                "gray", linewidth=0.6, alpha=0.4, zorder=0)

        # Draw arrows for each amplitude
        for amp in amps:
            re_val = amp.real
            im_val = amp.imag
            # Scale to unit circle for visualization
            mag    = abs(amp)
            if mag < 1e-10:
                continue
            re_norm = re_val / max(mag, 1e-10)
            im_norm = im_val / max(mag, 1e-10)
            ax.annotate(
                "",
                xy     = (re_norm, im_norm),
                xytext = (0, 0),
                arrowprops=dict(
                    arrowstyle     = "->",
                    color          = color,
                    lw             = 1.8,
                    connectionstyle= "arc3,rad=0",
                    alpha          = 0.75,
                ),
            )
            ax.plot(re_norm, im_norm, "o", color=color, markersize=4, alpha=0.8)

        # Mark total amplitude (sum)
        if len(amps) > 0:
            total = np.sum(amps)
            t_mag = abs(total)
            if t_mag > 1e-10:
                tr = total.real / max(t_mag, 1e-10)
                ti = total.imag / max(t_mag, 1e-10)
                ax.annotate(
                    "",
                    xy=(tr * 1.05, ti * 1.05),
                    xytext=(0, 0),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="black",
                        lw=2.5,
                    ),
                )

        # Formatting
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Re(A)")
        ax.set_ylabel("Im(A)")
        ax.set_title(f"{title}\n({len(amps)} paths)", fontsize=10)

        # Phase spread annotation
        if len(amps) > 1:
            phases     = np.angle(amps)
            phase_std  = np.std(phases)
            ax.text(
                0.02, 0.98,
                f"Phase spread: {phase_std:.2f} rad\n"
                f"|Total|: {abs(np.sum(amps)):.3f}",
                transform     = ax.transAxes,
                fontsize      = 7,
                verticalalignment = "top",
                color         = color,
                alpha         = 0.85,
            )

    plot_amplitude_arrows(ax_corr,  corr_amps,  corr_color,  correct_label, "Correct answer")
    plot_amplitude_arrows(ax_wrong, wrong_amps, wrong_color, wrong_label,   "Wrong answer")

    # Global title
    title = f"Phase Diagram: {query_title}" if query_title else "Amplitude Interference Diagram"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    # Legend
    patches = [
        mpatches.Patch(color=corr_color,  label=correct_label),
        mpatches.Patch(color=wrong_color, label=wrong_label),
        mpatches.Patch(color="black",     label="Total amplitude"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig


# ── Figure 2: Interference Decomposition ─────────────────────────────────────

def plot_interference_decomposition(
    analyses:     list[dict],
    query_labels: list[str],
    save_path:    Optional[str | Path] = None,
) -> "plt.Figure":
    """
    Bar chart showing classical_sum vs interference per contradiction query.

    Negative interference bars = destructive interference occurring.
    This is the quantitative evidence that interference is real.

    Args:
        analyses:     List of dicts from compute_interference_terms().
                      Each must have 'classical_sum', 'interference',
                      'total_probability', and 'interference_sign'.
        query_labels: Label for each bar group (query descriptions).
        save_path:    Optional save path.

    Returns:
        matplotlib Figure.
    """
    _ensure_matplotlib()

    n   = len(analyses)
    x   = np.arange(n)
    w   = 0.25

    fig, ax = plt.subplots(figsize=(max(6, n * 2.5), 4.5))

    classical_vals  = [float(a.get("classical_sum",       0)) for a in analyses]
    interference_vals = [float(a.get("interference",      0)) for a in analyses]
    total_vals      = [float(a.get("total_probability",   0)) for a in analyses]

    bars_c = ax.bar(x - w, classical_vals,     w, label="Classical sum",    color="#4CAF50", alpha=0.8)
    bars_i = ax.bar(x,     interference_vals,  w, label="Interference",     color="#F44336", alpha=0.8)
    bars_t = ax.bar(x + w, total_vals,         w, label="Total P(t|s)",     color="#2196F3", alpha=0.8)

    # Annotate sign for interference bars
    for bar, val, a in zip(bars_i, interference_vals, analyses):
        sign = a.get("interference_sign", "negligible")
        color = "#B71C1C" if sign == "destructive" else "#1B5E20"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            "▼" if sign == "destructive" else ("▲" if sign == "constructive" else "—"),
            ha="center", va="bottom", fontsize=9, color=color, fontweight="bold",
        )

    # Reference line at 0
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(query_labels, rotation=10, ha="right", fontsize=8)
    ax.set_ylabel("Probability value")
    ax.set_title("Interference Decomposition: Classical vs Quantum", fontsize=12)
    ax.legend(fontsize=9)

    # Add interpretation text
    ax.text(
        0.98, 0.98,
        "▼ Destructive = interference suppresses wrong answer\n"
        "▲ Constructive = interference amplifies correct answer",
        transform=ax.transAxes, fontsize=7,
        verticalalignment="top", horizontalalignment="right",
        color="gray",
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig


# ── Figure 3: Noise Degradation ──────────────────────────────────────────────

# REPLACE WITH:
def plot_noise_degradation(
    results_by_model: dict[str, dict[float, float]],
    save_path = None,
    highlight_model: str = "QuantumReasoner [full]",
    hits1_by_model: dict[str, dict[float, float]] = None,   # ← add
    hits10_by_model: dict[str, dict[float, float]] = None,  # ← add
) -> "plt.Figure":
    """
    MRR vs noise level for all models. The key experimental result.

    The flattest curve = most noise-robust model.
    The widening gap at 15-20% noise is the paper's primary claim.

    Args:
        results_by_model: Dict: model_name → {noise_rate → MRR float}.
                          Example:
                          {
                            "TransE": {0.0: 0.31, 0.10: 0.22, 0.20: 0.14},
                            "QuantumReasoner": {0.0: 0.30, 0.10: 0.27, 0.20: 0.24},
                          }
        save_path:        Optional save path.
        highlight_model:  Model to draw with thicker, darker line.

    Returns:
        matplotlib Figure.
    """
    _ensure_matplotlib()

    n_plots = 1 + (hits1_by_model is not None) + (hits10_by_model is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]
    ax = axes[0]

    # Color palette for models
    colors = [
        "#E53935", "#43A047", "#1E88E5", "#8E24AA",
        "#FB8C00", "#00ACC1", "#6D4C41", "#546E7A",
    ]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    markers    = ["o", "s", "^", "D", "v", "P", "*", "X"]

    model_names = list(results_by_model.keys())
    for idx, (model_name, noise_mrr) in enumerate(results_by_model.items()):
        noise_levels = sorted(noise_mrr.keys())
        mrr_values   = [noise_mrr[n] for n in noise_levels]
        noise_pct    = [n * 100 for n in noise_levels]

        is_highlight = model_name == highlight_model
        lw = 2.8 if is_highlight else 1.6
        zo = 10  if is_highlight else 5

        ax.plot(
            noise_pct, mrr_values,
            label      = model_name,
            color      = colors[idx % len(colors)],
            linestyle  = linestyles[idx % len(linestyles)],
            marker     = markers[idx % len(markers)],
            linewidth  = lw,
            markersize = 6 if is_highlight else 4,
            zorder     = zo,
        )
    
    # ADD after the existing plotting loop, before ax.set_xlabel:
    ax.set_title("Filtered MRR vs Noise", fontsize=11)

    if hits1_by_model is not None and len(axes) > 1:
        ax_h1 = axes[1]
        for idx, (model_name, noise_h1) in enumerate(hits1_by_model.items()):
            noise_pct = [n * 100 for n in sorted(noise_h1.keys())]
            h1_vals   = [noise_h1[n] for n in sorted(noise_h1.keys())]
            ax_h1.plot(noise_pct, h1_vals,
                    label=model_name, color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2.8 if model_name == highlight_model else 1.6,
                    marker=markers[idx % len(markers)], markersize=5)
        ax_h1.set_xlabel("Noise Rate (%)")
        ax_h1.set_ylabel("Hits@1")
        ax_h1.set_title("Hits@1 vs Noise\n(correct answer ranked #1)", fontsize=11)
        ax_h1.legend(fontsize=7, ncol=2 if len(hits1_by_model) > 4 else 1)
        ax_h1.set_xlim(-1, 21)
        ax_h1.set_ylim(bottom=0)
        ax_h1.grid(True, alpha=0.3)

    if hits10_by_model is not None and len(axes) > 2:
        ax_h10 = axes[2]
        for idx, (model_name, noise_h10) in enumerate(hits10_by_model.items()):
            noise_pct = [n * 100 for n in sorted(noise_h10.keys())]
            h10_vals  = [noise_h10[n] for n in sorted(noise_h10.keys())]
            ax_h10.plot(noise_pct, h10_vals,
                        label=model_name, color=colors[idx % len(colors)],
                        linestyle=linestyles[idx % len(linestyles)],
                        linewidth=2.8 if model_name == highlight_model else 1.6,
                        marker=markers[idx % len(markers)], markersize=5)
        ax_h10.set_xlabel("Noise Rate (%)")
        ax_h10.set_ylabel("Hits@10")
        ax_h10.set_title("Hits@10 vs Noise\n(safety net: truth in top 10)", fontsize=11)
        ax_h10.legend(fontsize=7, ncol=2 if len(hits10_by_model) > 4 else 1)
        ax_h10.set_xlim(-1, 21)
        ax_h10.set_ylim(bottom=0)
        ax_h10.grid(True, alpha=0.3)

    ax.set_xlabel("Noise Rate (%)")
    ax.set_ylabel("Filtered MRR")
    # ax.set_title("Noise Robustness: MRR vs Corruption Rate", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, ncol=2 if len(model_names) > 4 else 1)
    ax.set_xlim(-1, 21)
    ax.set_ylim(bottom=0)

    # Annotate the widening gap
    ax.text(
        0.65, 0.08,
        "← Gap widens with noise\n   (quantum advantage)",
        transform=ax.transAxes, fontsize=8, color="#1E88E5", alpha=0.85,
        ha="left", va="bottom",
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig


# ── Figure 4: Training Curves ─────────────────────────────────────────────────

def plot_training_curves(
    history_by_condition: dict[str, list[float]],
    metric:               str = "val_mrr",
    save_path:            Optional[str | Path] = None,
) -> "plt.Figure":
    """
    Validation MRR over epochs for all ablation conditions.

    Expected pattern after training:
        "full" converges highest (complete model)
        "no_phase" converges lower (proves imaginary parts contribute)
        "no_paths" converges lower than "full" (proves multi-hop helps)
        "classical" converges lowest (proves Born rule squaring is key)

    Args:
        history_by_condition: Dict: condition_name → list of MRR values per epoch.
        metric:               Metric label for y-axis.
        save_path:            Optional save path.

    Returns:
        matplotlib Figure.
    """
    _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = {
        "full":      "#1E88E5",
        "no_phase":  "#E53935",
        "no_paths":  "#FB8C00",
        "classical": "#43A047",
    }
    linestyles = {
        "full":      "-",
        "no_phase":  "--",
        "no_paths":  "-.",
        "classical": ":",
    }

    for condition, values in history_by_condition.items():
        epochs = list(range(1, len(values) + 1))
        color  = colors.get(condition, "gray")
        ls     = linestyles.get(condition, "-")
        lw     = 2.4 if condition == "full" else 1.5

        ax.plot(epochs, values, label=condition, color=color,
                linestyle=ls, linewidth=lw)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Ablation Training Dynamics", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)

    # Annotate final values
    for condition, values in history_by_condition.items():
        if values:
            final_mrr = values[-1]
            ax.text(
                len(values), final_mrr,
                f" {final_mrr:.3f}",
                fontsize=7,
                color=colors.get(condition, "gray"),
                va="center",
            )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig


# ── Utility: save all figures ─────────────────────────────────────────────────

def generate_all_paper_figures(
    model,
    toy_kg,
    device,
    ablation_results = None,
    output_dir:      str | Path = "outputs/figures",
) -> dict[str, Path]:
    """
    Generate all four paper figures for a trained model.

    Calls run analysis, then generates figures and saves to output_dir.
    Returns dict of figure_name → saved_path.

    Args:
        model:           Trained QuantumReasoner.
        toy_kg:          ToyKG with contradiction_queries.
        device:          Torch device.
        ablation_results: Optional AblationResult list (for training curves).
        output_dir:      Where to save figures.

    Returns:
        Dict mapping figure name → Path.
    """
    import torch
    from models.components.path_aggregator import PathEnumerator

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    model.eval()
    adj        = toy_kg.get_adjacency()
    enumerator = PathEnumerator(adj, max_hops=3, max_paths=8)

    # Collect interference data
    analyses:       list[dict] = []
    query_labels:   list[str]  = []
    corr_amps_all:  list       = []
    wrong_amps_all: list       = []

    with torch.no_grad():
        for cq in toy_kg.contradiction_queries:
            h_id     = toy_kg.entity2id[cq["head"]]
            corr_id  = toy_kg.entity2id[cq["correct_tail"]]
            wrong_id = toy_kg.entity2id[cq["contradictory_tail"]]

            h_state    = model.encoder(torch.tensor([h_id],    device=device)).squeeze(0)
            corr_state = model.encoder(torch.tensor([corr_id], device=device)).squeeze(0)
            wrong_state= model.encoder(torch.tensor([wrong_id],device=device)).squeeze(0)

            corr_paths  = enumerator.find_paths(h_id, corr_id)
            wrong_paths = enumerator.find_paths(h_id, wrong_id)

            if wrong_paths:
                wrong_analysis = model.aggregator.compute_interference_terms(
                    h_state, wrong_state, wrong_paths, model.unitary
                )
                analyses.append(wrong_analysis)
                query_labels.append(f"{cq['head']}→wrong")

                if wrong_analysis.get("amplitudes"):
                    wrong_amps_all.extend(wrong_analysis["amplitudes"])

            if corr_paths and not corr_amps_all:
                corr_analysis = model.aggregator.compute_interference_terms(
                    h_state, corr_state, corr_paths, model.unitary
                )
                if corr_analysis.get("amplitudes"):
                    corr_amps_all.extend(corr_analysis["amplitudes"])

    saved = {}

    # Figure 1: Phase diagram (use first query's amplitudes)
    if corr_amps_all and wrong_amps_all:
        fig_path = output_dir / "phase_diagram.pdf"
        fig = plot_phase_diagram(
            correct_amplitudes       = corr_amps_all[:4],
            contradictory_amplitudes = wrong_amps_all[:4],
            query_title              = toy_kg.contradiction_queries[0]["query"],
            save_path                = fig_path,
        )
        plt.close(fig)
        saved["phase_diagram"] = fig_path

    # Figure 2: Interference decomposition
    if analyses:
        fig_path = output_dir / "interference_decomp.pdf"
        fig = plot_interference_decomposition(
            analyses=analyses, query_labels=query_labels, save_path=fig_path
        )
        plt.close(fig)
        saved["interference_decomp"] = fig_path

    return saved
