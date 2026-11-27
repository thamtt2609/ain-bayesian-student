#!/usr/bin/env python3
"""
Sensitivity Analysis Test with Visualization for Project 05.

This script conducts the mandatory CPT sensitivity analysis using scenario P05-CPT-SENSITIVITY
and generates visual charts showing the KL divergence and absolute probability drift.

Outputs:
- visualization/sensitivity_kl_divergence.png - KL divergence bar chart
- visualization/sensitivity_absolute_drift.png - Absolute drift heatmap
- visualization/sensitivity_posteriors.png - Side-by-side posterior comparison

Run:
    python tests/test_sensitivity_analysis.py
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starter import loader, student_core


# ─── Configuration ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIS_DIR = PROJECT_ROOT / "visualization"
VIS_DIR.mkdir(exist_ok=True)

# Color palette (colorblind-safe)
COLORS = {
    'fail': '#E74C3C',
    'pass': '#F39C12',
    'distinction': '#27AE60',
    'primary': '#2C3E50',
    'secondary': '#34495E',
    'background': '#ECF0F1',
    'grid': '#BDC3C7',
}


def compute_kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Compute KL divergence D_KL(P||Q) = sum P(x) * log(P(x)/Q(x))."""
    kl = 0.0
    for state in p:
        if p[state] > 0 and q[state] > 0:
            kl += p[state] * math.log(p[state] / q[state])
    return kl


def compute_absolute_drift(p: Dict[str, float], q: Dict[str, float]) -> Dict[str, float]:
    """Compute absolute probability drift |P(x) - Q(x)| for each state."""
    return {state: abs(p[state] - q[state]) for state in p}


def create_modified_cpt(
    base_cpt: Dict[str, dict],
    node: str,
    parent_key: str,
    new_row: List[float],
) -> Dict[str, dict]:
    """Create a deep copy of CPT with one row modified."""
    modified = copy.deepcopy(base_cpt)
    modified[node]["table"][parent_key] = new_row
    return modified


def normalize_row(row: List[float]) -> List[float]:
    """Normalize a probability row to sum to 1.0, ensuring non-negative values."""
    row = [max(0, v) for v in row]
    s = sum(row)
    if abs(s - 1.0) > 1e-10 and s > 0:
        row = [v / s for v in row]
    return row


def run_experiment(
    evidence: Dict[str, str],
    query: str,
    bn_schema: dict,
    cpt: Dict[str, dict],
    name: str,
    modified_cpt: Dict[str, dict],
) -> dict:
    """Run a single sensitivity experiment."""
    baseline = student_core.infer_posterior(evidence, query, bn_schema, cpt)
    modified = student_core.infer_posterior(evidence, query, bn_schema, modified_cpt)

    kl_div = compute_kl_divergence(baseline, modified)
    abs_drift = compute_absolute_drift(baseline, modified)

    return {
        "name": name,
        "baseline": baseline,
        "modified": modified,
        "kl_divergence": kl_div,
        "absolute_drift": abs_drift,
        "max_drift": max(abs_drift.values()),
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CPT SENSITIVITY ANALYSIS - VISUALIZATION")
    print("=" * 60)

    # Load data
    bn_schema = loader.load_bn_schema()
    cpt = loader.load_base_cpt()
    scenarios = loader.load_scenarios()

    sens_scenario = next(s for s in scenarios if s["id"] == "P05-CPT-SENSITIVITY")
    evidence = sens_scenario["evidence"]
    query = sens_scenario["query"]

    print(f"Scenario: {sens_scenario['id']}")
    print(f"Evidence: {evidence}")
    print(f"Query: {query}")

    # Baseline posterior
    baseline_post = student_core.infer_posterior(evidence, query, bn_schema, cpt)
    print(f"\nBaseline Posterior:")
    for state, prob in baseline_post.items():
        print(f"  {state}: {prob:.4f}")

    # ─── Define CPT Modifications ──────────────────────────────────────────

    modifications = []

    # The direct parent of Performance is AssignmentCompletion and ExamDifficulty
    orig_perf = cpt["Performance"]["table"]["Medium|Medium"]

    modifications.append({
        "name": "Direct: Performance | Medium|Medium\nShift 0.1 Pass -> Distinction",
        "node": "Performance",
        "parent_key": "Medium|Medium",
        "new_row": normalize_row([orig_perf[0], orig_perf[1] - 0.1, orig_perf[2] + 0.1]),
        "type": "direct",
    })

    modifications.append({
        "name": "Direct: Performance | Medium|Medium\nShift 0.1 Fail -> Pass",
        "node": "Performance",
        "parent_key": "Medium|Medium",
        "new_row": normalize_row([orig_perf[0] - 0.1, orig_perf[1] + 0.1, orig_perf[2]]),
        "type": "direct",
    })

    modifications.append({
        "name": "Direct: Performance | Medium|Medium\nReduce Fail/Pass 50%, boost Distinction",
        "node": "Performance",
        "parent_key": "Medium|Medium",
        "new_row": normalize_row([orig_perf[0] * 0.5, orig_perf[1] * 0.5, 1.0 - orig_perf[0] * 0.5 - orig_perf[1] * 0.5]),
        "type": "direct",
    })

    # 2. Indirect modification: StudyConsistency (affects AssignmentCompletion -> Performance)
    orig_sc = cpt["StudyConsistency"]["table"]["High|High"]
    modifications.append({
        "name": "Indirect: StudyConsistency | High|High\nReduce Low, increase Medium",
        "node": "StudyConsistency",
        "parent_key": "High|High",
        "new_row": normalize_row([orig_sc[0] * 0.5, orig_sc[1] * 1.2, 1.0 - orig_sc[0] * 0.5 - orig_sc[1] * 1.2]),
        "type": "indirect",
    })

    # 3. Indirect modification: AssignmentCompletion
    orig_ac = cpt["AssignmentCompletion"]["table"]["Medium"]
    modifications.append({
        "name": "Indirect: AssignmentCompletion | Medium\nReduce Low, increase Medium",
        "node": "AssignmentCompletion",
        "parent_key": "Medium",
        "new_row": normalize_row([orig_ac[0] * 0.5, orig_ac[1] * 1.2, 1.0 - orig_ac[0] * 0.5 - orig_ac[1] * 1.2]),
        "type": "indirect",
    })

    # ─── Run All Experiments ──────────────────────────────────────────────

    results = []
    for mod in modifications:
        mod_cpt = create_modified_cpt(cpt, mod["node"], mod["parent_key"], mod["new_row"])
        res = run_experiment(evidence, query, bn_schema, cpt, mod["name"], mod_cpt)
        res["type"] = mod["type"]
        results.append(res)

        print(f"\n--- {mod['name']} ---")
        print(f"  KL Divergence: {res['kl_divergence']:.6f}")
        print(f"  Max Abs Drift: {res['max_drift']:.6f}")
        for state in baseline_post:
            drift = res['absolute_drift'][state]
            print(f"  {state}: {baseline_post[state]:.4f} -> {res['modified'][state]:.4f} (d={drift:.4f})")

    # ─── Generate Visualizations ──────────────────────────────────────────

    # 1. KL Divergence Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r["name"] for r in results]
    kl_values = [r["kl_divergence"] for r in results]
    types = [r["type"] for r in results]

    colors = [COLORS['fail'] if t == "direct" else COLORS['distinction'] for t in types]
    bars = ax.barh(range(len(names)), kl_values, color=colors, edgecolor=COLORS['primary'], height=0.6)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("KL Divergence D_KL(Baseline || Modified)", fontsize=12, fontweight='bold')
    ax.set_title("CPT Sensitivity: KL Divergence by Modification", fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, color=COLORS['grid'])

    # Add value labels on bars
    for bar, val in zip(bars, kl_values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['fail'], edgecolor=COLORS['primary'], label='Direct (Performance CPT)'),
        Patch(facecolor=COLORS['distinction'], edgecolor=COLORS['primary'], label='Indirect (Upstream CPTs)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    kl_path = VIS_DIR / "sensitivity_kl_divergence.png"
    plt.savefig(kl_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n[OK] Saved: {kl_path}")

    # 2. Absolute Drift Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    states = list(baseline_post.keys())
    drift_matrix = np.array([[r["absolute_drift"][s] for s in states] for r in results])

    im = ax.imshow(drift_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max(0.5, drift_matrix.max()))

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, fontsize=11, fontweight='bold')
    ax.set_title("Absolute Probability Drift |P - Q| by State", fontsize=14, fontweight='bold', pad=15)

    # Add text annotations
    for i in range(len(results)):
        for j in range(len(states)):
            val = drift_matrix[i, j]
            color = 'white' if val > drift_matrix.max() / 2 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=10, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Absolute Drift')
    plt.tight_layout()
    drift_path = VIS_DIR / "sensitivity_absolute_drift.png"
    plt.savefig(drift_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {drift_path}")

    # 3. Side-by-side Posterior Comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (r, ax) in enumerate(zip(results, axes)):
        states_list = list(r["baseline"].keys())
        baseline_vals = [r["baseline"][s] for s in states_list]
        modified_vals = [r["modified"][s] for s in states_list]
        x = np.arange(len(states_list))
        width = 0.35

        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline',
                       color=[COLORS['fail'], COLORS['pass'], COLORS['distinction']],
                       edgecolor=COLORS['primary'], alpha=0.8)
        bars2 = ax.bar(x + width/2, modified_vals, width, label='Modified',
                       color=[COLORS['fail'], COLORS['pass'], COLORS['distinction']],
                       edgecolor=COLORS['primary'], alpha=0.4, hatch='//')

        ax.set_xticks(x)
        ax.set_xticklabels(states_list, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Probability", fontsize=10)
        ax.set_title(r["name"][:40], fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, color=COLORS['grid'])
        if idx == 0:
            ax.legend(fontsize=9)

    # Hide unused subplot
    if len(results) < 6:
        axes[-1].axis('off')

    fig.suptitle("Posterior Comparison: Baseline vs Modified CPT", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    post_path = VIS_DIR / "sensitivity_posteriors.png"
    plt.savefig(post_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {post_path}")

    # 4. Summary Table as Image
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    table_data = [
        ["Modification", "Type", "KL Divergence", "Max Abs Drift",
         "Δ Fail", "Δ Pass", "Δ Distinction"]
    ]
    for r in results:
        table_data.append([
            r["name"][:50],
            r["type"].capitalize(),
            f"{r['kl_divergence']:.6f}",
            f"{r['max_drift']:.6f}",
            f"{r['absolute_drift']['Fail']:.4f}",
            f"{r['absolute_drift']['Pass']:.4f}",
            f"{r['absolute_drift']['Distinction']:.4f}",
        ])

    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.35, 0.1, 0.12, 0.12, 0.1, 0.1, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor(COLORS['primary'])
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor(COLORS['background'])

    plt.title("Sensitivity Analysis Summary", fontsize=14, fontweight='bold', pad=20)
    table_path = VIS_DIR / "sensitivity_summary_table.png"
    plt.savefig(table_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {table_path}")

    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS GENERATED IN visualization/")
    print("=" * 60)


if __name__ == "__main__":
    main()