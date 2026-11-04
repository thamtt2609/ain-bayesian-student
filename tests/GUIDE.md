# Sensitivity Analysis Test Guide

This guide explains how to run the CPT sensitivity analysis test and generate visualizations for Project 05.

## Overview

The `test_sensitivity_analysis.py` script conducts the mandatory **CPT sensitivity analysis** required by the midterm milestone:

> **Mandatory Controlled Experiment**: Conduct a formal CPT sensitivity analysis (`sensitivity_experiment`). Using scenario `P05-CPT-SENSITIVITY` from `data/scenarios.json`, alter one specific conditional probability row in `base_cpt.json` (e.g., adjusting the conditional impact of `StudyConsistency` on `Performance`). Compute and report the Kullback-Leibler (KL) divergence or absolute probability drift between the baseline and perturbed posterior distributions.

## Prerequisites

1. Python 3.11+ with the project dependencies installed
2. Virtual environment activated (or use the project's `.venv`)

```bash
# From project root
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Test

```bash
# From project root
python tests/test_sensitivity_analysis.py
```

Or using the virtual environment directly:

```bash
# Windows
.venv\Scripts\python.exe tests\test_sensitivity_analysis.py

# Linux/Mac
.venv/bin/python tests/test_sensitivity_analysis.py
```

## What the Test Does

1. **Loads the Bayesian Network** schema, CPTs, and scenarios from `data/`
2. **Uses scenario `P05-CPT-SENSITIVITY`** as the baseline:
   - Evidence: `AssignmentCompletion=Medium`, `ExamDifficulty=Medium`
   - Query: `Performance`
3. **Runs 5 sensitivity experiments** modifying different CPT rows:

   ### Direct Modifications (Performance CPT)
   - `Performance | Medium|Medium`: Shift 0.1 from Pass → Distinction
   - `Performance | Medium|Medium`: Shift 0.1 from Fail → Pass
   - `Performance | Medium|Medium`: Reduce Fail/Pass by 50%, boost Distinction

   ### Indirect Modifications (Upstream CPTs)
   - `StudyConsistency | High|High`: Reduce Low, increase Medium
   - `AssignmentCompletion | Medium`: Reduce Low, increase Medium

4. **Computes metrics** for each modification:
   - **KL Divergence**: D_KL(Baseline || Modified) = Σ P(x) log(P(x)/Q(x))
   - **Absolute Probability Drift**: |P(x) - Q(x)| for each state
   - **Max Absolute Drift**: Maximum drift across all states

5. **Generates 4 visualizations** in `visualization/`:
   - `sensitivity_kl_divergence.png` — Bar chart of KL divergence by modification
   - `sensitivity_absolute_drift.png` — Heatmap of absolute drift by state
   - `sensitivity_posteriors.png` — Side-by-side posterior comparison
   - `sensitivity_summary_table.png` — Formatted summary table

## Expected Output

```
============================================================
CPT SENSITIVITY ANALYSIS - VISUALIZATION
============================================================
Scenario: P05-CPT-SENSITIVITY
Evidence: {'AssignmentCompletion': 'Medium', 'ExamDifficulty': 'Medium'}
Query: Performance

Baseline Posterior:
  Fail: 0.2500
  Pass: 0.5500
  Distinction: 0.2000

--- Direct: Performance | Medium|Medium
Shift 0.1 Pass -> Distinction ---
  KL Divergence: 0.029276
  Max Abs Drift: 0.100000
  ...

[OK] Saved: visualization/sensitivity_kl_divergence.png
[OK] Saved: visualization/sensitivity_absolute_drift.png
[OK] Saved: visualization/sensitivity_posteriors.png
[OK] Saved: visualization/sensitivity_summary_table.png

============================================================
ALL VISUALIZATIONS GENERATED IN visualization/
============================================================
```

## Interpreting Results

### Key Findings

| Modification Type | KL Divergence | Max Drift | Interpretation |
|---|---|---|---|
| Direct (Performance CPT) | 0.03 - 0.33 | 0.10 - 0.40 | **High sensitivity** — Direct changes to the query node's CPT immediately affect the posterior |
| Indirect (Upstream CPTs) | ~0.00 | 0.00 | **Low/No sensitivity** — With evidence on AssignmentCompletion and ExamDifficulty, upstream changes don't propagate to Performance |

### Why Indirect Changes Show Zero Drift

The evidence `AssignmentCompletion=Medium` and `ExamDifficulty=Medium` **d-separates** Performance from upstream variables (StudyConsistency, PriorPreparation, Attendance). Once we observe the direct parents of Performance, the CPT of Performance fully determines its posterior — upstream CPT changes have no effect.

This is a key **Bayesian Network property**: observing the Markov blanket of a node makes it independent of the rest of the network.

## Troubleshooting

### "ModuleNotFoundError: No module named 'matplotlib'"
```bash
pip install matplotlib numpy
```

### "UnicodeEncodeError" on Windows
The script uses ASCII-safe characters. If you see encoding errors, ensure you're using the project's virtual environment Python.

### No visualizations generated
Check that the `visualization/` directory exists and is writable. The script creates it automatically.

## Files Generated

```
visualization/
├── sensitivity_kl_divergence.png      # KL divergence bar chart
├── sensitivity_absolute_drift.png     # Absolute drift heatmap
├── sensitivity_posteriors.png         # Side-by-side posterior bars
└── sensitivity_summary_table.png      # Formatted results table
```

## Customizing the Analysis

To test different CPT modifications, edit the `modifications` list in `test_sensitivity_analysis.py`:

```python
modifications.append({
    "name": "Your Description",
    "node": "NodeName",
    "parent_key": "ParentState1|ParentState2",  # or "" for root nodes
    "new_row": [0.1, 0.3, 0.6],  # Must sum to 1.0
    "type": "direct" or "indirect",
})
```

The `normalize_row()` function ensures probabilities are valid (non-negative, sum to 1.0).

## References

- Project specification: `INFORMATION.md` (Section 3.1 Midterm Milestone)
- Bayesian Network: `data/bn_schema.json`
- CPTs: `data/base_cpt.json`
- Scenarios: `data/scenarios.json`
- Inference implementation: `starter/student_core.py`