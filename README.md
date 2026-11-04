# Project 05 — Bayesian Student Scenario Simulator

## 1. Problem Description

Reasoning under uncertainty is a core capability in Artificial Intelligence (AI), enabling agents to make coherent inferences from incomplete, noisy, or indirect observations. A Bayesian Network (BN) is a Directed Acyclic Graph (DAG) representing a factored Joint Probability Distribution (JPD) over a set of random variables, where directed edges capture conditional dependencies and missing edges encode explicit conditional independence assertions.

<<<<<<< HEAD
This project models student academic trajectories using a Bayesian Network parameterized with discrete Conditional Probability Tables (CPTs). The network models interactions among six discrete educational random variables:
=======
**Current Branch**: `midterm` — Midterm milestone (AI Core) completed ✅

## Features
>>>>>>> 4c2fc54 (docs: update README.md for midterm branch)

1. `PriorPreparation` ($\{\text{Low}, \text{Medium}, \text{High}\}$): Foundational programming competency prior to enrollment.
2. `Attendance` ($\{\text{Low}, \text{Medium}, \text{High}\}$): Lecture and laboratory attendance rate.
3. `StudyConsistency` ($\{\text{Low}, \text{Medium}, \text{High}\}$): Regularity of weekly self-study habits.
4. `AssignmentCompletion` ($\{\text{Incomplete}, \text{Complete}\}$): Timely submission of assignments and problem sets.
5. `ExamDifficulty` ($\{\text{Standard}, \text{Challenging}\}$): Perceived complexity of semester examination papers.
6. `Performance` ($\{\text{Fail}, \text{Pass}, \text{Good}, \text{Excellent}\}$): Final academic course grade.

Given observed evidence, the system performs probabilistic inference to calculate posterior marginal distributions, assess sensitivity to prior assumptions, and explain belief updates. All student records are strictly synthetic educational datasets generated for academic exploration.

---

## 2. Provided Materials & Starter Resources

The project package provides the network structural topology, baseline conditional probability distributions, synthetic student cohorts, and an interactive Streamlit dashboard.

<<<<<<< HEAD
### File Structure
```text
project-05-bayesian-student/
├── data/
│   ├── README.md                 # Data dictionary and schema specifications
│   ├── bn_schema.json            # Directed acyclic graph topology, nodes, and parent mappings
│   ├── base_cpt.json             # Canonical conditional probability tables for all nodes
│   ├── scenarios.json            # Benchmark evidence fixtures (P05-LITTLE-EVIDENCE to P05-CPT-SENSITIVITY)
│   ├── students_seed42.csv       # 1,000 synthetic student profile records (seed 42)
│   └── SHA256SUMS                # Cryptographic checksums ensuring data reproducibility
├── scripts/
│   └── generate_students.py      # Synthetic cohort generator using forward ancestral sampling
├── starter/
│   ├── loader.py                 # JSON parser and CPT stochastic validation utilities
│   ├── student_core.py           # Algorithmic stubs for exact/approximate Bayesian inference
│   └── app.py                   # Streamlit interactive dashboard skeleton
├── tests/
│   └── test_project_05_sanity.py # Unit tests verifying schema validity, sums, and interfaces
└── requirements.txt              # Pinned Python package dependencies (including streamlit)
```

### Starter Infrastructure vs. Student Implementation
The module `starter/loader.py` validates that all rows in `data/base_cpt.json` sum strictly to $1.0 \pm 10^{-6}$. The script `starter/app.py` renders the interactive visual interface. Students implement the inference core from first principles in `starter/student_core.py`, specifically: exact inference by enumeration or likelihood weighting (`infer_posterior`) and the comparative evaluation harness (`sensitivity_experiment`). Third-party Bayesian inference libraries (e.g., `pgmpy`) and Large Language Model (LLM) APIs are strictly prohibited.
=======
5. **Interactive Streamlit Dashboard** - Located in `starter/app.py` featuring:
   - Network structure visualization
   - Interactive evidence toggles for all variables
   - Prior vs. posterior distribution comparison
   - Scenario loading capabilities
   - Synthetic data notice banner

## Midterm Requirements (AI Core) — **COMPLETED**

Students implemented the following in `starter/student_core.py`:

### 1. Probabilistic Inference (`infer_posterior`) ✅
Implemented **exact inference by enumeration** to compute:
```
P(query | evidence) = α ∑_y P(query, evidence, y)
```
Output: normalized probability distribution summing to 1.0.

### 2. Controlled Experiment (`sensitivity_experiment`) ✅
Using scenario P05-CPT-SENSITIVITY:
- Alter specific conditional probability rows in `base_cpt.json`
- Compute **KL divergence** and **absolute probability drift** between baseline and perturbed posteriors
- Comprehensive analysis with 5 experiments (3 direct, 2 indirect modifications)

### 3. Sensitivity Analysis Test Suite & Visualizations ✅
- `tests/test_sensitivity_analysis.py` — Automated test generating 4 visualization charts
- `tests/GUIDE.md` — Documentation for running the analysis
- `visualization/` — Generated charts:
  - `sensitivity_kl_divergence.png` — KL divergence bar chart
  - `sensitivity_absolute_drift.png` — Absolute drift heatmap
  - `sensitivity_posteriors.png` — Side-by-side posterior comparison
  - `sensitivity_summary_table.png` — Formatted results table
  - `EXPLAIN.md` — Detailed results explanation

### Key Finding
Direct modifications to Performance CPT show **high sensitivity** (KL: 0.03–0.33, Max Drift: up to 0.40).  
Indirect/upstream modifications show **zero drift** due to **d-separation** — observing AssignmentCompletion and ExamDifficulty makes Performance independent of upstream variables.

## Technical Implementation

### Dependencies
See `requirements.txt` for pinned package versions including Streamlit, matplotlib, numpy.

### Data Flow
1. `loader.py`: Infrastructure layer for loading JSON/CSV data
2. `student_core.py`: **Student implementation** - Bayesian inference algorithms
3. `app.py`: Presentation layer - Streamlit dashboard calling student_core

### Validation
- Sanity tests in `tests/test_project_05_sanity.py` verify data integrity
- Sensitivity test in `tests/test_sensitivity_analysis.py` validates inference and generates visualizations
- Grading focuses on:
  - Mathematical correctness of inference (15 pts)
  - Rigor of sensitivity analysis (10 pts) 
  - Oral defense and live demonstration (15 pts)

## Final Project Enhancements (AI Product) — *Planned for next milestone*

### Required Features
- Interactive network graph visualizer (beyond current JSON display)
- Multi-variable evidence toggles with instant updates
- Side-by-side prior vs. posterior distribution charts
- Scenario comparison mode for evaluating two student profiles simultaneously
- Explicit, persistent synthetic data notice banner

### Validation Scenarios
Demonstrate correct reasoning across all four benchmark fixtures:
1. P05-LITTLE-EVIDENCE - Minimal uninformative evidence
2. P05-STRONG-POSITIVE - Strong converging evidence  
3. P05-CONFLICTING - Conflicting evidence streams
4. P05-CPT-SENSITIVITY - Parameter perturbation analysis

### Deliverables
- Functional Streamlit web application
- Clean GitHub repository with proper documentation
- IEEE-formatted engineering technical report (`report.pdf`)

## Getting Started
>>>>>>> 4c2fc54 (docs: update README.md for midterm branch)

### Installation and Execution
```bash
<<<<<<< HEAD
=======
# Clone repository
git clone https://github.com/manhanbanhcanh/ain-bayesian-student.git
cd ain-bayesian-student

# Switch to midterm branch
git checkout midterm

# Create virtual environment
>>>>>>> 4c2fc54 (docs: update README.md for midterm branch)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run starter/app.py

# Run sensitivity analysis (generates visualizations in visualization/)
python tests/test_sensitivity_analysis.py
```

---

## 3. Requirements & Deliverables

### 3.1 Midterm Milestone — AI Core
- **Algorithmic Implementation**: Implement probabilistic inference over discrete Bayesian Networks from first principles in `starter/student_core.py`. Implement `infer_posterior(evidence, query, bn_schema, cpt)` using either Exact Inference by Enumeration (evaluating the joint distribution sum $\alpha \sum_{\mathbf{y}} P(X, \mathbf{e}, \mathbf{y})$) or Likelihood Weighting sampling. The output must be a normalized probability distribution summing to 1.0.
- **Mandatory Controlled Experiment**: Conduct a formal CPT sensitivity analysis (`sensitivity_experiment`). Using scenario `P05-CPT-SENSITIVITY` from `data/scenarios.json`, alter one specific conditional probability row in `base_cpt.json` (e.g., adjusting the conditional impact of `StudyConsistency` on `Performance`). Compute and report the Kullback-Leibler (KL) divergence or absolute probability drift between the baseline and perturbed posterior distributions.
- **Deliverables**: Fully implemented `student_core.py`, empirical sensitivity benchmark scripts, and an oral presentation slide deck (`presentation.pdf`).
- **Grading Criteria**: Mathematical formalization and inference correctness (15 pts); sensitivity analysis rigor and experimental depth (10 pts); oral defense and live query demonstration (15 pts). Total: 40 points (40% weight).

### 3.2 Final Milestone — AI Product
- **Interactive Application**: Package the Bayesian reasoning engine into an interactive local web dashboard using Streamlit. Features must include: an interactive network graph visualizer; multi-variable evidence toggles; side-by-side prior vs. posterior distribution charts; a scenario comparison mode evaluating two hypothetical student profiles simultaneously; and an explicit, persistent banner indicating that all data is synthetic.
- **Stress & Edge Scenarios**: Demonstrate correct probabilistic reasoning across all four benchmark fixtures in `data/scenarios.json`: minimal uninformative evidence (`P05-LITTLE-EVIDENCE`), strongly positive converging evidence (`P05-STRONG-POSITIVE`), conflicting evidence (`P05-CONFLICTING`), and parameter perturbation (`P05-CPT-SENSITIVITY`).
- **Deliverables**: Functional Streamlit web application, clean GitHub repository, and an IEEE-formatted engineering technical report (`report.pdf`).
- **Grading Criteria**: Dashboard visual clarity, graph interactivity, and usability (25 pts); technical report rigor, probability derivations, and sensitivity analysis (15 pts); oral defense against complex multi-evidence queries (10 pts). Total: 50 points (50% weight).

---

## 4. References

- [1] J. Pearl, *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. San Mateo, CA, USA: Morgan Kaufmann, 1988.
- [2] D. Koller and N. Friedman, *Probabilistic Graphical Models: Principles and Techniques*. Cambridge, MA, USA: MIT Press, 2009.
- [3] S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. Hoboken, NJ, USA: Pearson, 2020, pp. 385–427.
