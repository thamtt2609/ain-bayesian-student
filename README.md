# Bayesian Student Scenario Simulator

An interactive probabilistic reasoning system for student academic trajectories using Bayesian Networks. Features exact inference by enumeration, CPT sensitivity analysis, and a Streamlit web dashboard for scenario exploration.

## Python Version

**Python 3.11+** (tested on 3.11, 3.12)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run Application (Streamlit Dashboard)

```bash
streamlit run starter/app.py
```

Then open http://localhost:8501 in your browser.

## Run Sensitivity Analysis

```bash
# Run CPT sensitivity analysis with visualizations
python tests/test_sensitivity_analysis.py
```

Results saved to `visualization/` as PNG plots.

## Run Sensitivity Experiment

```bash
# Run the required CPT sensitivity experiment
python scripts/run_sensitivity_experiment.py
```

## Regenerate Synthetic Student Data

```bash
# Regenerate the committed 1,000-row sample (seed 42)
python scripts/generate_students.py --n 1000 --seed 42

# Generate larger sample for exploration
python scripts/generate_students.py --n 5000 --seed 42
```

Outputs `data/students_seed42.csv` (committed) or `data/students_n5000_seed42.csv` (non-committed).

## Run Tests

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_project_05_sanity.py -v
```

## Project Structure (for Submission)

```
.
├── README.md                 # This file
├── requirements.txt          # Pinned dependencies
├── starter/
│   ├── __init__.py
│   ├── loader.py             # Data loading & validation
│   ├── student_core.py       # Inference: infer_posterior, sensitivity_experiment
│   └── app.py                # Streamlit dashboard
├── scripts/
│   ├── generate_students.py  # Synthetic cohort generator
│   └── run_sensitivity_experiment.py
├── tests/
│   ├── test_inference.py
│   ├── test_project_05_sanity.py
│   ├── test_sensitivity_analysis.py
│   └── GUIDE.md
└── data/
    ├── README.md
    ├── bn_schema.json        # BN structure (nodes, edges, states)
    ├── base_cpt.json         # Conditional probability tables
    ├── scenarios.json        # Benchmark evidence scenarios
    ├── students_seed42.csv   # 1,000 synthetic student records
    └── SHA256SUMS            # Checksums for reproducibility
```

## License

All data is synthetic educational simulation. Not for real student assessment.
Code: MIT License

## Links

- **GitHub Repository**: https://github.com/thamtt2609/ain-bayesian-student
- **References**: Pearl (1988), Koller & Friedman (2009), Russell & Norvig (2020)
