"""Streamlit starter for Project 05. Loads and displays the network; does NOT infer a posterior.

The graded AI core (`infer_posterior`) lives in `student_core.py` and is called from here, but
this file must not implement inference itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starter import loader, student_core  # noqa: E402

st.set_page_config(page_title="Bayesian Student Scenario Simulator — starter", layout="wide")

st.title("Bayesian Student Scenario Simulator — starter")
st.warning(
    "⚠️ " + loader.SYNTHETIC_DATA_NOTICE,
    icon="⚠️",
)

bn_schema = loader.load_bn_schema()
cpt = loader.load_base_cpt()
scenarios = loader.load_scenarios()

node_names = [node["name"] for node in bn_schema["nodes"]]
node_by_name = {node["name"]: node for node in bn_schema["nodes"]}

with st.sidebar:
    st.header("Evidence")
    st.caption("Check a variable to observe it, then pick its value.")

    evidence: dict[str, str] = {}
    for name in node_names:
        include = st.checkbox(f"Observe {name}", key=f"include_{name}")
        if include:
            states = node_by_name[name]["states"]
            value = st.selectbox(name, states, key=f"value_{name}")
            evidence[name] = value

    st.divider()
    query = st.selectbox("Query variable", node_names, index=node_names.index("Performance"))

    st.divider()
    st.header("Load a scenario")
    scenario_id = st.selectbox(
        "Scenario",
        ["(none)"] + [scenario["id"] for scenario in scenarios],
    )
    if scenario_id != "(none)":
        chosen = next(s for s in scenarios if s["id"] == scenario_id)
        st.caption(chosen["description"])
        st.json(chosen.get("evidence", {}))

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Network structure")
    st.json({"nodes": node_names, "edges": bn_schema["edges"]})
    st.caption("Independence notes:")
    for note in bn_schema.get("independence_notes", []):
        st.markdown(f"- {note}")

with col_right:
    st.subheader("Prior over query variable")
    prior_spec = cpt[query]
    if not prior_spec["parents"]:
        prior = dict(zip(prior_spec["states"], prior_spec["table"][""]))
        st.bar_chart(prior)
    else:
        st.caption(f"{query} has parents {prior_spec['parents']}; its marginal prior is not a single CPT row.")

st.subheader("Posterior given evidence")
st.write("Evidence:", evidence or "(none)")
st.write("Query:", query)

if st.button("Compute posterior"):
    try:
        posterior = student_core.infer_posterior(evidence, query, bn_schema, cpt)
        st.bar_chart(posterior)
    except NotImplementedError as exc:
        st.error(f"Not implemented yet: {exc}\nImplement `infer_posterior` in `starter/student_core.py`.")

st.divider()
st.subheader("Committed synthetic sample (first 20 rows)")
students = loader.load_students()
st.dataframe(students[:20])

# ==========================================
# SENSITIVITY EXPERIMENTS
# ==========================================

st.divider()
st.header("Sensitivity Experiments")

# Load the CPT sensitivity scenario
sensitivity_scenario = next(
    s for s in scenarios
    if s["id"] == "P05-CPT-SENSITIVITY"
)

baseline_evidence = sensitivity_scenario["evidence"]
sensitivity_query = sensitivity_scenario["query"]

# Baseline posterior
baseline_posterior = student_core.infer_posterior(
    baseline_evidence,
    sensitivity_query,
    bn_schema,
    cpt
)

# Experiment 1: Modify one CPT row
import copy

modified_cpt = copy.deepcopy(cpt)
modified_cpt["Performance"]["table"]["Medium|Medium"] = [
    0.15, 0.55, 0.30
]

experiment_1 = student_core.sensitivity_experiment(
    baseline_evidence,
    sensitivity_query,
    bn_schema,
    cpt,
    modified_cpt
)

# Experiment 2: Change one evidence value
changed_evidence = dict(baseline_evidence)
changed_evidence["AssignmentCompletion"] = "High"

experiment_2_posterior = student_core.infer_posterior(
    changed_evidence,
    sensitivity_query,
    bn_schema,
    cpt
)

# Display evidence
st.subheader("Baseline Evidence")
st.json(baseline_evidence)

# Display three results
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Baseline")
    st.bar_chart(baseline_posterior)

with col2:
    st.subheader("Experiment 1")
    st.caption("Modified CPT")
    st.bar_chart({
        state: experiment_1["modified_posterior"][i]
        for i, state in enumerate(
            ["Fail", "Pass", "Distinction"]
        )
    })
    st.metric(
        "KL Divergence",
        f'{experiment_1["kl_divergence"][0]:.6f}'
    )

with col3:
    st.subheader("Experiment 2")
    st.caption("Changed AssignmentCompletion to High")
    st.bar_chart(experiment_2_posterior)

# Comparison table
st.subheader("Posterior Comparison")

comparison = {
    "Baseline": baseline_posterior,
    "Experiment 1": {
        state: experiment_1["modified_posterior"][i]
        for i, state in enumerate(
            ["Fail", "Pass", "Distinction"]
        )
    },
    "Experiment 2": experiment_2_posterior,
}

st.dataframe(comparison)
