"""Streamlit application for Project 05 — Bayesian Student Scenario Simulator.

Final Milestone: Interactive web dashboard with:
- Interactive network graph visualizer
- Multi-variable evidence toggles
- Side-by-side prior vs. posterior distribution charts
- Scenario comparison mode (two student profiles)
- Persistent synthetic data banner
- All four benchmark scenarios demonstrated
"""

from __future__ import annotations

import sys
import math
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starter import loader, student_core  # noqa: E402

# Page configuration
st.set_page_config(
    page_title="Bayesian Student Scenario Simulator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for persistent banner and styling
st.markdown("""
<style>
    .synthetic-banner {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.85; }
        100% { opacity: 1; }
    }
    .scenario-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .metric-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 6px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data
bn_schema = loader.load_bn_schema()
cpt = loader.load_base_cpt()
scenarios = loader.load_scenarios()
students = loader.load_students()

node_names = [node["name"] for node in bn_schema["nodes"]]
node_by_name = {node["name"]: node for node in bn_schema["nodes"]}

# Query variable initialized from sidebar (will be set via selectbox)
query: str = "Performance"

def get_max_state(prob_dict: dict[str, float]) -> str:
    """Get the state with maximum probability from a distribution dict."""
    return max(prob_dict.keys(), key=lambda k: prob_dict[k])

# ==========================================
# PERSISTENT SYNTHETIC DATA BANNER
# ==========================================
st.markdown(f"""
<div class="synthetic-banner">
    ⚠️ {loader.SYNTHETIC_DATA_NOTICE}
</div>
""", unsafe_allow_html=True)

# Title
st.title("🎓 Bayesian Student Scenario Simulator")
st.caption("Interactive probabilistic reasoning over student academic trajectories")

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def create_network_graph():
    """Create an interactive network graph using pyvis."""
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="#333333")
    net.set_options("""
    {
        "nodes": {
            "shape": "box",
            "font": {"size": 14, "face": "Arial"},
            "margin": 10,
            "borderWidth": 2,
            "shadow": true
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
            "smooth": {"type": "curvedCW", "roundness": 0.2},
            "color": {"color": "#666666", "highlight": "#333333"},
            "width": 2
        },
        "physics": {
            "enabled": true,
            "barnesHut": {"gravitationalConstant": -8000, "springLength": 200},
            "stabilization": {"iterations": 100}
        },
        "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)

    # Color scheme for nodes
    node_colors = {
        "PriorPreparation": "#4CAF50",
        "Attendance": "#2196F3",
        "ExamDifficulty": "#FF9800",
        "StudyConsistency": "#9C27B0",
        "AssignmentCompletion": "#F44336",
        "Performance": "#607D8B"
    }

    for node in bn_schema["nodes"]:
        name = node["name"]
        states = ", ".join(node["states"])
        desc = node.get("description", "")
        color = node_colors.get(name, "#757575")

        net.add_node(
            name,
            label=name,
            title=f"{desc}\n\nStates: {states}",
            color=color,
            borderWidth=2,
            size=25
        )

    for edge in bn_schema["edges"]:
        net.add_edge(edge[0], edge[1])

    # Save to temporary file and return HTML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        net.save_graph(f.name)
        with open(f.name, 'r') as html_file:
            return html_file.read()


def compute_prior(query: str, bn_schema: dict, cpt: dict) -> dict:
    """Compute marginal prior distribution for a query variable."""
    return student_core.infer_posterior({}, query, bn_schema, cpt)


def create_distribution_chart(prior: dict, posterior: dict, _query: str, title: str = "") -> go.Figure:  # noqa: ARG001
    """Create a side-by-side prior vs posterior bar chart using plotly."""
    states = list(prior.keys())
    prior_values = [prior[s] for s in states]
    posterior_values = [posterior[s] for s in states]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Prior',
        x=states,
        y=prior_values,
        marker_color='#2196F3',
        opacity=0.7,
        text=[f'{v:.3f}' for v in prior_values],
        textposition='auto',
    ))

    fig.add_trace(go.Bar(
        name='Posterior',
        x=states,
        y=posterior_values,
        marker_color='#F44336',
        opacity=0.7,
        text=[f'{v:.3f}' for v in posterior_values],
        textposition='auto',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="States",
        yaxis_title="Probability",
        barmode='group',
        yaxis=dict(range=[0, max(max(prior_values), max(posterior_values)) * 1.2]),
        legend=dict(x=0.02, y=0.98),
        height=400,
        template="plotly_white"
    )

    return fig


def create_radar_chart(posteriors: dict, _labels: list[str], title: str = "") -> go.Figure:
    """Create radar chart comparing multiple posteriors."""
    _ = _labels  # Acknowledge unused parameter
    fig = go.Figure()

    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#607D8B']

    for i, (label, posterior) in enumerate(posteriors.items()):
        states = list(posterior.keys())
        values = [posterior[s] for s in states]
        # Close the loop
        states_closed = states + [states[0]]
        values_closed = values + [values[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=states_closed,
            fill='toself',
            name=label,
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)],
            opacity=0.2
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title=title,
        height=400,
        template="plotly_white"
    )

    return fig


def run_scenario(scenario_id: str, custom_evidence: dict | None = None) -> dict:
    """Run inference for a given scenario."""
    scenario = next(s for s in scenarios if s["id"] == scenario_id)
    evidence = custom_evidence if custom_evidence else scenario["evidence"]
    query = scenario["query"]
    posterior = student_core.infer_posterior(evidence, query, bn_schema, cpt)
    prior = compute_prior(query, bn_schema, cpt)
    return {
        "scenario": scenario,
        "evidence": evidence,
        "query": query,
        "posterior": posterior,
        "prior": prior
    }


# ==========================================
# SIDEBAR - EVIDENCE & CONTROLS
# ==========================================
with st.sidebar:
    st.header("🔧 Evidence Configuration")
    st.caption("Select variables to observe and their values")

    # Evidence toggles
    evidence: dict[str, str] = {}
    for name in node_names:
        with st.expander(f"{name}", expanded=False):
            include = st.checkbox(f"Observe", key=f"include_{name}")
            if include:
                states = node_by_name[name]["states"]
                value = st.selectbox("Value", states, key=f"value_{name}")
                evidence[name] = value

    st.divider()

    # Query selection
    query = st.selectbox(
        "🎯 Query Variable",
        node_names,
        index=node_names.index("Performance"),
        key="query_select"
    ) or "Performance"

    st.divider()

    # Scenario loader
    st.header("📋 Benchmark Scenarios")
    scenario_option = st.selectbox(
        "Load Scenario",
        ["(Custom Evidence)"] + [s["id"] for s in scenarios],
        key="scenario_select"
    )

    if scenario_option != "(Custom Evidence)":
        chosen = next(s for s in scenarios if s["id"] == scenario_option)
        st.caption(chosen["description"])
        with st.expander("Evidence Details"):
            st.json(chosen.get("evidence", {}))

        if st.button("🔄 Apply Scenario Evidence", use_container_width=True):
            # Update session state to reflect scenario evidence
            for var, val in chosen["evidence"].items():
                st.session_state[f"include_{var}"] = True
                st.session_state[f"value_{var}"] = val
            st.session_state["query_select"] = chosen["query"]
            st.rerun()

    st.divider()

    # Compute button
    compute_clicked = st.button("🚀 Compute Posterior", type="primary", use_container_width=True)

    st.divider()
    st.caption("All data is synthetic educational simulation.")

# ==========================================
# MAIN CONTENT - TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Network & Inference",
    "📈 Prior vs Posterior",
    "⚖️ Scenario Comparison",
    "🧪 Benchmark Scenarios",
    "🔬 Sensitivity Analysis"
])

# Compute posterior if button clicked or scenario loaded
posterior = None
prior = None
if compute_clicked or scenario_option != "(Custom Evidence)":
    try:
        if scenario_option != "(Custom Evidence)":
            chosen = next(s for s in scenarios if s["id"] == scenario_option)
            evidence = chosen["evidence"]
            query = chosen["query"]
        # query is guaranteed to be str from sidebar selectbox
        assert isinstance(query, str)

        posterior = student_core.infer_posterior(evidence, query, bn_schema, cpt)
        prior = compute_prior(query, bn_schema, cpt)
    except Exception as e:
        st.error(f"Error computing posterior: {e}")

# ==========================================
# TAB 1: NETWORK & INFERENCE
# ==========================================
with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🕸️ Interactive Bayesian Network")
        st.caption("Hover nodes for details. Drag to rearrange. Use controls to zoom/pan.")

        # Render network graph
        graph_html = create_network_graph()
        st.components.v1.html(graph_html, height=520)  # type: ignore[attr-defined]

        # Network structure details
        with st.expander("📐 Network Structure (JSON)"):
            st.json({
                "nodes": [{"name": n["name"], "states": n["states"], "parents": n["parents"]} for n in bn_schema["nodes"]],
                "edges": bn_schema["edges"],
                "independence_notes": bn_schema.get("independence_notes", [])
            })

    with col_right:
        st.subheader("📋 Current Inference")

        # Evidence summary
        if evidence:
            st.write("**Evidence:**")
            evidence_df = pd.DataFrame([
                {"Variable": k, "Observed Value": v} for k, v in evidence.items()
            ])
            st.dataframe(evidence_df, hide_index=True, use_container_width=True)
        else:
            st.info("No evidence selected. Posterior equals prior.")

        st.write(f"**Query Variable:** `{query}`")

        if posterior:
            st.write("**Posterior Distribution:**")
            post_df = pd.DataFrame([
                {"State": k, "Probability": f"{v:.4f}", "Percentage": f"{v*100:.1f}%"}
                for k, v in posterior.items()
            ])
            st.dataframe(post_df, hide_index=True, use_container_width=True)

            # Bar chart
            fig = px.bar(
                x=list(posterior.keys()),
                y=list(posterior.values()),
                labels={'x': 'State', 'y': 'Probability'},
                color=list(posterior.values()),
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False, height=300, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Most likely outcome
            max_state = get_max_state(posterior)
            max_prob = posterior[max_state]
            st.metric("Most Likely Outcome", max_state, f"{max_prob*100:.1f}% probability")

# ==========================================
# TAB 2: PRIOR VS POSTERIOR
# ==========================================
with tab2:
    st.subheader("📊 Side-by-Side Prior vs Posterior Comparison")

    if posterior and prior:
        # Side-by-side bar chart
        fig = create_distribution_chart(prior, posterior, query, f"Prior vs Posterior for {query}")
        st.plotly_chart(fig, use_container_width=True)

        # Detailed comparison table
        st.subheader("📋 Detailed Comparison")
        comparison_data = []
        for state in prior.keys():
            prior_val = prior[state]
            post_val = posterior[state]
            drift = post_val - prior_val
            comparison_data.append({
                "State": state,
                "Prior": f"{prior_val:.4f}",
                "Posterior": f"{post_val:.4f}",
                "Absolute Drift": f"{drift:+.4f}",
                "Relative Change": f"{drift/prior_val*100:+.1f}%" if prior_val > 0 else "N/A"
            })

        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        # KL Divergence
        kl_div = sum(
            posterior[s] * math.log(posterior[s] / prior[s])
            for s in posterior if posterior[s] > 0 and prior[s] > 0
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("KL Divergence", f"{kl_div:.6f}")
        with col2:
            total_drift = sum(abs(posterior[s] - prior[s]) for s in posterior)
            st.metric("Total Absolute Drift", f"{total_drift:.4f}")
        with col3:
            max_drift = max(abs(posterior[s] - prior[s]) for s in posterior)
            st.metric("Max State Drift", f"{max_drift:.4f}")

        # Interpretation
        st.subheader("💡 Interpretation")
        if total_drift < 0.1:
            st.info("📉 **Weak evidence**: Posterior remains close to prior. The observed evidence has little discriminative power.")
        elif total_drift < 0.3:
            st.warning("📊 **Moderate evidence**: Posterior shows noticeable shift from prior. Evidence provides meaningful information.")
        else:
            st.success("📈 **Strong evidence**: Posterior differs substantially from prior. Evidence is highly informative.")
    else:
        st.info("Compute a posterior in the Network & Inference tab to see the comparison.")

# ==========================================
# TAB 3: SCENARIO COMPARISON (Two Profiles)
# ==========================================
with tab3:
    st.subheader("⚖️ Two Student Profile Comparison")
    st.caption("Compare two hypothetical student profiles side by side")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 👤 Student Profile A")
        evidence_a = {}
        for name in node_names:
            with st.expander(f"{name}", expanded=False):
                include = st.checkbox(f"Observe A: {name}", key=f"include_a_{name}")
                if include:
                    states = node_by_name[name]["states"]
                    value = st.selectbox(f"Value A", states, key=f"value_a_{name}")
                    evidence_a[name] = value

        # Quick scenario buttons for Profile A
        st.write("**Quick Load:**")
        quick_cols = st.columns(2)
        for i, scenario in enumerate(scenarios[:2]):
            with quick_cols[i]:
                if st.button(scenario["id"], key=f"quick_a_{i}", use_container_width=True):
                    for var, val in scenario["evidence"].items():
                        st.session_state[f"include_a_{var}"] = True
                        st.session_state[f"value_a_{var}"] = val
                    st.rerun()

    with col_b:
        st.markdown("### 👤 Student Profile B")
        evidence_b = {}
        for name in node_names:
            with st.expander(f"{name}", expanded=False):
                include = st.checkbox(f"Observe B: {name}", key=f"include_b_{name}")
                if include:
                    states = node_by_name[name]["states"]
                    value = st.selectbox(f"Value B", states, key=f"value_b_{name}")
                    evidence_b[name] = value

        # Quick scenario buttons for Profile B
        st.write("**Quick Load:**")
        quick_cols = st.columns(2)
        for i, scenario in enumerate(scenarios[2:]):
            with quick_cols[i]:
                if st.button(scenario["id"], key=f"quick_b_{i}", use_container_width=True):
                    for var, val in scenario["evidence"].items():
                        st.session_state[f"include_b_{var}"] = True
                        st.session_state[f"value_b_{var}"] = val
                    st.rerun()

    st.divider()

    # Compare button
    if st.button("🔍 Compare Profiles", type="primary", use_container_width=True):
        if not evidence_a and not evidence_b:
            st.warning("Please configure at least one profile with evidence.")
        else:
            query_compare = st.selectbox("Query Variable for Comparison", node_names, index=node_names.index("Performance"), key="compare_query") or "Performance"

            # Compute posteriors
            posterior_a = student_core.infer_posterior(evidence_a, query_compare, bn_schema, cpt) if evidence_a else compute_prior(query_compare, bn_schema, cpt)
            posterior_b = student_core.infer_posterior(evidence_b, query_compare, bn_schema, cpt) if evidence_b else compute_prior(query_compare, bn_schema, cpt)
            prior_compare = compute_prior(query_compare, bn_schema, cpt)

            # Side-by-side bar chart
            states = list(prior_compare.keys())

            fig = go.Figure()
            fig.add_trace(go.Bar(name='Prior', x=states, y=[prior_compare[s] for s in states], marker_color='#9E9E9E', opacity=0.6))
            fig.add_trace(go.Bar(name='Profile A', x=states, y=[posterior_a[s] for s in states], marker_color='#2196F3', opacity=0.8))
            fig.add_trace(go.Bar(name='Profile B', x=states, y=[posterior_b[s] for s in states], marker_color='#F44336', opacity=0.8))

            fig.update_layout(
                title=f"Prior vs Profile A vs Profile B for {query_compare}",
                xaxis_title="States",
                yaxis_title="Probability",
                barmode='group',
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Radar chart
            st.subheader("📡 Radar Comparison")
            radar_fig = create_radar_chart(
                {"Prior": prior_compare, "Profile A": posterior_a, "Profile B": posterior_b},
                ["Prior", "Profile A", "Profile B"],
                f"Distribution Shape Comparison for {query_compare}"
            )
            st.plotly_chart(radar_fig, use_container_width=True)

            # Comparison table
            st.subheader("📊 Numerical Comparison")
            comp_data = []
            for state in states:
                comp_data.append({
                    "State": state,
                    "Prior": f"{prior_compare[state]:.4f}",
                    "Profile A": f"{posterior_a[state]:.4f}",
                    "Profile B": f"{posterior_b[state]:.4f}",
                    "A vs Prior": f"{posterior_a[state] - prior_compare[state]:+.4f}",
                    "B vs Prior": f"{posterior_b[state] - prior_compare[state]:+.4f}",
                    "A vs B": f"{posterior_a[state] - posterior_b[state]:+.4f}"
                })
            st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

            # KL Divergences
            st.subheader("📏 Divergence Metrics")
            kl_a = sum(
                posterior_a[s] * math.log(posterior_a[s] / prior_compare[s])
                for s in posterior_a if posterior_a[s] > 0 and prior_compare[s] > 0
            )
            kl_b = sum(
                posterior_b[s] * math.log(posterior_b[s] / prior_compare[s])
                for s in posterior_b if posterior_b[s] > 0 and prior_compare[s] > 0
            )
            kl_ab = sum(
                posterior_a[s] * math.log(posterior_a[s] / posterior_b[s])
                for s in posterior_a if posterior_a[s] > 0 and posterior_b[s] > 0
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("KL(Profile A || Prior)", f"{kl_a:.6f}")
            with col2:
                st.metric("KL(Profile B || Prior)", f"{kl_b:.6f}")
            with col3:
                st.metric("KL(Profile A || Profile B)", f"{kl_ab:.6f}")

            # Most likely outcomes
            st.subheader("🏆 Most Likely Outcomes")
            max_a = get_max_state(posterior_a)
            max_b = get_max_state(posterior_b)
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Profile A**: {max_a} ({posterior_a[max_a]*100:.1f}%)")
            with col2:
                st.success(f"**Profile B**: {max_b} ({posterior_b[max_b]*100:.1f}%)")

# ==========================================
# TAB 4: BENCHMARK SCENARIOS
# ==========================================
with tab4:
    st.subheader("🧪 All Four Benchmark Scenarios")
    st.caption("Demonstrating correct probabilistic reasoning across stress & edge cases")

    # Run all scenarios
    all_results = {}
    for scenario in scenarios:
        result = run_scenario(scenario["id"])
        all_results[scenario["id"]] = result

    # Summary cards
    st.subheader("📋 Scenario Summary")
    cols = st.columns(4)
    for i, (scenario_id, result) in enumerate(all_results.items()):
        with cols[i]:
            scenario = result["scenario"]
            posterior = result["posterior"]
            max_state = get_max_state(posterior)
            max_prob = posterior[max_state]

            st.markdown(f"""
            <div class="scenario-card">
                <h4>{scenario_id}</h4>
                <p><small>{scenario['description'][:100]}...</small></p>
                <hr style="margin: 8px 0;">
                <strong>Query:</strong> {result['query']}<br>
                <strong>Evidence:</strong> {len(result['evidence'])} variables<br>
                <strong>Top Prediction:</strong> {max_state} ({max_prob*100:.1f}%)
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Detailed view for each scenario
    for scenario_id, result in all_results.items():
        with st.expander(f"🔍 {scenario_id} - Detailed Analysis", expanded=False):
            scenario = result["scenario"]
            evidence = result["evidence"]
            query = result["query"]
            posterior = result["posterior"]
            prior = result["prior"]

            col1, col2 = st.columns([1, 2])

            with col1:
                st.write("**Scenario Description:**")
                st.write(scenario["description"])

                st.write("**Evidence:**")
                ev_df = pd.DataFrame([{"Variable": k, "Value": v} for k, v in evidence.items()])
                st.dataframe(ev_df, hide_index=True, use_container_width=True)

                st.write("**Expected Behavior:**")
                st.info(scenario.get("expected_direction", "N/A"))

            with col2:
                # Prior vs Posterior chart
                fig = create_distribution_chart(prior, posterior, query, f"{scenario_id}: Prior vs Posterior")
                st.plotly_chart(fig, use_container_width=True)

                # Posterior table
                post_df = pd.DataFrame([
                    {"State": k, "Probability": f"{v:.4f}", "%": f"{v*100:.1f}%"}
                    for k, v in posterior.items()
                ])
                st.dataframe(post_df, hide_index=True, use_container_width=True)

    # Combined comparison chart
    st.subheader("📊 All Scenarios: Posterior Comparison")

    # Create grouped bar chart for all scenarios
    fig = go.Figure()
    states = node_by_name["Performance"]["states"]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    for i, (scenario_id, result) in enumerate(all_results.items()):
        posterior = result["posterior"]
        fig.add_trace(go.Bar(
            name=scenario_id,
            x=states,
            y=[posterior[s] for s in states],
            marker_color=colors[i],
            opacity=0.8
        ))

    # Add prior as reference
    prior_perf = compute_prior("Performance", bn_schema, cpt)
    fig.add_trace(go.Bar(
        name="Prior (Reference)",
        x=states,
        y=[prior_perf[s] for s in states],
        marker_color='#9E9E9E',
        opacity=0.5,
        marker_pattern_shape="/"
    ))

    fig.update_layout(
        title="Posterior Distributions Across All Benchmark Scenarios",
        xaxis_title="Performance State",
        yaxis_title="Probability",
        barmode='group',
        height=500,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart comparison
    st.subheader("📡 Distribution Shape Comparison (Radar)")
    radar_data = {sid: res["posterior"] for sid, res in all_results.items()}
    radar_data["Prior"] = prior_perf
    radar_fig = create_radar_chart(radar_data, list(radar_data.keys()), "Scenario Posterior Shapes")
    st.plotly_chart(radar_fig, use_container_width=True)

# ==========================================
# TAB 5: SENSITIVITY ANALYSIS
# ==========================================
with tab5:
    st.subheader("🔬 CPT Sensitivity Analysis")
    st.caption("Required experiment: P05-CPT-SENSITIVITY scenario with parameter perturbations")

    # Load sensitivity scenario
    sens_scenario = next(s for s in scenarios if s["id"] == "P05-CPT-SENSITIVITY")
    baseline_evidence = sens_scenario["evidence"]
    sensitivity_query = sens_scenario["query"]

    st.write("**Baseline Scenario:**")
    st.json(baseline_evidence)
    st.write(f"**Query:** {sensitivity_query}")
    st.write(f"**Description:** {sens_scenario['description']}")

    # Baseline posterior
    baseline_posterior = student_core.infer_posterior(baseline_evidence, sensitivity_query, bn_schema, cpt)
    st.write("**Baseline Posterior:**")
    st.bar_chart(baseline_posterior)

    st.divider()

    # Experiment 1: Modify CPT row
    st.subheader("🧪 Experiment 1: CPT Row Modification")
    st.caption(sens_scenario["required_experiment"])

    # Let user choose which CPT row to modify
    perf_cpt = cpt["Performance"]
    parent_combos = list(perf_cpt["table"].keys())

    col1, col2 = st.columns(2)
    with col1:
        selected_combo = st.selectbox(
            "Select CPT Row to Modify (AssignmentCompletion | ExamDifficulty)",
            parent_combos,
            index=parent_combos.index("Medium|Medium") if "Medium|Medium" in parent_combos else 0
        )

    with col2:
        st.write("Current values:")
        current_row = perf_cpt["table"][selected_combo]
        for i, (state, prob) in enumerate(zip(perf_cpt["states"], current_row)):
            st.write(f"  {state}: {prob:.3f}")

    # New values input
    st.write("**New Probability Values (must sum to 1.0):**")
    new_probs = []
    prob_cols = st.columns(len(perf_cpt["states"]))
    for i, state in enumerate(perf_cpt["states"]):
        with prob_cols[i]:
            val = st.number_input(
                f"P({state} | {selected_combo})",
                min_value=0.0,
                max_value=1.0,
                value=float(current_row[i]),
                step=0.01,
                format="%.3f",
                key=f"new_prob_{i}"
            )
            new_probs.append(val)

    total = sum(new_probs)
    if abs(total - 1.0) > 0.001:
        st.warning(f"⚠️ Probabilities sum to {total:.3f}, not 1.0. Normalizing...")
        new_probs = [p/total for p in new_probs]

    if st.button("Run Experiment 1", type="primary"):
        import copy
        modified_cpt = copy.deepcopy(cpt)
        modified_cpt["Performance"]["table"][selected_combo] = new_probs

        try:
            exp1_result = student_core.sensitivity_experiment(
                baseline_evidence,
                sensitivity_query,
                bn_schema,
                cpt,
                modified_cpt
            )

            st.success("Experiment 1 completed!")

            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("KL Divergence", f"{exp1_result['kl_divergence'][0]:.6f}")
            with col2:
                st.metric("Total Drift", f"{sum(exp1_result['absolute_drift']):.4f}")
            with col3:
                st.metric("Max Drift", f"{max(exp1_result['absolute_drift']):.4f}")

            # Comparison chart
            states = perf_cpt["states"]
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Baseline', x=states, y=exp1_result['baseline_posterior'], marker_color='#2196F3', opacity=0.7))
            fig.add_trace(go.Bar(name='Modified', x=states, y=exp1_result['modified_posterior'], marker_color='#F44336', opacity=0.7))
            fig.update_layout(title="Experiment 1: Baseline vs Modified CPT", barmode='group', height=350, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Drift breakdown
            drift_df = pd.DataFrame({
                "State": states,
                "Baseline": [f"{v:.4f}" for v in exp1_result['baseline_posterior']],
                "Modified": [f"{v:.4f}" for v in exp1_result['modified_posterior']],
                "Absolute Drift": [f"{v:+.4f}" for v in exp1_result['absolute_drift']]
            })
            st.dataframe(drift_df, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    # Experiment 2: Change Evidence
    st.subheader("🧪 Experiment 2: Evidence Perturbation")
    st.caption("Change one evidence value from the baseline scenario")

    # Show current evidence
    st.write("**Baseline Evidence:**")
    for var, val in baseline_evidence.items():
        st.write(f"  {var}: {val}")

    # Let user change one evidence
    evidence_var = st.selectbox("Variable to Change", list(baseline_evidence.keys()))
    evidence_new_val = st.selectbox(
        "New Value",
        [s for s in node_by_name[evidence_var]["states"] if s != baseline_evidence[evidence_var]]
    )

    if st.button("Run Experiment 2", type="primary"):
        changed_evidence = dict(baseline_evidence)
        changed_evidence[evidence_var] = evidence_new_val

        exp2_posterior = student_core.infer_posterior(
            changed_evidence,
            sensitivity_query,
            bn_schema,
            cpt
        )

        st.success("Experiment 2 completed!")

        # Comparison
        states = perf_cpt["states"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Baseline', x=states, y=[baseline_posterior[s] for s in states], marker_color='#2196F3', opacity=0.7))
        fig.add_trace(go.Bar(name='Changed Evidence', x=states, y=[exp2_posterior[s] for s in states], marker_color='#FF9800', opacity=0.7))
        fig.update_layout(title=f"Experiment 2: {evidence_var}={baseline_evidence[evidence_var]} → {evidence_new_val}", barmode='group', height=350, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # KL divergence
        kl_div = sum(
            baseline_posterior[s] * math.log(baseline_posterior[s] / exp2_posterior[s])
            for s in baseline_posterior if baseline_posterior[s] > 0 and exp2_posterior[s] > 0
        )

        drift = [abs(exp2_posterior[s] - baseline_posterior[s]) for s in states]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("KL Divergence", f"{kl_div:.6f}")
        with col2:
            st.metric("Total Drift", f"{sum(drift):.4f}")
        with col3:
            st.metric("Max Drift", f"{max(drift):.4f}")

        # Table
        comp_df = pd.DataFrame({
            "State": states,
            "Baseline": [f"{baseline_posterior[s]:.4f}" for s in states],
            f"Evidence {evidence_var}={evidence_new_val}": [f"{exp2_posterior[s]:.4f}" for s in states],
            "Drift": [f"{exp2_posterior[s] - baseline_posterior[s]:+.4f}" for s in states]
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

    st.divider()

    # Predefined experiments from the original app
    st.subheader("📋 Predefined Sensitivity Experiments")

    # Experiment 1 from original: Modify Performance | Medium|Medium
    import copy
    modified_cpt = copy.deepcopy(cpt)
    modified_cpt["Performance"]["table"]["Medium|Medium"] = [0.15, 0.55, 0.30]

    exp1 = student_core.sensitivity_experiment(
        baseline_evidence,
        sensitivity_query,
        bn_schema,
        cpt,
        modified_cpt
    )

    # Experiment 2 from original: Change AssignmentCompletion to High
    changed_evidence = dict(baseline_evidence)
    changed_evidence["AssignmentCompletion"] = "High"

    exp2_posterior = student_core.infer_posterior(
        changed_evidence,
        sensitivity_query,
        bn_schema,
        cpt
    )

    # Display all three
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Baseline**")
        st.bar_chart(baseline_posterior)

    with col2:
        st.write("**Exp 1: CPT Modified**")
        st.caption("Performance | Medium|Medium → [0.15, 0.55, 0.30]")
        mod_posterior = {states[i]: exp1["modified_posterior"][i] for i in range(len(states))}
        st.bar_chart(mod_posterior)
        st.metric("KL Div", f"{exp1['kl_divergence'][0]:.6f}")

    with col3:
        st.write("**Exp 2: AssignmentCompletion=High**")
        st.bar_chart(exp2_posterior)
        kl2 = sum(
            baseline_posterior[s] * math.log(baseline_posterior[s] / exp2_posterior[s])
            for s in baseline_posterior if baseline_posterior[s] > 0 and exp2_posterior[s] > 0
        )
        st.metric("KL Div", f"{kl2:.6f}")

    # Comparison table
    st.subheader("📊 Three-Way Comparison")
    comp_table = {
        "Baseline": baseline_posterior,
        "CPT Modified": mod_posterior,
        "AssignmentCompletion=High": exp2_posterior
    }
    st.dataframe(pd.DataFrame(comp_table), use_container_width=True)

# ==========================================
# FOOTER
# ==========================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Bayesian Student Scenario Simulator</strong> — Final Milestone Implementation</p>
    <p><small>All data is synthetic educational simulation. Not for real student assessment.</small></p>
    <p><small>Built with Streamlit, Plotly, NetworkX, PyVis | Project 05 — AI Core & AI Product</small></p>
</div>
""", unsafe_allow_html=True)

import math  # For KL divergence calculations