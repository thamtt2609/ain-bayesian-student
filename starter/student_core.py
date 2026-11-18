"""Graded AI core for Project 05 — YOUR implementation goes here.

This module intentionally ships EMPTY. Implementing ``infer_posterior`` is the graded
midterm AI core (Bayesian inference under evidence). Do not import a third-party exact-inference
or sampling library to replace this function: write and explain your own implementation, whether
it is exact (enumeration) or approximate (sampling).
"""
from __future__ import annotations

from itertools import product
import math
from typing import Dict, List


def _network_parts(bn_schema: dict, cpt: Dict[str, dict]):
    """Return node names and state lists after checking schema/CPT consistency."""
    if "nodes" not in bn_schema or not isinstance(bn_schema["nodes"], list):
        raise ValueError("bn_schema must contain a 'nodes' list")
    nodes = bn_schema["nodes"]
    names = [node["name"] for node in nodes]
    if len(names) != len(set(names)):
        raise ValueError("Bayesian Network contains duplicate node names")
    states = {}
    for node in nodes:
        name = node["name"]
        if name not in cpt:
            raise ValueError(f"Missing CPT for variable '{name}'")
        states[name] = node["states"]
        spec = cpt[name]
        if spec["states"] != node["states"]:
            raise ValueError(f"State order mismatch between schema and CPT for '{name}'")
        if spec["parents"] != node["parents"]:
            raise ValueError(f"Parent mismatch between schema and CPT for '{name}'")
        if not states[name] or len(states[name]) != len(set(states[name])):
            raise ValueError(f"Invalid states for variable '{name}'")
    return nodes, names, states


def _validate_cpt(cpt: Dict[str, dict], names: List[str]) -> None:
    """Check CPT row dimensions, finite probabilities, and normalization."""
    for name in names:
        spec = cpt[name]
        for key, row in spec["table"].items():
            if len(row) != len(spec["states"]):
                raise ValueError(f"CPT row '{name}:{key}' has the wrong number of entries")
            if any(not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0
                   for p in row):
                raise ValueError(f"CPT row '{name}:{key}' contains an invalid probability")
            if not math.isclose(math.fsum(row), 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"CPT row '{name}:{key}' does not sum to 1.0")


def infer_posterior(
    evidence: Dict[str, str],
    query: str,
    bn_schema: dict,
    cpt: Dict[str, dict],
) -> Dict[str, float]:
    """Compute P(query | evidence) by exact enumeration.

    Evidence is a mapping from observed variable names to observed states.
    The returned dictionary follows the state order in bn_schema and sums to 1.
    """
    if not isinstance(evidence, dict):
        raise TypeError("evidence must be a dictionary")
    nodes, variable_names, variable_states = _network_parts(bn_schema, cpt)
    _validate_cpt(cpt, variable_names)

    if query not in variable_states:
        raise ValueError(f"Unknown query variable: '{query}'")
    for variable, value in evidence.items():
        if variable not in variable_states:
            raise ValueError(f"Unknown evidence variable: '{variable}'")
        if value not in variable_states[variable]:
            raise ValueError(f"Invalid evidence state '{value}' for '{variable}'")

    query_states = variable_states[query]
    # If the query itself is observed, its posterior is a point mass.
    if query in evidence:
        return {state: float(state == evidence[query]) for state in query_states}

    hidden_vars = [
        variable for variable in variable_names
        if variable not in evidence and variable != query
    ]
    hidden_state_lists = [variable_states[var] for var in hidden_vars]
    results = {state: 0.0 for state in query_states}

    # Enumerate every assignment to hidden variables and every query state.
    for hidden_values in product(*hidden_state_lists):
        hidden_assignment = dict(zip(hidden_vars, hidden_values))
        for query_state in query_states:
            assignment = {**evidence, **hidden_assignment, query: query_state}
            joint_probability = 1.0

            # Bayesian Network factorization:
            # P(x1,...,xn) = product_i P(x_i | Parents(X_i)).
            for variable in variable_names:
                spec = cpt[variable]
                parent_values = [assignment[parent] for parent in spec["parents"]]
                row_key = "|".join(parent_values) if parent_values else ""
                probability_row = spec["table"][row_key]
                state_index = spec["states"].index(assignment[variable])
                joint_probability *= probability_row[state_index]

            results[query_state] += joint_probability

    normalization_constant = math.fsum(results.values())
    if normalization_constant <= 0.0:
        raise ValueError("Evidence has zero probability under the supplied Bayesian Network")

    return {
        state: probability / normalization_constant
        for state, probability in results.items()
    }


def sensitivity_experiment(
    baseline_evidence: Dict[str, str],
    query: str,
    bn_schema: dict,
    cpt: Dict[str, dict],
    modified_cpt: Dict[str, dict],
) -> Dict[str, List[float]]:
    """Compare baseline and modified posteriors using drift and KL divergence.

    KL is D_KL(P_baseline || P_modified). It is infinity if a state has
    positive baseline probability and zero modified probability.
    """
    baseline = infer_posterior(baseline_evidence, query, bn_schema, cpt)
    modified = infer_posterior(baseline_evidence, query, bn_schema, modified_cpt)

    states = list(baseline)
    absolute_drift = [
        abs(modified[state] - baseline[state]) for state in states
    ]
    kl_divergence = 0.0
    for state in states:
        p = baseline[state]
        q = modified[state]
        if p > 0.0 and q == 0.0:
            kl_divergence = math.inf
            break
        if p > 0.0:
            kl_divergence += p * math.log(p / q)

    return {
        "kl_divergence": [kl_divergence],
        "absolute_drift": absolute_drift,
        "baseline_posterior": [baseline[state] for state in states],
        "modified_posterior": [modified[state] for state in states],
    }
