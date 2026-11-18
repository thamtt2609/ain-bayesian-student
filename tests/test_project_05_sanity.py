"""Sanity tests for Project 05. These check data integrity and starter imports.

They do NOT grade the student's Bayesian inference implementation.
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from starter import loader, student_core
from scripts import generate_students


@pytest.fixture
def network():
    return loader.load_bn_schema(), loader.load_base_cpt()


def test_schema_has_six_expected_nodes(network):
    schema, _ = network
    assert {n["name"] for n in schema["nodes"]} == {
        "PriorPreparation", "Attendance", "ExamDifficulty",
        "StudyConsistency", "AssignmentCompletion", "Performance"
    }


def test_all_cpt_rows_sum_to_one(network):
    _, cpt = network
    for spec in cpt.values():
        for row in spec["table"].values():
            assert sum(row) == pytest.approx(1.0, abs=1e-6)
            assert len(row) == len(spec["states"])


def test_sample_is_deterministic_and_has_1000_rows():
    rows1 = generate_students.generate_students(1000, 42)
    rows2 = generate_students.generate_students(1000, 42)
    assert rows1 == rows2
    assert len(loader.load_students()) == 1000


def test_no_evidence_returns_normalized_prior(network):
    schema, cpt = network
    result = student_core.infer_posterior({}, "Performance", schema, cpt)
    assert list(result) == ["Fail", "Pass", "Distinction"]
    assert sum(result.values()) == pytest.approx(1.0)
    assert all(0 <= p <= 1 for p in result.values())


def test_query_observed_returns_point_mass(network):
    schema, cpt = network
    result = student_core.infer_posterior(
        {"Performance": "Pass"}, "Performance", schema, cpt
    )
    assert result == {"Fail": 0.0, "Pass": 1.0, "Distinction": 0.0}


def test_invalid_query_and_evidence_are_rejected(network):
    schema, cpt = network
    with pytest.raises(ValueError):
        student_core.infer_posterior({}, "Unknown", schema, cpt)
    with pytest.raises(ValueError):
        student_core.infer_posterior({"Attendance": "Sometimes"}, "Performance", schema, cpt)
    with pytest.raises(ValueError):
        student_core.infer_posterior({"Unknown": "High"}, "Performance", schema, cpt)


def test_little_evidence_scenario_is_valid_and_normalized(network):
    schema, cpt = network
    scenario = next(s for s in loader.load_scenarios() if s["id"] == "P05-LITTLE-EVIDENCE")
    result = student_core.infer_posterior(scenario["evidence"], scenario["query"], schema, cpt)
    assert sum(result.values()) == pytest.approx(1.0)


def test_strong_positive_scenario_shifts_to_pass_or_distinction(network):
    schema, cpt = network
    prior = student_core.infer_posterior({}, "Performance", schema, cpt)
    scenario = next(s for s in loader.load_scenarios() if s["id"] == "P05-STRONG-POSITIVE")
    posterior = student_core.infer_posterior(scenario["evidence"], "Performance", schema, cpt)
    assert posterior["Pass"] + posterior["Distinction"] > prior["Pass"] + prior["Distinction"]


def test_conflicting_scenario_produces_valid_distribution(network):
    schema, cpt = network
    scenario = next(s for s in loader.load_scenarios() if s["id"] == "P05-CONFLICTING")
    posterior = student_core.infer_posterior(scenario["evidence"], "Performance", schema, cpt)
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert all(math.isfinite(p) for p in posterior.values())


def test_sensitivity_experiment_reports_changes(network):
    schema, cpt = network
    modified = copy.deepcopy(cpt)
    modified["Performance"]["table"]["Medium|Medium"] = [0.15, 0.55, 0.30]
    result = student_core.sensitivity_experiment(
        {"AssignmentCompletion": "Medium", "ExamDifficulty": "Medium"},
        "Performance", schema, cpt, modified
    )
    assert len(result["absolute_drift"]) == 3
    assert sum(result["baseline_posterior"]) == pytest.approx(1.0)
    assert sum(result["modified_posterior"]) == pytest.approx(1.0)
    assert result["kl_divergence"][0] >= 0
    assert any(value > 0 for value in result["absolute_drift"])


def test_cpt_sensitivity_rejects_invalid_modified_row(network):
    schema, cpt = network
    modified = copy.deepcopy(cpt)
    modified["Performance"]["table"]["Medium|Medium"] = [0.2, 0.2, 0.2]
    with pytest.raises(ValueError):
        student_core.sensitivity_experiment(
            {"AssignmentCompletion": "Medium", "ExamDifficulty": "Medium"},
            "Performance", schema, cpt, modified
        )
