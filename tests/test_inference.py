
import json
import math
from pathlib import Path

import pytest

from starter.student_core import infer_posterior


# --------------------------------------------------
# 1. Load the original Bayesian Network data
# --------------------------------------------------

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define paths to the original JSON files
SCHEMA_PATH = PROJECT_ROOT / "data" / "bn_schema.json"
CPT_PATH = PROJECT_ROOT / "data" / "base_cpt.json"


@pytest.fixture(scope="module")
def bn_schema():
    """Load the Bayesian Network schema."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="module")
def cpt():
    """Load the original Conditional Probability Tables."""
    with open(CPT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# 2. Test the infer_posterior() function
# --------------------------------------------------

def test_infer_posterior_returns_distribution(bn_schema, cpt):
    """
    Check that infer_posterior() returns a probability distribution.
    """

    result = infer_posterior(
        evidence={},
        query="Performance",
        bn_schema=bn_schema,
        cpt=cpt
    )

    assert isinstance(result, dict)
    assert len(result) > 0


def test_posterior_probabilities_are_valid(bn_schema, cpt):
    """
    Check that every posterior probability is between 0 and 1.
    """

    result = infer_posterior(
        evidence={},
        query="Performance",
        bn_schema=bn_schema,
        cpt=cpt
    )

    for probability in result.values():
        assert isinstance(probability, (int, float))
        assert 0 <= probability <= 1
        assert math.isfinite(probability)


def test_posterior_probabilities_sum_to_one(bn_schema, cpt):
    """
    Check that the posterior distribution is normalized.
    The sum of all probabilities must equal 1.
    """

    result = infer_posterior(
        evidence={},
        query="Performance",
        bn_schema=bn_schema,
        cpt=cpt
    )

    total_probability = sum(result.values())

    assert total_probability == pytest.approx(1.0)


def test_inference_with_evidence(bn_schema, cpt):
    """
    Check that inference works when evidence is provided.
    """

    result = infer_posterior(
        evidence={"PriorPreparation": "High"},
        query="Performance",
        bn_schema=bn_schema,
        cpt=cpt
    )

    assert isinstance(result, dict)
    assert len(result) > 0

    # The posterior distribution must still be normalized.
    assert sum(result.values()) == pytest.approx(1.0)


def test_inference_returns_expected_performance_states(bn_schema, cpt):
    """
    Check that the query result contains the Performance states
    defined in the original project data.
    """

    result = infer_posterior(
        evidence={},
        query="Performance",
        bn_schema=bn_schema,
        cpt=cpt
    )

    expected_states = {"Fail", "Pass", "Distinction"}

    assert set(result.keys()) == expected_states


def test_inference_is_deterministic(bn_schema, cpt):
    """
    Check that the same input produces the same result.
    """

    first_result = infer_posterior(
        evidence={"PriorPreparation": "High"},
        query="Performance",
        bn_schema=bn_schema,
        cpt=cpt
    )

    second_result = infer_posterior(
        evidence={"PriorPreparation": "High"},
        query="Performance",
        bn_schema=bn_schema,
        cpt=cpt
    )

    assert first_result == second_result
