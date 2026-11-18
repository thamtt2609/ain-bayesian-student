"""Run the required CPT sensitivity experiment and print reproducible results."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from starter import loader, student_core


def main():
    schema = loader.load_bn_schema()
    base_cpt = loader.load_base_cpt()
    scenario = next(s for s in loader.load_scenarios()
                    if s["id"] == "P05-CPT-SENSITIVITY")
    evidence = scenario["evidence"]
    query = scenario["query"]

    # Experiment A: perturb exactly one CPT row; probabilities still sum to 1.
    modified_cpt = copy.deepcopy(base_cpt)
    modified_cpt["Performance"]["table"]["Medium|Medium"] = [0.15, 0.55, 0.30]
    result_a = student_core.sensitivity_experiment(
        evidence, query, schema, base_cpt, modified_cpt
    )

    # Experiment B: change one evidence value while keeping the baseline CPT.
    changed_evidence = dict(evidence)
    changed_evidence["AssignmentCompletion"] = "High"
    baseline = student_core.infer_posterior(evidence, query, schema, base_cpt)
    changed = student_core.infer_posterior(changed_evidence, query, schema, base_cpt)

    print("BASELINE EVIDENCE:", json.dumps(evidence, ensure_ascii=False))
    print("BASELINE POSTERIOR:", json.dumps(baseline, indent=2))
    print("\\nEXPERIMENT A — modified CPT row Performance | Medium|Medium")
    print(json.dumps(result_a, indent=2))
    print("\\nEXPERIMENT B — AssignmentCompletion changed from Medium to High")
    print("Changed evidence:", json.dumps(changed_evidence))
    print("Changed posterior:", json.dumps(changed, indent=2))
    print("Absolute drift by state:", json.dumps({
        state: abs(changed[state] - baseline[state]) for state in baseline
    }, indent=2))


if __name__ == "__main__":
    main()
