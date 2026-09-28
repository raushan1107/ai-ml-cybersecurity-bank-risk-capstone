"""
Phase 3 — baseline modeling.

Trains a model on the real, independent `failed` label (not the rule-based
composite score, which is derived from the same features and would make
"prediction" circular) and reports metrics that make sense under severe
class imbalance. See docs/00_problem_statement.md section 3 for why two
separate labels exist and how each is used.
"""
