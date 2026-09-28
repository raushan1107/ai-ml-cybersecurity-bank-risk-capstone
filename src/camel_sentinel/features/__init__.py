"""
Phase 1/2 — feature engineering.

Turns raw FDIC Call Report fields into the six CAMEL ratio groups, and the
ratio groups into a single composite health score / proxy rating per
bank-quarter. See docs/00_problem_statement.md for why a proxy is needed,
and the "recipe" table in ../../docs/CAMEL-Rating-ML-Project-Blueprint.md
(Section 3) for the ratio choices.
"""
