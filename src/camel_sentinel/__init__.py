"""
camel_sentinel — the reusable code behind the CAMEL Sentinel project.

WHAT this package is: the actual, importable, testable logic for the
project (data acquisition, feature engineering, and — in later phases —
modeling, XAI, RAI, federated learning, and the agent).

WHY it's separate from notebooks/: notebooks are for walking through and
visualizing a phase step by step; they should stay thin and call into this
package rather than contain the real logic inline. That way the same
tested code is reusable from a notebook, the FastAPI backend, and the
agent, instead of being copy-pasted between them.

See docs/03_module_mapping.md for which submodule corresponds to which
build phase.
"""

__version__ = "0.1.0"
