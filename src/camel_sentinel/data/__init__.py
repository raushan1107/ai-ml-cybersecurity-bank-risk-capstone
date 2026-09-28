"""
Phase 1 — data acquisition.

Pulls bank identity, financial, and failure data from the FDIC BankFind
Suite API (see docs/02_data_sources.md), with a schema-matched synthetic
fallback for environments without network access to the live API.
"""
