"""Observability layer for the SMB Underwriting preprocessing pipeline.

Wraps the leakage-safe ``preprocess.py`` transform with (1) automated
integrity / quality / drift monitors and (2) regression fixtures pinned to the
verified golden values. See the build brief and ``contracts.py``.
"""
