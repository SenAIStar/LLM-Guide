"""Data preparation helpers for legal-domain SFT."""

from .data_pipeline import (
    convert_disc_pair,
    convert_disc_triplet,
    deduplicate,
    quality_report,
    split_by_group,
)
from .pii import PiiRedactor

__all__ = [
    "PiiRedactor",
    "convert_disc_pair",
    "convert_disc_triplet",
    "deduplicate",
    "quality_report",
    "split_by_group",
]
__version__ = "0.1.0"
