"""
Tests for Ensemble class + methods
"""
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np

if TYPE_CHECKING:
    from core.data_structs import Sample, ScanArray


def test_composite_spectrum_generation(ensemble):
    ensemble.get_composite_spectrum(
        ms_level=1,
        fraction=0.8,
    )





