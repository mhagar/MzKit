"""
Represents an Ensemble's single general MS1/MS2 representation

This is the generic "show me the Ensemble's MS1/MS2 spectra"
Produced on demand by `Ensemble.composite_spectrum`

For DIA the current definition is a placeholder ("tallest scan").
I plan on later refining this (m/z averaging for DIA; precursor-based
stitching for DDA)

# TODO: Look into redundancy with ConsensusSpectrumArray (?)
"""
from dataclasses import dataclass
from typing import Optional

from core.utils.array_types import SpectrumArray


@dataclass(frozen=True)
class CompositeSpectrum:
    """
    A single representative MS1 + MS2 pair for an Ensemble, plus the precursor
    needed to interpret it.

    :param ms1: representative MS1 spectrum (ensemble cofeature lanes @ apex).
    :param ms2: representative MS2 spectrum, or None if the ensemble has no MS2.
    :param precursor_mz: the ensemble's resolved (possibly virtual) precursor.
    :param charge: the ensemble's resolved charge.
    """
    ms1: SpectrumArray
    ms2: Optional[SpectrumArray]
    precursor_mz: float
    charge: int
