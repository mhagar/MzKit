"""
FormulaQuery: inputs for one MS2-aware find-mfs query

Built by core/formula/extract.py from an Ensemble (or, later, any MS source) and
consumed by core/formula/assign_formula.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.utils.array_types import SpectrumArray
    from core.data_structs.uuid_types import EnsembleUUID


@dataclass
class FormulaQuery:
    precursor_mz: float
    source_uuid: 'EnsembleUUID'

    # Observed spectra (MzKit SpectrumArray; dtype-compatible with find-mfs).
    ms1_spec: Optional['SpectrumArray'] = None
    ms2_spec: Optional['SpectrumArray'] = None
    charge: int = 1

    # find-mfs steering. adducts=None -> find-mfs DEFAULT_ADDUCTS (joint ranking).
    adducts: Optional[list[str]] = None
    elements: str = 'CHNOPS'
    autodetect_cl_br: bool = True
    error_ppm: float = 5.0
    instrument: str = 'unknown'
    ms2_weight: float = 1.0

    # Recorded for provenance on the resulting FormulaAssignment.
    ms2_mode: str = 'tallest'

    finder_kwargs: dict = field(default_factory=dict)
