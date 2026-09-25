"""
FormulaAssignment: a molecular-formula assignment for a single MS source
(currently an Ensemble), produced by core/formula/assign_formula.py.

Standalone entity keyed by source UUID (NOT embedded on the already-mega
Ensemble), so the same machinery can serve non-Ensemble MS data and
AlignedAnalytes can reconcile per-sample assignments without owning them.

`AssignedCandidate` is deliberately primitive-only (no find-mfs types), so the
whole ranked result — including every score term — round-trips losslessly
through the .mzk format.
"""
from __future__ import annotations

import time
import uuid as uuid_module
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.data_structs.uuid_types import AssignmentUUID, EnsembleUUID


@dataclass
class AssignedCandidate:
    """
    One ranked (formula, adduct) candidate with its full score breakdown.

    A `manual` candidate was typed in by the user rather than found by a
    search; its mass terms are None when no adduct puts it near the precursor.
    """
    formula_str: str
    adduct: Optional[str]
    error_ppm: Optional[float]
    error_da: Optional[float]
    rdbe: Optional[float]
    mass_loglik: Optional[float]
    iso_loglik: Optional[float]
    chem_logprior: Optional[float]
    ms2_loglik: Optional[float]
    log_posterior: Optional[float]
    manual: bool = False


@dataclass
class FormulaAssignment:
    """Ranked formula candidates for one MS source, plus the query provenance
    needed to interpret and reproduce the result."""
    source_uuid: 'EnsembleUUID'
    candidates: list[AssignedCandidate]

    uuid: 'AssignmentUUID' = field(default_factory=lambda: uuid_module.uuid4().int)
    source_kind: str = 'ensemble'
    chosen_idx: Optional[int] = None
    # Who made the current choice: 'auto' (find-mfs's top hit, picked by an
    # automatic run) or 'user'. Automatic runs never replace a 'user' choice.
    chosen_by: Optional[str] = None

    # --- Provenance (how this assignment was produced) ---
    precursor_mz: Optional[float] = None
    charge: Optional[int] = None
    adducts: Optional[list[str]] = None
    ms2_mode: Optional[str] = None

    # The search find-mfs actually ran: element set and bounds *after* any
    # halogen widening, and whether the Cl/Br pattern was detected (None when
    # detection was off).
    elements: Optional[str] = None
    max_counts: Optional[str] = None
    min_counts: Optional[str] = None
    halogen_detected: Optional[bool] = None

    # Snapshot of the FindMfsParams used (asdict), for reproducing the search.
    params: Optional[dict] = None

    created_at: float = field(default_factory=time.time)

    @property
    def top(self) -> Optional[AssignedCandidate]:
        """The best-ranked candidate, or None if the search was empty."""
        return self.candidates[0] if self.candidates else None

    @property
    def user_chosen(self) -> bool:
        return self.chosen is not None and self.chosen_by == 'user'

    @property
    def chosen(self) -> Optional[AssignedCandidate]:
        """The user-accepted candidate, or None if none accepted yet."""
        if self.chosen_idx is None or self.chosen_idx >= len(self.candidates):
            return None
        return self.candidates[self.chosen_idx]
