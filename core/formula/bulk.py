"""
"Apply its formula to all aligned members" (Alignment Viewer): the formula
finder's compound search, by hand on one ensemble, looped over the others.

Per member: find its envelope at the source envelope's m/z (within the
alignment's m/z tolerance), search it as the source's ion, and choose the
source's formula if it comes up (else the member's top hit, 'auto').

Qt-free; runs through ProcessController.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np

from core.formula.compound import search_compound
from core.formula.manual import canonical_formula
from core.formula.params import FindMfsParams

if TYPE_CHECKING:
    from core.data_structs import Ensemble
    from core.data_structs.ensemble import AdductNetwork
    from core.data_structs.formula_assignment import FormulaAssignment
    from core.utils.array_types import SpectrumArray

logger = logging.getLogger(__name__)


@dataclass
class BulkFormulaResult:
    n_members: int = 0
    # (EnsembleUUID, FormulaAssignment, AdductNetwork) to set
    updates: list[tuple[int, 'FormulaAssignment', Optional['AdductNetwork']]] = field(default_factory=list)
    no_feature: list[int] = field(default_factory=list)   # no envelope at the source's m/z
    disagreed: list[int] = field(default_factory=list)    # source formula not among the hits: top chosen
    n_cancelled: int = 0

    def summary(self) -> str:
        text = f"Formula applied to {len(self.updates)}/{self.n_members} member(s)"
        if self.disagreed:
            text += f"; {len(self.disagreed)} didn't find it (their top hit chosen)"
        if self.no_feature:
            text += f"; {len(self.no_feature)} skipped (no feature at that m/z)"
        if self.n_cancelled:
            text += f"; {self.n_cancelled} cancelled"
        return text


def match_envelope(
    src_mzs: list[float],
    spec: 'SpectrumArray',
    tol_da: float,
) -> Optional[list[int]]:
    """
    The idxs into `spec` (a cofeature-ordered spectrum) nearest each of
    `src_mzs` within `tol_da`, among peaks with signal. None if the first
    (monoisotopic) one has no match; other unmatched peaks are dropped.
    """
    live = np.flatnonzero(spec['intsy'] > 0)
    out = []
    for k, mz in enumerate(src_mzs):
        if live.size == 0:
            return None
        diffs = np.abs(spec['mz'][live] - mz)
        best = int(np.argmin(diffs))
        if diffs[best] > tol_da:
            if k == 0:
                return None
            continue
        out.append(int(live[best]))
    return out


def source_ion(source: 'Ensemble'):
    """The source's ion as a find-mfs IonType: its network's base ion."""
    from find_mfs.spectra.ions import ION_VOCAB

    network = source.adduct_network
    label = network.base_ion if network is not None else None
    return next((ion for ion in ION_VOCAB if ion.label == label), None)


def apply_formula_to_members(
    source: 'Ensemble',
    members: list['Ensemble'],
    tol_da: float,
    params: Optional[FindMfsParams] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> BulkFormulaResult:
    """
    Search each of `members` (the source itself is skipped) as the source's
    ion at the source's envelope, choosing the source's formula if found.

    :raises ValueError: the source has no searched envelope, chosen formula
        or known ion.
    """
    params = params if params is not None else FindMfsParams()
    assignment = source.formula_assignment
    chosen = assignment.chosen if assignment is not None else None
    if chosen is None or not assignment.envelope or assignment.scan_num is None:
        raise ValueError("The source ensemble has no chosen formula from a searched envelope")
    ion = source_ion(source)
    if ion is None:
        raise ValueError("The source ensemble's ion is unknown")

    src_spec = source.get_spectrum(ms_level=1, scan_num=assignment.scan_num)
    src_mzs = sorted(float(src_spec['mz'][i]) for i in assignment.envelope)
    formula = canonical_formula(chosen.formula_str)

    todo = [m for m in members if m.uuid != source.uuid]
    result = BulkFormulaResult(n_members=len(todo))
    for i, member in enumerate(todo):
        if cancel_event is not None and cancel_event.is_set():
            result.n_cancelled = len(todo) - i
            break
        if progress_callback is not None:
            progress_callback(100.0 * i / len(todo), f"Formula {i + 1}/{len(todo)}")

        scan_num = member.composite_ms1_scan_num()
        spec = member.get_spectrum(ms_level=1, scan_num=scan_num)
        envelope = match_envelope(src_mzs, spec, tol_da)
        if envelope is None:
            result.no_feature.append(member.uuid)
            continue
        signals = [(float(spec['mz'][j]), float(spec['intsy'][j])) for j in envelope]
        # DDA: the composite MS2, as the formula finder's review mode does
        kwargs = dict(ms2_spec=member.composite_spectrum.ms2, ms2_mode="selected_scan") if member.is_dda else {}
        try:
            found, network = search_compound(member, signals, envelope, scan_num, ion, params, **kwargs)
        except (ValueError, NotImplementedError) as exc:
            logger.warning("Skipping ensemble %s: %s", member.uuid, exc)
            result.no_feature.append(member.uuid)
            continue

        idx = next((k for k, c in enumerate(found.candidates)
                    if canonical_formula(c.formula_str) == formula), None)
        if idx is not None:
            found.chosen_idx, found.chosen_by = idx, 'user'
        else:
            result.disagreed.append(member.uuid)
            found.chosen_idx, found.chosen_by = (0, 'auto') if found.candidates else (None, None)
        result.updates.append((member.uuid, found, network))

    if progress_callback is not None:
        progress_callback(100.0, result.summary())
    return result
