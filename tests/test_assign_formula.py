"""
Tests for core/formula: the MS2-based formula-assignment orchestration.

- DDA guard on the extractor (no data needed).
- The worker end-to-end on a synthetic thiamphenicol MS1 envelope: proves
  the halogen cap flows MzKit -> find-mfs and the FormulaAssignment records
  the search that actually ran.
- End-to-end from the real `ensemble` fixture (Cycloheximide; skips without the
  gitignored .mzk).
"""
from types import SimpleNamespace

import numpy as np
import pytest
from molmass import Formula

from core.utils.array_types import to_spec_arr
from core.formula import (
    FindMfsParams, FormulaQuery, query_from_ensemble, query_from_signals,
    assign_formula,
)
from core.data_structs.formula_assignment import FormulaAssignment

from find_mfs.ms2.net import bundled_npz_path

_NPZ = bundled_npz_path()
needs_mistnet = pytest.mark.skipif(
    not _NPZ.exists(), reason="needs bundled MistNet npz"
)


def _canon(s: str) -> str:
    return Formula(s).formula


def _has_formula(assignment: FormulaAssignment, formula_str: str) -> bool:
    target = _canon(formula_str)
    return any(_canon(c.formula_str) == target for c in assignment.candidates)


# --- DDA guard (no data) ----------------------------------------------------

def test_query_from_ensemble_rejects_dda():
    dda_like = SimpleNamespace(is_dda=True)
    with pytest.raises(NotImplementedError):
        query_from_ensemble(dda_like)


def test_query_from_signals_rejects_dda():
    dda_like = SimpleNamespace(is_dda=True)
    with pytest.raises(NotImplementedError):
        query_from_signals(dda_like, [(356.0, 1.0)])


# --- query_from_signals: the selection IS the precursor envelope -------------

def test_query_from_signals_uses_selection_as_envelope():
    """The selected MS1 signals become ms1_peaks; precursor_mz is their
    monoisotopic (lowest-m/z) peak — even when passed out of order."""
    stub = SimpleNamespace(
        is_dda=False,
        resolved_charge=1,
        uuid=777,
        get_ms2_spectra=lambda mode=None: [],   # no MS2 in this unit test
    )
    # Thiamphenicol [M+H]+ isotopologues, deliberately unsorted.
    signals = [
        (358.00965, 285583.0), (356.01261, 398411.0), (357.01537, 59720.0),
        (360.00702, 59256.0), (359.01269, 40221.0),
    ]

    q = query_from_signals(stub, signals, adducts=["H"])

    assert q.source_uuid == 777
    assert q.charge == 1
    assert q.ms2_spec is None
    assert q.precursor_mz == pytest.approx(356.01261)   # monoisotopic = lowest
    assert q.ms1_spec is not None and q.ms1_spec.shape[0] == 5
    # ms1_spec is sorted ascending by m/z
    assert list(q.ms1_spec["mz"]) == sorted(q.ms1_spec["mz"])


# --- worker end-to-end on a synthetic envelope ------------------------------

@needs_mistnet
def test_worker_widens_to_halogens_and_builds_assignment():
    """Thiamphenicol [M+H]+ (C12H15Cl2NO5S) Cl2 envelope: with a halogen-free
    max_counts, detection must widen the search via the cap, and the
    FormulaAssignment must record that it did."""
    ms1 = to_spec_arr(
        [356.01261, 357.01537, 358.00965, 359.01269, 360.00702],
        [398411.0, 59720.0, 285583.0, 40221.0, 59256.0],
    )
    query = FormulaQuery(
        precursor_mz=356.0126,
        source_uuid=12345,
        ms1_spec=ms1,
        ms2_spec=None,
        charge=1,
        adducts=["H"],
        params=FindMfsParams(
            max_counts="C*H*N*O*P0S2", halogen_cap="Cl4Br4", top_n=20,
        ),
    )

    out = assign_formula([query])

    assert len(out) == 1
    a = out[0]
    assert isinstance(a, FormulaAssignment)
    assert a.source_uuid == 12345
    assert a.chosen_idx is None
    assert a.top is not None
    assert len(a.candidates) <= 20                 # params.top_n respected
    # provenance: the search that actually ran
    assert a.halogen_detected is True
    assert a.elements == "CHNOSClBr"               # P0 dropped, Cl/Br added
    assert a.max_counts == "C*H*N*O*S2Cl4Br4"
    assert a.params["halogen_cap"] == "Cl4Br4"
    assert a.precursor_mz == pytest.approx(356.0126)
    # the cap widened the search, so the true Cl2 formula is reachable
    assert _has_formula(a, "C12H15Cl2NO5S")


# --- end-to-end from a real Ensemble ----------------------------------------

@needs_mistnet
def test_end_to_end_from_ensemble_manual_precursor(ensemble):
    """Real DIA Cycloheximide ensemble -> query -> assignment.

    Cycloheximide loses water in-source, so the base (tallest) MS1 ion is the
    dehydrated [M+H-H2O]+ (264.16), not [M+H]+. The manual precursor override
    pins the intact 282.17 -- the MVP's manual-steering path.
    """
    if ensemble.is_dda:
        with pytest.raises(NotImplementedError):
            query_from_ensemble(ensemble)
        pytest.skip("fixture ensemble is DDA; extractor correctly refuses")

    # Automated default picks the dominant in-source water-loss ion.
    auto = query_from_ensemble(ensemble)
    assert auto.precursor_mz == pytest.approx(264.1587, abs=0.02)

    # Manual override pins the true [M+H]+.
    query = query_from_ensemble(ensemble, precursor_mz=282.17024, adducts=["H"])
    assert query.ms1_spec is not None and query.ms1_spec.shape[0] > 0
    assert query.precursor_mz == pytest.approx(282.17024, abs=0.02)

    out = assign_formula([query])
    assert len(out) == 1
    a = out[0]
    assert a.source_uuid == ensemble.uuid
    assert a.top is not None
    # Cycloheximide is C15H23NO4
    assert _has_formula(a, "C15H23NO4")
