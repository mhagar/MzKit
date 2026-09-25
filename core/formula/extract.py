"""
Adapter: Ensemble -> FormulaQuery.

This is the only Ensemble-aware code in core/formula.
 Everything after uses FormulaQuery (which is source-agnostic).

For now, DIA-only; DDA raises NotImplemented for now.
"""
from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING

import numpy as np

from core.formula.params import FindMfsParams
from core.formula.query import FormulaQuery
from core.utils.array_types import to_spec_arr

if TYPE_CHECKING:
    from core.data_structs.ensemble import Ensemble
    from core.utils.array_types import SpectrumArray

# Isotope window around the monoisotopic peak (Da) for the observed MS1 envelope.
# Lower bound sits just below M0 so the monoisotopic peak is the lowest-m/z peak
# in the crop — what find-mfs's iso term and envelope_is_halogen both assume.
_ISO_WINDOW_LO = 0.5
_ISO_WINDOW_HI = 5.0


def _ms1_envelope(
    ensemble: 'Ensemble',
    center_mz: float,
) -> Optional['SpectrumArray']:
    """
    Get observed isotope envelope around `center_mz`,
     from the apex MS1 scan.

    Since this uses cofeature ensemble, don't need to worry about
    -2H analogues and such. All signals are coeluting/corrleating.

    Assumes center_mz ~= M0 (true for small-molecule NP targets; breaks for very
    high MW where M+1/M+2 can be tallest).
    """
    if ensemble.injection is None or ensemble.injection.scan_array_ms1 is None:
        return None

    spec = ensemble.get_spectrum(ms_level=1, scan_num=ensemble.base_scan_num)
    mask = (
        (spec['mz'] >= center_mz - _ISO_WINDOW_LO)
        & (spec['mz'] <= center_mz + _ISO_WINDOW_HI)
        & (spec['intsy'] > 0)
    )
    env = spec[mask]
    return env if env.shape[0] > 0 else None


def _as_spectrum(
    signals: 'Union[SpectrumArray, list[tuple[float, float]]]',
) -> Optional['SpectrumArray']:
    """
    Coerce selected signals to a SpectrumArray sorted by m/z.

    Accepts either a structured SpectrumArray or a list of (mz, intensity) pairs.
    """
    if signals is None:
        return None

    dtype = getattr(signals, 'dtype', None)
    if dtype is not None and dtype.names:
        spec = signals
    else:
        if len(signals) == 0:
            return None
        mzs = [float(s[0]) for s in signals]
        intsys = [float(s[1]) for s in signals]
        spec = to_spec_arr(mzs, intsys)

    return spec[np.argsort(spec['mz'])]


def query_from_signals(
    ensemble: 'Ensemble',
    ms1_signals: 'Union[SpectrumArray, list[tuple[float, float]]]',
    *,
    charge: Optional[int] = None,
    adducts: Optional[list[str]] = None,
    params: Optional[FindMfsParams] = None,
    ms2_mode: str = 'tallest',
    ms2_spec: Optional['SpectrumArray'] = None,
) -> FormulaQuery:
    """
    Build a FormulaQuery from user-selected MS1 isotopologue signals.

    The user selection is the precursor envelope:
     `precursor_mz` is its monoisotopic (lowest-m/z) peak and
     `ms1_peaks` is the selection itself.

    MS2 normally comes from the ensemble (`get_ms2_spectra(mode=ms2_mode)`).
    Callers may instead pass an explicit `ms2_spec` — the TEMPORARY DDA path
    does this to feed find-mfs whatever MS2 spectrum the viewer has on screen,
    since DDA MS2 production isn't wired up yet.

    :param ms1_signals: selected MS1 peaks — a SpectrumArray or a list of
        (mz, intensity) pairs.
    :param params: find-mfs constraints/scoring; defaults to FindMfsParams().
    :param ms2_spec: explicit MS2 spectrum override. When given, it is used
        directly and the DDA guard is skipped.
    :raises NotImplementedError: for DDA ensembles when no `ms2_spec` override
        is supplied (ensemble MS2 production is DIA-only for now).
    :raises ValueError: if no MS1 signals are given.
    """
    if ensemble.is_dda and ms2_spec is None:
        raise NotImplementedError(
            "Compound formula assignment is not yet implemented for DDA "
            "ensembles; DIA / MS1-only only for now."
        )

    ms1_spec = _as_spectrum(ms1_signals)
    if ms1_spec is None or ms1_spec.shape[0] == 0:
        raise ValueError("query_from_signals requires at least one MS1 signal")

    precursor_mz = float(ms1_spec['mz'].min())

    if ms2_spec is None:
        ms2_list = ensemble.get_ms2_spectra(mode=ms2_mode)
        ms2_spec = ms2_list[0].spectrum if ms2_list else None

    return FormulaQuery(
        precursor_mz=precursor_mz,
        source_uuid=ensemble.uuid,
        ms1_spec=ms1_spec,
        ms2_spec=ms2_spec,
        charge=charge if charge is not None else ensemble.resolved_charge,
        adducts=adducts,
        params=params if params is not None else FindMfsParams(),
        ms2_mode=ms2_mode,
    )


def query_from_ensemble(
    ensemble: 'Ensemble',
    *,
    precursor_mz: Optional[float] = None,
    charge: Optional[int] = None,
    adducts: Optional[list[str]] = None,
    params: Optional[FindMfsParams] = None,
    ms2_mode: str = 'tallest',
) -> FormulaQuery:
    """
    Build a FormulaQuery from a (DIA) Ensemble

    Picks the MS2 spectrum via get_ms2_spectra(mode) and the cofeature-filtered
    MS1 envelope from the apex scan.

    `precursor_mz` / `charge` default to the ensemble's resolved values, but are
    overridable.

    :raises NotImplementedError: for DDA ensembles (not yet supported).
    """
    if ensemble.is_dda:
        raise NotImplementedError(
            "Formula assignment is not yet implemented for DDA ensembles; "
            "DIA / MS1-only only for now."
        )

    resolved_precursor = (
        precursor_mz if precursor_mz is not None
        else ensemble.resolved_precursor_mz
    )

    ms2_list = ensemble.get_ms2_spectra(mode=ms2_mode)
    ms2_spec = ms2_list[0].spectrum if ms2_list else None

    return FormulaQuery(
        precursor_mz=resolved_precursor,
        source_uuid=ensemble.uuid,
        ms1_spec=_ms1_envelope(ensemble, resolved_precursor),
        ms2_spec=ms2_spec,
        charge=charge if charge is not None else ensemble.resolved_charge,
        adducts=adducts,
        params=params if params is not None else FindMfsParams(),
        ms2_mode=ms2_mode,
    )
