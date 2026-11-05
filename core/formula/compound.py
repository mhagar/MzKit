"""
Compound formula search for a user-selected MS1 envelope of an ensemble: the
formula finder's 'Find Compound MF (MS2)'. Sets the envelope's ion too, so it
returns an AdductNetwork with the FormulaAssignment.

  - ion unknown: find-mfs's mass-difference network decides it, from the
    whole scan with the base envelope pinned to the selection.
  - ion given (an `IonType` of find_mfs.spectra.ions.ION_VOCAB): the same
    network, anchored on it: M follows from the ion, the other envelopes' ions
    from M (network `by='user'`). If find-mfs finds no envelope at the selection
    (e.g. below its noise floor), that ion alone is searched and the network is
    just "this envelope is <ion>".

Qt-free.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, TYPE_CHECKING

from molmass import Formula

from core.formula.assign_formula import _get_scorer, assign_formula
from core.formula.extract import query_from_signals
from core.formula.params import FindMfsParams
from core.data_structs.ensemble import AdductNetwork

if TYPE_CHECKING:
    from find_mfs.spectra.ions import IonType
    from core.data_structs import Ensemble
    from core.data_structs.formula_assignment import FormulaAssignment
    from core.utils.array_types import SpectrumArray

_H = Formula('H').monoisotopic_mass


def ion_query(
    mono_mz: float,
    ion: 'IonType',
) -> tuple[float, list[tuple[Optional[str], int]], float]:
    """
    What to hand find-mfs's `annotate_precursor` for an envelope at `mono_mz`
    read as `ion`: (query m/z, adducts, implied neutral mass M).

    find-mfs's finder only does z=1 m/z arithmetic, so a multimer or
    multiply-charged ion is searched as its M's [M+H]+, the same trick
    `annotate_analyte_dia` uses.
    """
    from find_mfs.spectra.ions import deconv_mass

    M = ion.neutral_mass(deconv_mass(mono_mz, ion.charge))
    if ion.n == 1 and ion.charge == 1:
        return mono_mz, [(ion.adduct_str, 1)], M
    return M + _H, [('H', 1)], M


def search_compound(
    ensemble: 'Ensemble',
    signals: list[tuple[float, float]],
    envelope: list[int],
    scan_num: int,
    ion: Optional['IonType'],
    params: Optional[FindMfsParams] = None,
    *,
    ms2_spec: Optional['SpectrumArray'] = None,
    ms2_mode: str = 'tallest',
) -> tuple['FormulaAssignment', Optional[AdductNetwork]]:
    """
    Rank compound formulae for the selected envelope (`signals` as (m/z,
    intensity), `envelope` their MS1 cofeature idxs, read from MS1 scan
    `scan_num`), with the ensemble's MS2.

    :param ion: None lets find-mfs decide the ion.
    :param ms2_spec: explicit MS2 (the DDA stopgap; see `query_from_signals`).
    :raises NotImplementedError: DDA without `ms2_spec`.
    :raises ValueError: no signals, or (ion unknown) no envelope at the selection.
    """
    params = params if params is not None else FindMfsParams()
    query = query_from_signals(
        ensemble, signals, params=params, ms2_spec=ms2_spec, ms2_mode=ms2_mode,
    )

    from find_mfs import annotate_analyte_dia
    from core.cli.auto_find_mfs import assignment_from, network_from

    try:
        res = annotate_analyte_dia(
            ms1_peaks=ensemble.get_spectrum(ms_level=1, scan_num=scan_num),
            precursor_mz=query.precursor_mz,
            ms2_peaks=query.ms2_spec,
            scorer=_get_scorer(None),
            base_ion=ion,
            **params.search_kwargs(),
        )
    except ValueError:
        if ion is None:
            raise
        res = None
    network = None if res is None else network_from(res, len(ensemble.ms1_cofeatures))
    # An unmappable anchored network falls back too, so the user's ion is kept
    if res is not None and (ion is None or network is not None):
        if ion is not None:
            network.by = 'user'
        return assignment_from(
            res, ensemble, params, scan_num, network, ms2_mode=ms2_mode,
        ), network

    query_mz, adducts, M = ion_query(query.precursor_mz, ion)
    query = replace(query, precursor_mz=query_mz, adducts=adducts, charge=ion.charge)
    assignment = assign_formula([query])[0]
    assignment.adducts = [ion.label]     # for display; `adducts` above is find-mfs's form
    assignment.envelope, assignment.scan_num, assignment.M = list(envelope), scan_num, M
    # The ion is labelled on the envelope's monoisotopic (lowest-m/z) peak
    mono = envelope[min(range(len(signals)), key=lambda k: signals[k][0])]
    network = AdductNetwork(
        M=M, base_envelope=list(envelope), ions={mono: ion.label}, by='user',
    )
    return assignment, network
