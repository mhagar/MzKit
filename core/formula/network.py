"""
Setting an ensemble's AdductNetwork (which ion each MS1 envelope is).

The network is upstream of the formula: a FormulaAssignment records the
neutral mass `M` it searched with, so a network whose M moves makes the
formula stale, and it is cleared in the same update.

Qt-free.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.data_structs import Ensemble
    from core.data_structs.ensemble import AdductNetwork
    from core.interfaces.data_sources import EnsembleDataSource
    from core.utils.array_types import SpectrumArray


def set_adduct_network(
    data_source: 'EnsembleDataSource',
    ensemble: 'Ensemble',
    network: Optional['AdductNetwork'],
    tol_ppm: float = 5.0,
) -> bool:
    """
    Set the ensemble's network; clear its formula assignment if that was
    searched with a different M (beyond `tol_ppm`), or with no recorded M.

    :return: whether the formula assignment was cleared.
    """
    assignment = ensemble.formula_assignment
    stale = assignment is not None and (
        network is None
        or assignment.M is None
        or abs(assignment.M - network.M) > network.M * tol_ppm * 1e-6
    )
    changes = {'adduct_network': network}
    if stale:
        changes['formula_assignment'] = None
    data_source.update_ensemble(ensemble.uuid, **changes)
    return stale


# Representative-MS2 preference tiers (lower fragments better): see `ion_tier`
TIER_MONOMER, TIER_MULTICHARGED, TIER_MULTIMER, TIER_UNLABELLED, TIER_METAL = range(5)
_METALS = {'Li', 'Na', 'K', 'Rb', 'Cs', 'Mg', 'Ca', 'Fe', 'Cu', 'Zn', 'Ag'}


def ion_tier(label: Optional[str]) -> int:
    """
    How well an ion (find-mfs ION_VOCAB label, e.g. '[M+Na]+') is expected to
    fragment, best first: non-metal monomers < non-metal multiply charged <
    non-metal multimers < unlabelled < metal adducts (poor fragments).
    """
    from find_mfs.spectra.ions import ION_VOCAB
    from molmass import Formula

    ion = next((i for i in ION_VOCAB if i.label == label), None)
    if ion is None:
        return TIER_UNLABELLED
    if ion.adduct_str and _METALS & set(Formula(ion.adduct_str.lstrip('-')).composition()):
        return TIER_METAL
    if ion.n > 1:
        return TIER_MULTIMER
    return TIER_MULTICHARGED if ion.charge > 1 else TIER_MONOMER


def envelope_ions(
    ms1: 'SpectrumArray',
    network: Optional['AdductNetwork'] = None,
) -> dict[int, Optional[str]]:
    """
    Peak idx of a cofeature-ordered MS1 spectrum (e.g. the composite MS1) ->
    ion label of the isotope envelope it belongs to (None: unlabelled / noise).

    Envelopes come from find-mfs's grouping. Their ions come from `network`
    (keyed by envelope mono idx, so the user's anchors count) when given,
    else from find-mfs's mass-difference network around the tallest envelope.
    """
    import numpy as np
    from find_mfs import NoiseThreshold, group_signals
    from find_mfs.spectra.network import solve_for_base

    if ms1 is None or len(ms1) == 0 or not np.any(ms1['intsy'] > 0):
        return {}
    # Same noise floor as find-mfs's annotate_analyte_dia
    grouped = group_signals(ms1, noise=NoiseThreshold(min_rel=0.05), detect_halogens=False)
    assigned = np.flatnonzero(grouped.group_labels >= 0)
    if network is not None:
        labels = [network.ions.get(int(m)) for m in grouped.mono_idx]
    elif len(assigned):
        tallest = assigned[np.argmax(grouped.spec_arr['intsy'][assigned])]
        solve_for_base(grouped, int(grouped.group_labels[tallest]))
        labels = list(grouped.adduct_label)
    else:
        labels = []
    return {int(i): labels[grouped.group_labels[i]] for i in assigned}
