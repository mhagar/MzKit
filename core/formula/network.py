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
