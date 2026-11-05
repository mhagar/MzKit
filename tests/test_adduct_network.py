"""set_adduct_network: a network whose M moves clears the formula searched with the old M."""
from types import SimpleNamespace

from core.data_structs.ensemble import AdductNetwork
from core.data_structs.formula_assignment import FormulaAssignment
from core.formula.network import set_adduct_network


class _Source:
    def __init__(self, ensemble):
        self.ensemble, self.calls = ensemble, []

    def update_ensemble(self, uuid, **changes):
        self.calls.append(changes)
        for k, v in changes.items():
            setattr(self.ensemble, k, v)


def _setup(M):
    ens = SimpleNamespace(
        uuid=1, adduct_network=None,
        formula_assignment=FormulaAssignment(source_uuid=1, candidates=[], M=M),
    )
    return ens, _Source(ens)


def _net(M):
    return AdductNetwork(M=M, base_envelope=[0], ions={0: '[M+H]+'})


def test_same_M_keeps_formula():
    ens, src = _setup(180.0634)
    assert not set_adduct_network(src, ens, _net(180.0634 * (1 + 2e-6)))
    assert ens.formula_assignment is not None
    assert src.calls == [{'adduct_network': ens.adduct_network}]  # one update


def test_moved_M_clears_formula_in_one_update():
    ens, src = _setup(180.0634)
    net = _net(180.0634 - 21.98)          # [M+H]+ re-read as [M+Na]+
    assert set_adduct_network(src, ens, net)
    assert ens.formula_assignment is None and ens.adduct_network is net
    assert len(src.calls) == 1


def test_unknown_M_counts_as_stale():
    ens, src = _setup(None)               # e.g. a typed-in formula
    assert set_adduct_network(src, ens, _net(180.0634))


def test_no_formula_nothing_to_clear():
    ens, src = _setup(1.0)
    ens.formula_assignment = None
    assert not set_adduct_network(src, ens, _net(180.0634))
