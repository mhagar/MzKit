"""
Tests for AlignedAnalyte member resolution + representative spectrum.
"""
from types import SimpleNamespace

from core.data_structs.alignment import AlignedAnalyte


def _fake_ensemble(uuid: int, base_intsy: float):
    return SimpleNamespace(
        uuid=uuid,
        base_intsy=base_intsy,
        composite_spectrum=f"composite-{uuid}",
    )


def test_representative_spectrum_is_tallest_member():
    analyte = AlignedAnalyte(ensemble_map={1: 10, 2: 20, 3: 30})
    ensembles = {
        1: _fake_ensemble(10, 5e5),
        2: _fake_ensemble(20, 2e11),   # tallest, even if saturated
        3: _fake_ensemble(30, 1e6),
    }
    rep = analyte.representative_spectrum(ensembles)
    assert rep.sample_uuid == 2
    assert rep.ensemble_uuid == 20
    assert rep.composite == "composite-20"

    # Cached
    assert analyte.representative_spectrum({}) is rep


def test_representative_spectrum_ignores_non_members_and_empty():
    analyte = AlignedAnalyte(ensemble_map={1: 10})
    assert analyte.representative_spectrum({}) is None
    rep = analyte.representative_spectrum({
        1: _fake_ensemble(10, 1.0),
        9: _fake_ensemble(90, 1e9),    # not a member
    })
    assert rep.ensemble_uuid == 10


def test_resolve_members():
    ens = _fake_ensemble(10, 1.0)
    samples = {
        1: SimpleNamespace(injection=SimpleNamespace(ensembles={10: ens})),
        2: SimpleNamespace(injection=None),
    }
    analyte = AlignedAnalyte(ensemble_map={1: 10, 2: 20, 3: 30})
    assert analyte.resolve_members(samples.get) == {1: ens}
