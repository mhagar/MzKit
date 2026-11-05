"""
Tests for the EnsembleGroup surface of DataRegistry: the ensemble index,
groups outliving alignments, pruning with their ensembles, and per-analyte
titles for exports.
"""
from types import SimpleNamespace

from core.cli.filter_alignment import filter_alignment
from core.data_structs import DataRegistry, Ensemble, EnsembleGroup
from core.data_structs.alignment import AlignedAnalyte, EnsembleAlignment


def _alignment(analytes):
    return EnsembleAlignment(sample_uuids=(1,), analytes=list(analytes))


def _analytes(n):
    """Analyte i's one member is ensemble i of sample 1."""
    return [
        AlignedAnalyte(ensemble_map={1: i}, consensus_rt=10.0 * i, consensus_mz=100.0 + i)
        for i in range(n)
    ]


def _registry(n_ensembles):
    """A registry holding sample 1 with bare ensembles 0..n-1."""
    injection = SimpleNamespace(ensembles={i: Ensemble([], [], uuid=i) for i in range(n_ensembles)})
    injection.remove_ensemble = injection.ensembles.pop
    reg = DataRegistry()
    reg.register_sample(SimpleNamespace(uuid=1, name='s', fingerprint=None, injection=injection))
    return reg


def test_groups_for_ensembles():
    reg = DataRegistry()
    g1 = EnsembleGroup([0, 1])
    g2 = EnsembleGroup([1, 2])
    reg.register_group(g1)
    reg.register_group(g2)

    assert reg.groups_for_ensembles([0]) == [g1]
    assert reg.groups_for_ensembles([1]) == [g1, g2]
    assert reg.groups_for_ensembles([3]) == []

    # In-place edit + notify re-indexes
    g1.ensemble_uuids.append(3)
    reg.notify_group_updated(g1)
    assert reg.groups_for_ensembles([3]) == [g1]

    reg.remove_group(g1.uuid)
    assert reg.groups_for_ensembles([0, 3]) == []
    assert reg.get_all_groups() == [g2]


def test_groups_outlive_alignments_and_go_with_their_ensembles():
    reg = _registry(3)
    alignment = _alignment(_analytes(3))
    reg.register_alignment(alignment)
    pair, single = EnsembleGroup([0, 1]), EnsembleGroup([2])
    reg.register_group(pair)
    reg.register_group(single)

    reg.remove_alignment(alignment.uuid)   # alignments are cheap; groups aren't
    assert reg.get_all_groups() == [pair, single]

    updated, removed = [], []
    reg.sigGroupUpdated.connect(updated.append)
    reg.sigGroupRemoved.connect(removed.append)
    reg.remove_ensembles([1, 2])
    assert reg.get_ensemble(1) is None and reg.get_ensemble(0) is not None
    assert pair.ensemble_uuids == [0] and updated == [pair]
    assert removed == [single]          # left empty

    reg.remove_sample(1)                # takes ensemble 0, and so `pair`
    assert reg.get_all_groups() == []


def test_group_titles_are_per_analyte_over_its_members():
    analytes = [
        AlignedAnalyte(ensemble_map={1: 10, 2: 20}),
        AlignedAnalyte(ensemble_map={1: 11}),
        AlignedAnalyte(ensemble_map={1: 12}),
    ]
    reg = DataRegistry()
    reg.register_group(EnsembleGroup([20, 11], title="Sugars", created_at=1.0))
    reg.register_group(EnsembleGroup([10], title="Glucose", created_at=2.0))
    reg.register_group(EnsembleGroup([10], title="Sugars", created_at=3.0))   # same title: once
    reg.register_group(EnsembleGroup([12], title="", created_at=4.0))         # untitled: skipped
    titles = reg.group_titles(_alignment(analytes))
    assert titles == {analytes[0].uuid: "Sugars;Glucose", analytes[1].uuid: "Sugars"}


def test_filter_keeps_groups():
    """Mirrors `mzkit filter`: register the filtered child, then remove the
    parent. Groups are on ensembles, so they carry over."""
    a = _analytes(3)
    parent = _alignment(a)
    reg = DataRegistry()
    reg.register_alignment(parent)
    group = EnsembleGroup([2], title="late eluter")
    reg.register_group(group)

    child = filter_alignment(parent, "rt > 15", sample_names={1: "S1"}).alignment
    assert [x.uuid for x in child.analytes] == [a[2].uuid]

    reg.register_alignment(child)
    reg.remove_alignment(parent.uuid)
    assert reg.get_all_groups() == [group]
    assert reg.group_titles(child) == {a[2].uuid: "late eluter"}


def test_ensemble_annotation_outlives_its_alignment():
    from types import SimpleNamespace
    from core.data_structs import Ensemble
    from core.data_structs.ensemble import EnsembleAnnotation

    ens = Ensemble([], [], uuid=0)
    reg = DataRegistry()
    reg.register_sample(SimpleNamespace(
        uuid=1, name='s', fingerprint=None,
        injection=SimpleNamespace(ensembles={ens.uuid: ens}),
    ))
    alignment = _alignment(_analytes(1))   # its one analyte is ensemble 0
    reg.register_alignment(alignment)

    seen = []
    reg.sigEnsembleAnnotated.connect(seen.append)
    annotation = EnsembleAnnotation(name="Psammaplin A", notes="sponge")
    reg.update_ensemble(ens.uuid, annotation=annotation)
    assert seen == [ens.uuid]

    reg.remove_alignment(alignment.uuid)   # alignments are cheap; annotations aren't
    assert reg.get_ensemble(ens.uuid).annotation == annotation
