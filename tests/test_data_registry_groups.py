"""
Tests for the AnalyteGroup surface of DataRegistry: the analyte index,
orphan collection on alignment removal, and groups surviving a filter.
"""
from core.cli.filter_alignment import filter_alignment
from core.data_structs import AnalyteGroup, DataRegistry
from core.data_structs.alignment import AlignedAnalyte, EnsembleAlignment


def _alignment(analytes):
    return EnsembleAlignment(sample_uuids=(1,), analytes=list(analytes))


def _analytes(n):
    return [
        AlignedAnalyte(ensemble_map={1: i}, consensus_rt=10.0 * i, consensus_mz=100.0 + i)
        for i in range(n)
    ]


def test_groups_for_analytes():
    a = _analytes(4)
    reg = DataRegistry()
    g1 = AnalyteGroup([a[0].uuid, a[1].uuid])
    g2 = AnalyteGroup([a[1].uuid, a[2].uuid])
    reg.register_group(g1)
    reg.register_group(g2)

    assert reg.groups_for_analytes([a[0].uuid]) == [g1]
    assert reg.groups_for_analytes([a[1].uuid]) == [g1, g2]
    assert reg.groups_for_analytes([a[3].uuid]) == []

    # In-place edit + notify re-indexes
    g1.analyte_uuids.append(a[3].uuid)
    reg.notify_group_updated(g1)
    assert reg.groups_for_analytes([a[3].uuid]) == [g1]

    reg.remove_group(g1.uuid)
    assert reg.groups_for_analytes([a[0].uuid, a[3].uuid]) == []
    assert reg.get_all_groups() == [g2]


def test_remove_alignment_collects_orphan_groups():
    a = _analytes(3)
    first = _alignment(a[:2])
    second = _alignment(a[1:])   # shares a[1]
    reg = DataRegistry()
    reg.register_alignment(first)
    reg.register_alignment(second)

    survives = AnalyteGroup([a[0].uuid, a[1].uuid])   # a[1] lives on in `second`
    orphaned = AnalyteGroup([a[0].uuid])
    reg.register_group(survives)
    reg.register_group(orphaned)

    removed = []
    reg.sigGroupRemoved.connect(removed.append)
    reg.remove_alignment(first.uuid)

    assert reg.get_all_groups() == [survives]
    assert removed == [orphaned]

    reg.clear()
    assert reg.get_all_groups() == []


def test_filter_keeps_groups():
    """Mirrors `mzkit filter`: register the filtered child, then remove the
    parent. Children share the parent's analytes, so groups carry over."""
    a = _analytes(3)
    parent = _alignment(a)
    reg = DataRegistry()
    reg.register_alignment(parent)
    group = AnalyteGroup([a[2].uuid], title="late eluter")
    reg.register_group(group)

    child = filter_alignment(parent, "rt > 15", sample_names={1: "S1"}).alignment
    assert [x.uuid for x in child.analytes] == [a[2].uuid]

    reg.register_alignment(child)
    reg.remove_alignment(parent.uuid)
    assert reg.get_all_groups() == [group]
    assert reg.group_titles() == {a[2].uuid: "late eluter"}


def test_annotations_replace_and_collect_orphans():
    from core.data_structs import AnalyteAnnotation

    a = _analytes(2)
    alignment = _alignment(a)
    reg = DataRegistry()
    reg.register_alignment(alignment)

    added, updated, removed = [], [], []
    reg.sigAnnotationAdded.connect(added.append)
    reg.sigAnnotationUpdated.connect(updated.append)
    reg.sigAnnotationRemoved.connect(removed.append)

    first = AnalyteAnnotation(a[0].uuid, name="Psammaplin A")
    second = AnalyteAnnotation(a[0].uuid, name="Psammaplin A", smiles="CCO")
    reg.register_annotation(first)
    reg.register_annotation(second)   # same analyte: replaces
    assert (added, updated) == ([first], [second])
    assert reg.get_annotation(a[0].uuid) == second

    reg.remove_alignment(alignment.uuid)
    assert reg.get_all_annotations() == [] and removed == [second]
