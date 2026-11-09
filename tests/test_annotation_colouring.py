from types import SimpleNamespace as NS

from core.data_structs.ensemble.annotations import EnsembleAnnotation
from gui.widgets.alignment_plot.layout import is_annotated


def ens(fa=None, annotation=None, identity=None):
    return NS(formula_assignment=fa, annotation=annotation, identity=identity)


def test_is_annotated():
    assert not is_annotated(ens())
    assert not is_annotated(ens(fa=NS(chosen=None)))         # unaccepted candidates only
    assert not is_annotated(ens(annotation=EnsembleAnnotation()))
    assert is_annotated(ens(fa=NS(chosen=object())))
    assert is_annotated(ens(annotation=EnsembleAnnotation(name="Psammaplin A")))
    assert is_annotated(ens(identity="caffeine"))


def test_viewer_colour_modes(data_registry, sample, ensemble):
    from PyQt5.QtGui import QColor
    from core.data_structs import EnsembleGroup
    from core.data_structs.alignment import AlignedAnalyte, EnsembleAlignment
    from gui.views.alignment_viewer import AlignmentViewer
    from gui.widgets.alignment_plot.layout import ANNOTATED_COLOR, SINGLETON_COLOR, group_color

    sample.injection.ensembles[ensemble.uuid] = ensemble
    analyte = AlignedAnalyte(
        ensemble_map={sample.uuid: ensemble.uuid},
        consensus_rt=ensemble.peak_rt, consensus_mz=ensemble.base_mz,
    )
    alignment = EnsembleAlignment(sample_uuids=(sample.uuid,), analytes=[analyte])
    data_registry.register_alignment(alignment)
    viewer = AlignmentViewer(data_registry)
    viewer.set_alignment(alignment)
    color = lambda: viewer.plot_item._strips.colors[0]
    combo = viewer.comboColorMode

    combo.setCurrentIndex(combo.findData('annotations'))
    assert color() == QColor(SINGLETON_COLOR)
    ensemble.identity = "caffeine"
    viewer._refresh_inspector()     # as on sigEnsembleAnnotated
    assert color() == QColor(ANNOTATED_COLOR)

    combo.setCurrentIndex(combo.findData('groups'))
    assert color() == QColor(SINGLETON_COLOR)
    data_registry.register_group(EnsembleGroup([ensemble.uuid], title="G"))
    assert color() == group_color(0)
    assert viewer.plotGroupTrack._rows[0].color == group_color(0)   # track matches the map
    combo.setCurrentIndex(combo.findData('alignments'))
    assert viewer.plotGroupTrack._rows[0].color != group_color(0)
    viewer.close()
