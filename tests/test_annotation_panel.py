"""
Offscreen smoke test of the Alignment Viewer's annotation tabs: a ranking
coming back fills the Search table, selecting a row mirrors it in the MS2
plot, hovering a predicted peak lights its fragment, and Accept adds it as the
chosen hypothesis; the Annotation tab takes a typed SMILES and mirrors it too.
"""
import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtWidgets  # noqa: E402

# The app must exist before the GUI modules load (and in that import order)
_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
import gui.controllers.main_controller  # noqa: E402,F401

from core.cli.rank_candidates import (  # noqa: E402
    DbCandidate, RankParams, RankResult, ScoredCandidate,
)
from core.utils.array_types import to_spec_arr  # noqa: E402


def _scored(db_id, smiles, explained, cosine):
    pred = to_spec_arr(np.array([50.0, 100.0]), np.array([0.5, 1.0]))
    n_atoms = 3
    return ScoredCandidate(
        DbCandidate('NPAtlas', db_id, db_id.lower(), smiles, 'C3H8O', 'Fungi', 'Aspergillus'),
        explained, cosine, pred, np.ones((2, n_atoms), bool), [(0, 1)], [0],
        ions=['C2H5+', 'C3H7O+'], h_shifts=[0, -2],
    )


def test_ranking_select_mirror_accept(data_registry, sample, ensemble, monkeypatch):
    from core.data_structs.alignment import AlignedAnalyte, EnsembleAlignment
    from gui.views.alignment_viewer import AlignmentViewer

    sample.injection.ensembles[ensemble.uuid] = ensemble
    analyte = AlignedAnalyte(
        ensemble_map={sample.uuid: ensemble.uuid},
        consensus_rt=ensemble.peak_rt, consensus_mz=ensemble.base_mz,
    )
    alignment = EnsembleAlignment(sample_uuids=(sample.uuid,), analytes=[analyte])
    data_registry.register_alignment(alignment)

    viewer = AlignmentViewer(data_registry)
    viewer.show()
    viewer.set_alignment(alignment)
    viewer.selection_model.set_selection(viewer.plot_item.analyte_targets([analyte.uuid]))
    panel = viewer.annotation_panel
    viewer.tabsAnnotation.setCurrentWidget(panel)
    # Independent of the local model / compound cache (re-focus checks both)
    monkeypatch.setattr('core.utils.glacier.model_available', lambda: True)
    panel.source_checks = [QtWidgets.QCheckBox('NPAtlas')]
    assert panel._analyte is analyte

    panel.ms2 = to_spec_arr(np.array([50.0, 100.0]), np.array([1.0, 0.4]))

    result = RankResult(
        [_scored('NPA2', 'CCCO', 2, 0.9), _scored('NPA1', 'CC(C)O', 1, 0.5)],
        n_considered=2, adduct='[M+H]+', adduct_fallback=False, params=RankParams(),
    )
    viewer.set_ranking((analyte.uuid, None), result)   # ranked with the analyte in focus
    assert panel.table.rowCount() == 2
    assert panel.table.item(0, 3).text() == "2/2 (0.90)"

    panel.table.selectRow(0)
    assert viewer._mirrored is panel.shown and viewer.plotMS2.is_mirror()
    viewer._on_ms2_mirror_hovered(('bottom', 1, 100.0, 1.0))
    assert not panel.structure.isHidden()
    assert panel.structure._caption == "C3H7O+\n-2H"
    viewer._on_ms2_mirror_left()
    assert panel.structure._caption == ""

    panel.accept_btn.click()
    annotation = data_registry.get_annotation(analyte.uuid)
    h = annotation.chosen
    assert (h.smiles, h.origin, h.db_id, annotation.label) == ('CCCO', 'NPAtlas', 'NPA2', 'npa2')
    assert h.scores['explained'] == "2/2"
    # Ranked with the analyte in focus: the representative's sample
    assert h.sample_uuid == sample.uuid
    # ...shown, selected and still mirrored in the Annotation tab
    editor, tree = viewer.annotation_editor, viewer.treeHypotheses
    assert viewer.tabsAnnotation.currentWidget() is viewer.tabAnnotation
    assert tree.topLevelItemCount() == 2 and tree.topLevelItem(0).text(1) == "★ npa2"
    assert editor.selected() == h and viewer.plotMS2.is_mirror()

    # Deselecting gives the bottom plot back to the focus
    tree.clearSelection()
    assert viewer._mirrored is None and not viewer.plotMS2.is_mirror()

    # A typed SMILES: a user hypothesis, scored by GLACIER for the mirror
    requests = []
    viewer.sigRankCandidatesRequested.connect(lambda key, kwargs: requests.append((key, kwargs)))
    add = tree.topLevelItem(1)
    add.setText(1, "CC(C)O")
    assert [h.origin for h in data_registry.get_annotation(analyte.uuid).hypotheses] == ['NPAtlas', 'user']
    assert editor.selected().smiles == "CC(C)O"
    (key, kwargs), = requests
    assert key[0] == 'hypothesis' and [c.smiles for c in kwargs['candidates']] == ["CC(C)O"]
    viewer.set_ranking(key, RankResult([_scored('', 'CC(C)O', 1, 0.5)], n_considered=2, adduct='[M+H]+',
                                       adduct_fallback=False, params=RankParams()))
    assert viewer.plotMS2.is_mirror()
    assert data_registry.get_annotation(analyte.uuid).hypotheses[1].scores['explained'] == "1/2"

    # Editing a database structure's SMILES makes a new user hypothesis (chosen, as the original was)
    monkeypatch.setattr(QtWidgets.QInputDialog, 'getText', lambda *a, **k: ("OCCC", True))
    editor._edit_smiles(0)
    annotation = data_registry.get_annotation(analyte.uuid)
    assert [h.smiles for h in annotation.hypotheses] == ['CCCO', 'CC(C)O', 'OCCC']
    assert annotation.chosen.origin == 'user' and annotation.label == 'npa2'
    editor._delete(2)
    assert data_registry.get_annotation(analyte.uuid).chosen_idx is None

    # Re-focusing elsewhere and back restores the cached ranking
    viewer.selection_model.set_selection([])
    assert panel.table.rowCount() == 0
    viewer.selection_model.set_selection(viewer.plot_item.analyte_targets([analyte.uuid]))
    assert panel.table.rowCount() == 2


def test_query_and_spectrum_follow_the_focused_member():
    """
    Members with different formulas and spectra: focusing one prefills its
    formula and adduct, ranks its own MS2, and keeps its own ranking.
    """
    from types import SimpleNamespace
    from core.data_structs.formula_assignment import AssignedCandidate, FormulaAssignment
    from gui.views.alignment_viewer.annotation_panel import AnnotationPanel

    def assignment(formula, adduct):
        return FormulaAssignment(source_uuid=0, candidates=[AssignedCandidate(
            formula_str=formula, adduct=adduct, error_ppm=1.0, error_da=0.0, rdbe=1.0, mass_loglik=-1.0,
            iso_loglik=None, chem_logprior=-1.0, ms2_loglik=None, log_posterior=-2.0,
        )], chosen_idx=0, chosen_by='auto')

    def member(uuid, mz):
        ms2 = to_spec_arr(np.array([mz]), np.array([1.0]))
        return SimpleNamespace(uuid=uuid, resolved_precursor_mz=181.0 + uuid,
                               composite_spectrum=SimpleNamespace(ms2=ms2),
                               formula_assignment=assignments[uuid])

    assignments = {1: assignment('C6H12O6', 'H'), 2: assignment('C7H16O5', 'Na')}
    a, b = member(1, 50.0), member(2, 60.0)
    ms2 = to_spec_arr(np.array([70.0]), np.array([1.0]))
    ctx = SimpleNamespace(
        representative=lambda analyte: SimpleNamespace(sample_uuid='s1', composite=SimpleNamespace(ms2=ms2)),
        members=lambda analyte: {'s1': a, 's2': b},
        formula=lambda e: assignments[e.uuid].candidates[0].formula_str,
        formula_agreement=lambda analyte: ('C6H12O6', 1, 2),
        focus_formula=lambda analyte, e=None: assignments[e.uuid].candidates[0].formula_str if e else 'C6H12O6',
    )
    analyte = SimpleNamespace(uuid=99, consensus_mz=181.07)
    panel = AnnotationPanel()

    panel.show_analyte(ctx, analyte)
    assert (panel.query_edit.text(), panel._adduct) == ('C6H12O6', '[M+H]+')
    assert panel.ms2['mz'].tolist() == [70.0] and panel._precursor_mz == 181.07   # representative / consensus
    panel.show_analyte(ctx, analyte, b)
    assert (panel.query_edit.text(), panel._adduct) == ('C7H16O5', '[M+Na]+')
    assert panel.ms2['mz'].tolist() == [60.0] and panel._precursor_mz == 183.0
    assert panel._sample_uuid == 's2'   # recorded on accept

    result = RankResult([_scored('NPA1', 'CCO', 1, 0.5)], n_considered=1, adduct='[M+Na]+',
                        adduct_fallback=False, params=RankParams())
    panel.set_ranking((99, 2), result)
    assert panel.table.rowCount() == 1
    panel.query_edit.setText('C8H8')   # typed by hand: a refresh of the same focus keeps it
    panel.show_analyte(ctx, analyte, b)
    assert panel.query_edit.text() == 'C8H8'
    panel.show_analyte(ctx, analyte)
    assert panel.query_edit.text() == 'C6H12O6'
    assert panel.table.rowCount() == 0   # b's ranking stays b's
    panel.show_analyte(ctx, analyte, b)
    assert panel.table.rowCount() == 1


def test_matches_formula():
    from gui.views.alignment_viewer.annotation_editor import matches_formula
    assert matches_formula('CCO', 'C2H6O') and matches_formula('OCC', 'H6C2O')
    assert not matches_formula('CCO', 'C2H4O')
    assert matches_formula('not smiles', 'C2H4O') and matches_formula('CCO', 'nonsense')
