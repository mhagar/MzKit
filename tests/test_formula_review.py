"""
Tests for reviewing / changing a formula assignment:
  - DataRegistry.set_chosen_candidate (+ chosen_by, update signal)
  - FormulaResultsWidget (sorting, current pick, what 'assign' reports)
  - FormulaFinderDialog.review (frozen envelope; assign / clear go through the registry)
  - to_formula_candidate (AssignedCandidate -> find-mfs FormulaCandidate)
"""
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtWidgets  # noqa: E402

from core.data_structs import DataRegistry  # noqa: E402
from core.data_structs.formula_assignment import (  # noqa: E402
    AssignedCandidate, FormulaAssignment,
)
from core.formula.assign_formula import to_formula_candidate  # noqa: E402

SOURCE = 4242


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _cand(formula, posterior, error_ppm=1.0, iso=None):
    return AssignedCandidate(
        formula_str=formula, adduct="H", error_ppm=error_ppm, error_da=0.001,
        rdbe=4.0, mass_loglik=-1.0, iso_loglik=iso, chem_logprior=-1.0,
        ms2_loglik=None, log_posterior=posterior,
    )


def _assignment(chosen_idx=0):
    return FormulaAssignment(
        source_uuid=SOURCE,
        candidates=[
            _cand("C8H10N4O2", -1.0, error_ppm=2.0, iso=-0.5),
            _cand("C6H12O6", -2.0, error_ppm=-0.5, iso=-0.1),
            _cand("C7H14O5", -3.0, error_ppm=1.0, iso=None),
        ],
        chosen_idx=chosen_idx,
        precursor_mz=195.0877, charge=1, elements="CHNO", halogen_detected=False,
    )


def _registry(*uuids) -> DataRegistry:
    """A registry holding one sample with a bare Ensemble per uuid (default SOURCE)."""
    from core.data_structs import Ensemble
    reg = DataRegistry()
    ensembles = {u: Ensemble([], [], uuid=u) for u in (uuids or (SOURCE,))}
    reg.register_sample(SimpleNamespace(
        uuid=1, name='s', fingerprint=None,
        injection=SimpleNamespace(ensembles=ensembles),
    ))
    return reg


def _assign(reg, assignment):
    reg.update_ensemble(assignment.source_uuid, formula_assignment=assignment)


def _assigned(reg, uuid=SOURCE):
    return reg.get_ensemble(uuid).formula_assignment


# --- registry ---------------------------------------------------------------

def test_set_chosen_candidate_records_who_chose(qapp):
    reg = _registry()
    a = _assignment()
    a.chosen_by = 'auto'
    _assign(reg, a)
    seen = []
    reg.sigEnsembleAnnotated.connect(seen.append)

    reg.set_chosen_candidate(SOURCE, 1)
    assert a.chosen.formula_str == "C6H12O6"
    assert a.chosen_by == 'user' and a.user_chosen
    assert reg.user_chosen_sources() == {SOURCE}
    assert reg.chosen_formulas() == {SOURCE: "C6H12O6"}
    assert seen == [SOURCE]

    reg.set_chosen_candidate(SOURCE, None)
    assert a.chosen is None and a.chosen_by is None and not a.user_chosen
    assert reg.user_chosen_sources() == set()
    assert reg.chosen_formulas() == {}


def test_set_chosen_candidate_rejects_bad_targets(qapp):
    reg = _registry()
    with pytest.raises(KeyError):
        reg.set_chosen_candidate(SOURCE, 0)
    _assign(reg, _assignment())
    with pytest.raises(IndexError):
        reg.set_chosen_candidate(SOURCE, 3)


def test_update_ensemble_emits_once_and_rejects_unknown_fields(qapp):
    reg = _registry()
    seen = []
    reg.sigEnsembleAnnotated.connect(seen.append)
    a = _assignment()
    reg.update_ensemble(SOURCE, formula_assignment=a, identity="caffeine")
    assert seen == [SOURCE]
    assert _assigned(reg) is a and reg.get_ensemble(SOURCE).identity == "caffeine"

    with pytest.raises(AttributeError):
        reg.update_ensemble(SOURCE, formula=None)
    with pytest.raises(KeyError):
        reg.update_ensemble(999, formula_assignment=None)
    assert seen == [SOURCE]


# --- results widget ---------------------------------------------------------

def test_widget_sorts_and_reports_candidate_index_not_row(qapp):
    from gui.widgets.formula_results_widget import FormulaResultsWidget

    w = FormulaResultsWidget()
    w.set_assignment(_assignment(chosen_idx=None))
    got = []
    w.sigAssignRequested.connect(got.append)

    w.comboSortBy.setCurrentText("Mass error")      # |ppm|: 0.5, 1.0, 2.0
    assert w._order == [1, 2, 0]
    w.tableResults.selectRow(0)
    w.btnAssignSelected.click()
    assert got == [1]                               # C6H12O6, not candidate 0

    w.comboSortBy.setCurrentText("Isotope envelope")  # missing term sorts last
    assert w._order == [1, 0, 2]


def test_widget_starts_on_current_pick(qapp):
    from gui.widgets.formula_results_widget import FormulaResultsWidget

    w = FormulaResultsWidget()
    w.set_assignment(_assignment(chosen_idx=2), allow_clear=True)
    assert w._selected_candidate_idx() == 2
    assert not w.btnAssignSelected.isEnabled()      # already the pick
    assert w.btnClearAssignment.isEnabled()
    assert "searched CHNO" in w.labelProvenance.text()
    assert "no Cl/Br pattern" in w.labelProvenance.text()


# --- review dialog ----------------------------------------------------------

def test_review_mode_assign_and_clear(qapp, data_registry, ensemble):
    import gui.controllers.main_controller  # noqa: F401  (app import order)
    from gui.dialogues.formula_finder import FormulaFinderDialog
    from core.data_structs.ensemble import AdductNetwork

    a = _assignment()
    a.source_uuid, a.envelope, a.scan_num = ensemble.uuid, [0], ensemble.base_scan_num
    net = AdductNetwork(M=1.0, base_envelope=[0], ions={0: '[M+Na]+'})
    data_registry.update_ensemble(ensemble.uuid, formula_assignment=a, adduct_network=net)

    dlg = FormulaFinderDialog.review(data_registry, ensemble)
    assert dlg.tableInput.rowCount() == 1 and not dlg.tableInput.isEnabled()
    assert dlg.btnFindIonMF.isHidden() and dlg.btnAddSignal.isHidden()
    assert dlg.comboIon.currentText() == '[M+Na]+'

    dlg.formulaResults.btnClearAssignment.click()
    assert ensemble.formula_assignment.chosen is None
    assert dlg.formulaResults.assignment.chosen_idx is None   # dialog refreshed

    dlg.formulaResults.tableResults.selectRow(1)             # posterior order
    dlg.formulaResults.btnAssignSelected.click()
    assert ensemble.formula_assignment.chosen_idx == 1
    assert ensemble.formula_assignment.chosen_by == 'user'
    assert ensemble.adduct_network is net

    # Closed + deleted: later registry updates must not reach it
    qapp.processEvents()
    data_registry.set_chosen_candidate(ensemble.uuid, 0)


# --- conversion -------------------------------------------------------------

def test_to_formula_candidate():
    fc = to_formula_candidate(_cand("C8H10N4O2", -1.0, error_ppm=2.0))
    assert str(fc.formula) == "C8H10N4O2"
    assert fc.adduct == "H" and fc.error_ppm == 2.0 and fc.rdbe == 4.0


# --- overwrite protection at registration + Alignment Viewer entry point -----

def test_completion_keeps_user_picks_made_mid_run(qapp):
    """A batch result must not replace an assignment the user picked while it
    ran -- unless the run was for that one ensemble (EV button)."""
    import gui.controllers.main_controller  # noqa: F401  (app import order)
    from gui.controllers.main_controller import MainController
    from core.cli.auto_find_mfs import BatchAnnotationResult

    reg = _registry()
    mine = _assignment()
    _assign(reg, mine)
    reg.set_chosen_candidate(SOURCE, 2)                  # user pick

    ctl = SimpleNamespace(
        data_registry=reg,
        main_view=SimpleNamespace(statusbar=QtWidgets.QStatusBar()),
    )
    fresh = _assignment()
    fresh.chosen_by = 'auto'
    result = BatchAnnotationResult(assignments=[fresh], n_ensembles=1)

    MainController._on_auto_find_mf_complete(ctl, result)
    assert _assigned(reg) is mine
    assert ctl.main_view.statusbar.currentMessage() == (
        "find-mfs: Annotated 0 of 1 ensemble; kept 1 user-chosen formula"
    )

    result = BatchAnnotationResult(assignments=[fresh], n_ensembles=1)
    MainController._on_auto_find_mf_complete(ctl, result, protect_user_choices=False)
    assert _assigned(reg) is fresh


def test_alignment_viewer_find_mfs_action(qapp):
    from gui.views.alignment_viewer import AlignmentViewer
    from gui.widgets.alignment_plot.layout import HoverTarget

    e1, e2, e3 = (SimpleNamespace(uuid=i) for i in (1, 2, 3))
    analyte_a, analyte_b = object(), object()
    members = {analyte_a: {'s1': e1, 's2': e2}, analyte_b: {'s1': e2, 's3': e3}}
    emitted = []
    selection_model = SimpleNamespace(selection=[])
    selection_model.contains = lambda t: t in selection_model.selection
    viewer = SimpleNamespace(
        _ctx=SimpleNamespace(members=lambda a: members[a]),
        selection_model=selection_model,
        sigAutoFindMfsRequested=SimpleNamespace(emit=emitted.append),
    )
    viewer._ensembles_for = lambda ts: AlignmentViewer._ensembles_for(viewer, ts)
    viewer._menu_targets = lambda t: AlignmentViewer._menu_targets(viewer, t)

    menus = []   # keep the menus (and so their actions) alive

    def menu_for(target):
        menu = QtWidgets.QMenu()
        menus.append(menu)
        AlignmentViewer._add_find_mfs_action(viewer, menu, target)
        return menu.actions()

    ta = HoverTarget(kind='analyte', analyte=analyte_a)
    tb = HoverTarget(kind='analyte', analyte=analyte_b)
    te = HoverTarget(kind='ensemble', ensemble=e3)

    (act,) = menu_for(ta)
    assert act.text() == "Auto find-mfs on this analyte (2 ensembles)"
    act.trigger()
    assert emitted[-1] == [e1, e2]

    (act,) = menu_for(te)
    assert act.text() == "Auto find-mfs on this ensemble"

    # Inside a multi-selection: all of it, with shared members deduplicated
    viewer.selection_model.selection = [ta, tb]
    (act,) = menu_for(tb)
    assert act.text() == "Auto find-mfs on 2 selected items (3 ensembles)"
    act.trigger()
    assert emitted[-1] == [e1, e2, e3]

    # Outside the selection: just the clicked item
    (act,) = menu_for(te)
    assert act.text() == "Auto find-mfs on this ensemble"

    # Samples aren't annotatable
    assert menu_for(HoverTarget(kind='sample', sample_uuid='s1')) == []


def test_sidebar_formula_batch_collects_and_clears(qapp, monkeypatch):
    from gui.controllers.main_controller import MainController

    registry = _registry(1, 2, 3)
    e1, e2, e3 = (registry.get_ensemble(i) for i in (1, 2, 3))
    samples = [
        SimpleNamespace(injection=SimpleNamespace(ensembles={1: e1, 2: e2})),
        SimpleNamespace(injection=None),   # a fingerprint-only sample
    ]
    alignment = SimpleNamespace(analytes=[
        SimpleNamespace(resolve_members=lambda get: {'s1': e2, 's2': e3}),
        SimpleNamespace(resolve_members=lambda get: {'s1': e3}),
    ])
    for e in (e2, e3):
        e.formula_assignment = FormulaAssignment(source_uuid=e.uuid, candidates=[])
    batches, messages = [], []
    controller = SimpleNamespace(
        sample_controller=SimpleNamespace(
            get_samples_by_index=lambda idx: samples,
            get_alignment_by_index=lambda idx: alignment,
        ),
        data_registry=registry,
        main_view=SimpleNamespace(statusbar=SimpleNamespace(showMessage=messages.append)),
        _handle_auto_find_mf_batch_request=batches.append,
    )

    def run(kind, action):
        MainController._handle_formula_batch_request(controller, kind, action, [None])

    run('Sample', 'find')
    run('Alignment', 'find')                 # shared members deduplicated
    assert batches == [[e1, e2], [e2, e3]]

    from PyQt5.QtWidgets import QMessageBox
    answers = [QMessageBox.No, QMessageBox.Yes]
    monkeypatch.setattr(QMessageBox, "question", lambda *a: answers.pop(0))
    run('Sample', 'clear')                   # declined: kept
    assert e2.formula_assignment is not None
    run('Sample', 'clear')                   # only the samples' ensembles go
    assert e2.formula_assignment is None
    assert e3.formula_assignment is not None
    assert messages[-1] == "Removed 1 formula assignment"
