"""
Tests for reviewing / changing a formula assignment:
  - DataRegistry.set_chosen_candidate (+ chosen_by, update signal)
  - FormulaResultsWidget (sorting, current pick, what 'assign' reports)
  - FormulaAssignmentDialog (assign / clear go through the registry)
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


class _Registry(DataRegistry):
    pass


# --- registry ---------------------------------------------------------------

def test_set_chosen_candidate_records_who_chose(qapp):
    reg = _Registry()
    a = _assignment()
    a.chosen_by = 'auto'
    reg.register_assignment(a)
    seen = []
    reg.sigAssignmentUpdated.connect(seen.append)

    reg.set_chosen_candidate(SOURCE, 1)
    assert a.chosen.formula_str == "C6H12O6"
    assert a.chosen_by == 'user' and a.user_chosen
    assert reg.user_chosen_sources() == {SOURCE}
    assert reg.chosen_formulas() == {SOURCE: "C6H12O6"}
    assert seen == [a]

    reg.set_chosen_candidate(SOURCE, None)
    assert a.chosen is None and a.chosen_by is None and not a.user_chosen
    assert reg.user_chosen_sources() == set()
    assert reg.chosen_formulas() == {}


def test_set_chosen_candidate_rejects_bad_targets(qapp):
    reg = _Registry()
    with pytest.raises(KeyError):
        reg.set_chosen_candidate(SOURCE, 0)
    reg.register_assignment(_assignment())
    with pytest.raises(IndexError):
        reg.set_chosen_candidate(SOURCE, 3)


def test_update_callback_subscription(qapp):
    reg = _Registry()
    reg.register_assignment(_assignment())
    updated = []
    reg.subscribe_to_changes(
        addition_callback=lambda a: None, removal_callback=lambda a: None,
        update_callback=updated.append, change_type='Assignment',
    )
    reg.set_chosen_candidate(SOURCE, 2)
    assert len(updated) == 1


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

def test_review_dialog_assign_and_clear(qapp):
    from gui.dialogues.formula_assignment_dialog import FormulaAssignmentDialog

    reg = _Registry()
    reg.register_assignment(_assignment())

    dlg = FormulaAssignmentDialog(reg, SOURCE)
    dlg.results.btnClearAssignment.click()
    assert reg.get_assignment_for_source(SOURCE).chosen is None
    assert dlg.results.assignment.chosen_idx is None   # dialog refreshed

    dlg.results.tableResults.selectRow(1)             # posterior order
    dlg.results.btnAssignSelected.click()
    assert reg.get_assignment_for_source(SOURCE).chosen_idx == 1

    # Closed + deleted: later registry updates must not reach it
    dlg.close()
    qapp.processEvents()
    reg.set_chosen_candidate(SOURCE, 0)


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

    reg = _Registry()
    mine = _assignment()
    reg.register_assignment(mine)
    reg.set_chosen_candidate(SOURCE, 2)                  # user pick

    ctl = SimpleNamespace(
        data_registry=reg,
        main_view=SimpleNamespace(statusbar=QtWidgets.QStatusBar()),
    )
    fresh = _assignment()
    fresh.chosen_by = 'auto'
    result = BatchAnnotationResult(assignments=[fresh], n_ensembles=1)

    MainController._on_auto_find_mf_complete(ctl, result)
    assert reg.get_assignment_for_source(SOURCE) is mine
    assert ctl.main_view.statusbar.currentMessage() == (
        "find-mfs: Annotated 0 of 1 ensemble; kept 1 user-chosen formula"
    )

    result = BatchAnnotationResult(assignments=[fresh], n_ensembles=1)
    MainController._on_auto_find_mf_complete(ctl, result, protect_user_choices=False)
    assert reg.get_assignment_for_source(SOURCE) is fresh


def test_alignment_viewer_find_mfs_action(qapp):
    from gui.views.alignment_viewer import AlignmentViewer
    from gui.widgets.alignment_plot.layout import HoverTarget

    e1, e2, e3 = (SimpleNamespace(uuid=i) for i in (1, 2, 3))
    analyte_a, analyte_b = object(), object()
    members = {analyte_a: {'s1': e1, 's2': e2}, analyte_b: {'s1': e2, 's3': e3}}
    emitted = []
    viewer = SimpleNamespace(
        _ctx=SimpleNamespace(members=lambda a: members[a]),
        _selection=[],
        sigAutoFindMfsRequested=SimpleNamespace(emit=emitted.append),
    )
    viewer._ensembles_for = lambda ts: AlignmentViewer._ensembles_for(viewer, ts)

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
    viewer._selection = [ta, tb]
    (act,) = menu_for(tb)
    assert act.text() == "Auto find-mfs on 2 selected items (3 ensembles)"
    act.trigger()
    assert emitted[-1] == [e1, e2, e3]

    # Outside the selection: just the clicked item
    (act,) = menu_for(te)
    assert act.text() == "Auto find-mfs on this ensemble"

    # Samples aren't annotatable
    assert menu_for(HoverTarget(kind='sample', sample_uuid='s1')) == []
