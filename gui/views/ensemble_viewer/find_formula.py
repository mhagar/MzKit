"""
Interactive formula finding. Includes a controller that manages
the ensemble viewer displays
"""
from gui.dialogues.formula_finder import FormulaFinderDialog
from gui.views.ensemble_viewer.tools import (
    ToolType, Mode, ToolStage
)
from gui.views.ensemble_viewer.tool_controllers import BaseToolController
from core.formula import query_from_signals, assign_formula
from core.formula.assign_formula import to_formula_candidate

from PyQt5 import QtCore, QtWidgets

from typing import Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from gui.views.ensemble_viewer import EnsembleViewer


class FindFormulaController(BaseToolController):
    """
    Controller for the 'Find Formula' tool
    """
    sigFormulaAssigned = QtCore.pyqtSignal(
        object,      # find_mfs.FormulaCandidate
        int,                # ms_level
        list,               # cofeature_idxs
    )

    def __init__(
        self,
        ensemble_viewer: 'EnsembleViewer'
    ):
        super().__init__(
            viewer=ensemble_viewer,
            tool_type=ToolType.FINDFORMULA,
        )
        self.selected_ms_level: Literal[None, 1, 2] = None
        self.selected_signals: list[tuple[float, float, int]] = []
        self.formula_finder_menu: FormulaFinderDialog = FormulaFinderDialog(
            parent=ensemble_viewer,
            config=ensemble_viewer.config,
            modal=True,
        )
        self._connect_signals()

    def _connect_signals(self):
        self.formula_finder_menu.sigFormulaAssigned.connect(
            self.handle_formula_assigned
        )
        self.formula_finder_menu.sigCompoundSearchRequested.connect(
            self.handle_compound_search
        )
        self.formula_finder_menu.sigCompoundAssigned.connect(
            self.handle_compound_assigned
        )

    def on_activated(self):
        """
        Called when Find Formula tool is activated
        """
        self.handle_clear_selections()
        # Request next stage (IDLE -> SELECTING)
        self.viewer.tool_manager.request_next_stage()

    def on_enter_pressed(self):
        """
        Called when user presses Enter while selecting signals
        """
        self.handle_show_finder_menu()
        self.viewer.tool_manager.request_next_stage()

    def on_cancelled(self):
        """
        Called when tool is cancelled
        """
        self.handle_clear_selections()

    def handle_ms_signal_clicked(
        self,
        data: tuple[float, float, int],  # mz, intsy, spec_idx
        ms_level: Literal[1, 2],
    ):
        if self.selected_ms_level != ms_level:
            if not self.selected_ms_level:
                self.selected_ms_level = ms_level
            else:
                self.handle_clear_selections()

        if self.viewer.tool_manager.active_stage == ToolStage.SELECTING:
            mz, intsy, spec_idx = data

            if data in self.selected_signals:
                # Remove signal from selection
                self.selected_signals.remove(data)
                self.sigSignalSelected.emit(
                    mz,
                    intsy,
                    spec_idx,
                    self.selected_ms_level,
                    False,
                )
            else:
                # Add signal to selection
                self.selected_signals.append(data)
                self.sigSignalSelected.emit(
                    mz,
                    intsy,
                    spec_idx,
                    self.selected_ms_level,
                    True,
                )

    def handle_clear_selections(self):
        self.selected_signals.clear()
        self.selected_ms_level = None
        self.sigSelectionCleared.emit()

    def handle_show_finder_menu(self):
        self.formula_finder_menu.show()

        # Send selected_signals to formula finder
        self.formula_finder_menu.populate_table(
            [(x[0], x[1]) for x in self.selected_signals]
        )

        self.formula_finder_menu.on_search_execute()

    def handle_formula_assigned(
        self,
        formula: 'FormulaCandidate'
    ):
        """
        Called when user selects a formula from finder dialogue.

        Passes the signal outwards, and reverts to IDLE stage.
        """
        self.sigFormulaAssigned.emit(
            formula,  # FormulaCandidate
            self.selected_ms_level,  # int
            [x[2] for x in self.selected_signals],  # feature coidxs (ints)
        )

        self.handle_clear_selections()

    def handle_compound_search(self, params: dict):
        """
        Run the MS2 compound assignment.

        The dialog forwards params + the selected MS1 envelope;
        here we add the ensemble + MS2 and run find-mfs

        Synchronous for now because EnsembleViewer doesn't have a ProcessController handle;
         # TODO: move to background task runner
        """
        dialog = self.formula_finder_menu
        ensemble = self.viewer.ensemble

        if ensemble is None:
            return

        if self.selected_ms_level == 2:
            dialog.statusbar.showMessage(
                "Compound assignment needs the MS1 isotopologue group, not MS2."
            )

            return

        ms1_signals = params.get("ms1_signals")
        if not ms1_signals:
            dialog.statusbar.showMessage("No MS1 signals selected")
            return

        query_kwargs = dict(
            adducts=params.get("adducts"),
            params=params["params"],
        )

        # TEMPORARY DDA path: ensemble MS2 production isn't wired up yet, so
        # hand find-mfs whatever MS2 spectrum the viewer currently shows
        # (mirrors the one-button AutoFindMF DDA stopgap). Retire once DDA
        # MS2 stitching lands and query_from_signals covers DDA natively.
        if ensemble.is_dda:
            ms2_spec = self.viewer.spectrum_manager.current_ms2
            if ms2_spec is not None and len(ms2_spec) == 0:
                ms2_spec = None
            query_kwargs["ms2_spec"] = ms2_spec
            query_kwargs["ms2_mode"] = "selected_scan"

        try:
            query = query_from_signals(ensemble, ms1_signals, **query_kwargs)
        except NotImplementedError:
            dialog.statusbar.showMessage(
                "Compound formula assignment is not yet supported for DDA."
            )
            return

        dialog.statusbar.showMessage(
            "Assigning compound formula (MS2)…"
        )
        QtWidgets.QApplication.setOverrideCursor(
            QtCore.Qt.WaitCursor
        )
        try:
            assignment = assign_formula([query])[0]
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        dialog.populate_compound_results(assignment)

    def handle_compound_assigned(
            self,
            assignment
    ):
        """
        Register the accepted compound assignment in the DataRegistry
        (the EnsembleViewer's data_source)
        """
        registry = self.viewer.data_source
        registry.register_assignment(assignment)

        chosen = assignment.chosen
        if self.viewer.ensemble is not None and chosen is not None:
            # Debug aid: also drop an ion annotation on the envelope that fed the
            # assignment, reusing the ion-formula draw path (add_ion_annot + the
            # on-plot predicted envelope). selected_signals still hold the picked
            # cofeatures at this point (cleared just below).
            if self.selected_ms_level and self.selected_signals:
                self.sigFormulaAssigned.emit(
                    to_formula_candidate(chosen),
                    self.selected_ms_level,
                    [x[2] for x in self.selected_signals],
                )

        # Show the assignment as the MS1 title strip.
        if hasattr(self.viewer, "refresh_assignment_display"):
            self.viewer.refresh_assignment_display()

        self.handle_clear_selections()




