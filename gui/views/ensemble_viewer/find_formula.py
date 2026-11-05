"""
Interactive formula finding. Includes a controller that manages
the ensemble viewer displays
"""
from gui.dialogues.formula_finder import FormulaFinderDialog
from gui.views.ensemble_viewer.tools import ToolType
from gui.views.ensemble_viewer.tool_controllers import SelectingToolController
from PyQt5 import QtCore

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from find_mfs import FormulaCandidate
    from gui.views.ensemble_viewer import EnsembleViewer


class FindFormulaController(SelectingToolController):
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
        self.formula_finder_menu.sigCompoundAssigned.connect(
            self.handle_compound_assigned
        )

    def on_enter_pressed(self):
        """
        Called when user presses Enter while selecting signals
        """
        self.handle_show_finder_menu()
        self.viewer.tool_manager.request_next_stage()

    def handle_show_finder_menu(self):
        self.formula_finder_menu.show()

        # Send selected_signals to formula finder
        self.formula_finder_menu.populate_table(
            [(x[0], x[1]) for x in self.selected_signals]
        )
        self.formula_finder_menu.set_source(*self._compound_source())

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

    def _compound_source(self) -> tuple:
        """
        `FormulaFinderDialog.set_source` args for the selection: the
        ensemble, the selected MS1 cofeatures and their scan (None for an MS2
        selection, which has no compound search).
        """
        ensemble = self.viewer.ensemble
        if ensemble is None or self.selected_ms_level != 1:
            return (None,)

        # TEMPORARY DDA path: ensemble MS2 production isn't wired up yet, so
        # hand find-mfs whatever MS2 spectrum the viewer currently shows
        # (mirrors the one-button AutoFindMF DDA stopgap). Retire once DDA
        # MS2 stitching lands and query_from_signals covers DDA natively.
        ms2_override = None
        if ensemble.is_dda:
            ms2_override = self.viewer.spectrum_manager.current_ms2
            if ms2_override is not None and len(ms2_override) == 0:
                ms2_override = None

        return (
            ensemble,
            [x[2] for x in self.selected_signals],
            self.viewer.ms1_scan_on_screen(),
            ms2_override,
        )

    def handle_compound_assigned(
            self,
            assignment,
            network,
    ):
        """
        Set the accepted compound assignment and the ion network it implies
        on its ensemble, through the DataRegistry (the EnsembleViewer's
        data_source); sigEnsembleAnnotated refreshes the viewer.
        """
        self.viewer.data_source.update_ensemble(
            assignment.source_uuid,
            formula_assignment=assignment,
            adduct_network=network,
        )
        self.handle_clear_selections()
