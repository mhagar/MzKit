"""
Interactive formula finding for neutral losses. Includes a controller
that manages ensemble viewer displays
"""
from gui.dialogues.formula_finder import FormulaFinderDialog
from gui.views.ensemble_viewer.tools import (
    ToolType, Mode, ToolStage
)
from gui.views.ensemble_viewer.tool_controllers import BaseToolController

from PyQt5 import QtCore, QtWidgets
from find_mfs import FormulaCandidate

from typing import Literal, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from gui.views.ensemble_viewer import EnsembleViewer


class MeasureLossController(BaseToolController):
    """
    Controller for the 'Measure Loss' tool
    """
    sigFormulaAssigned = QtCore.pyqtSignal(
        object,      # find_mfs.FormulaCandidate
        int,                # ms_level
        int,                # start cofeature_idx
        int,                # end cofeature idx
    )

    sigMzDiffMeasured = QtCore.pyqtSignal(
        int,    # cofeature_a_idx (spec_idx)
        int,    # cofeature_b_idx (spec_idx)
        float,  # delta_mz (snapshot at click time)
        int,    # ms_level
    )

    # Fired when the user picks a formula from the neutral-loss finder
    # popped up after a Δ m/z commit. Viewer attaches it to the most
    # recently created MzDiffAnnotation.
    sigMzDiffFormulaAssigned = QtCore.pyqtSignal(object)  # FormulaCandidate

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

        # Transient bracket shown while hovering
        self._transient_bracket_id: str | None = None

        # Tracks which path opened the finder so handle_formula_assigned
        # knows which signal to emit. Set by _open_finder_for_*; cleared
        # on hide/cancel.
        self._pending_finder_mode: Optional[Literal['ion', 'neutral_loss']] = None

        # Event filter to detect when the user closes the finder without
        # picking a formula — so we can reset the tool cleanly.
        self.formula_finder_menu.installEventFilter(self)

        self._connect_signals()

    def eventFilter(self, obj, event):
        if (
            obj is self.formula_finder_menu
            and event.type() == QtCore.QEvent.Hide
            and self._pending_finder_mode is not None
        ):
            # User dismissed the finder — clear state and exit tool.
            self._pending_finder_mode = None
            self.viewer.tool_manager.request_cancel()
        return False  # don't consume the event


    def _get_ms_plot(self, ms_level: Literal[1, 2]):
        return {1: self.viewer.ms1_plot, 2: self.viewer.ms2_plot}[ms_level]

    def _clear_transient_bracket(self):
        if self._transient_bracket_id and self.selected_ms_level:
            plot = self._get_ms_plot(self.selected_ms_level)
            plot.remove_delta_bracket(self._transient_bracket_id)
            self._transient_bracket_id = None

    def _connect_signals(self):
        self.formula_finder_menu.sigFormulaAssigned.connect(
            self.handle_formula_assigned
        )

    def on_activated(self):
        """
        Called when Measure Loss tool is activated
        """
        self.handle_clear_selections()
        # Request next stage (IDLE -> SELECTING)
        self.viewer.tool_manager.request_next_stage()

    def on_enter_pressed(self):
        """
        Called when user presses Enter while selecting signals.
        """
        self.handle_show_finder_menu()
        self.viewer.tool_manager.request_next_stage()
        return

    def on_cancelled(self):
        """
        Called when tool is cancelled
        """
        print('NLoss Cancelled')
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

            if len(self.selected_signals) > 0:
                # Clear transient bracket; emit signal for persistent annotation
                self._clear_transient_bracket()

                selected_mz, _, selected_spec_idx = self.selected_signals[0]
                # Snapshot delta_mz from the values the user actually saw,
                # not whatever the ensemble's peak_rt slice would return.
                delta_mz = abs(mz - selected_mz)
                self.sigMzDiffMeasured.emit(
                    selected_spec_idx,
                    spec_idx,
                    delta_mz,
                    ms_level,
                )

                # Open the formula finder pre-configured for a neutral
                # loss (charge = 0, single-row table at delta_mz). The
                # bracket already exists; if the user picks a formula
                # the viewer attaches it via sigMzDiffFormulaAssigned.
                # If they close without picking, the event filter on
                # Hide cancels the tool.
                self._open_finder_for_neutral_loss(delta_mz)
                return

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

    def handle_ms_signal_hovered(
        self,
        data: tuple[float, float, int],  # mz, intsy, spec_idx
        ms_level: Literal[1, 2],
    ):
        """
        If there's a signal selected, draw a transient bracket showing
        the m/z distance between hovered and selected signal.
        """

        if self.selected_ms_level != ms_level:
            if not self.selected_ms_level:
                self.selected_ms_level = ms_level
            else:
                self.handle_clear_selections()

        if self.viewer.tool_manager.active_stage == ToolStage.SELECTING:
            mz, intsy, spec_idx = data

            if len(self.selected_signals) == 0:
                # No signal selected yet
                return

            # Draw transient bracket between selected and hovered signal
            self._clear_transient_bracket()

            delta_mz = abs(mz - self.selected_signals[0][0])
            label = f"\u0394 {delta_mz:.4f}"

            plot = self._get_ms_plot(ms_level)
            selected_spec_idx = self.selected_signals[0][2]
            self._transient_bracket_id = plot.add_delta_bracket(
                spec_idx_a=selected_spec_idx,
                spec_idx_b=spec_idx,
                text=label,
                bracket_id='_transient',
            )

    def handle_clear_selections(self):
        self._clear_transient_bracket()
        self.selected_signals.clear()
        self.selected_ms_level = None
        self.sigSelectionCleared.emit()

    def handle_show_finder_menu(self):
        self._pending_finder_mode = 'ion'
        self.formula_finder_menu.show()

        # Send selected_signals to formula finder
        self.formula_finder_menu.populate_table(
            [(x[0], x[1]) for x in self.selected_signals]
        )

        self.formula_finder_menu.on_search_execute()

    def _open_finder_for_neutral_loss(self, delta_mz: float):
        """
        Pop the formula finder for a neutral-loss measurement: charge=0
        (so the m/z is interpreted as a neutral mass), single-row input
        table at `delta_mz`. The user picks a formula or closes; either
        way the bracket already exists, so no annotation is lost.
        """
        self._pending_finder_mode = 'neutral_loss'

        # Save the user's prior charge setting so the next ion-formula
        # search isn't accidentally locked at 0.
        self._saved_charge = self.formula_finder_menu.spinCharge.value()
        self.formula_finder_menu.spinCharge.setValue(0)

        self.formula_finder_menu.populate_table([(delta_mz, 1.0)])
        self.formula_finder_menu.show()
        self.formula_finder_menu.on_search_execute()

    def handle_formula_assigned(
        self,
        formula: 'FormulaCandidate'
    ):
        """
        Called when user selects a formula from finder dialogue. Routes
        based on which path opened the finder: neutral-loss → attach
        formula to the most recent MzDiffAnnotation; ion → existing
        ion-annotation flow.
        """
        mode = self._pending_finder_mode
        self._pending_finder_mode = None

        if mode == 'neutral_loss':
            self.sigMzDiffFormulaAssigned.emit(formula)
            # Restore the user's prior charge setting.
            if hasattr(self, '_saved_charge'):
                self.formula_finder_menu.spinCharge.setValue(self._saved_charge)
            self.formula_finder_menu.hide()
            # The Hide event filter will request_cancel.
            return

        # Legacy ion-formula path (Enter-press flow).
        self.sigFormulaAssigned.emit(
            formula,  # FormulaCandidate
            self.selected_ms_level,  # int
            [x[2] for x in self.selected_signals],  # feature coidxs (ints)
        )

        self.handle_clear_selections()
