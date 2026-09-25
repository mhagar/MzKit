"""
Interactive formula search: the selected signals + the shared find-mfs
parameter sheet in, ranked candidates (FormulaResultsWidget) out.

Two searches:
  - Ion ('Find Ion MF'): runs here, synchronously; no MS2. Results are a
    transient (unregistered) FormulaAssignment, and assigning one annotates
    the selected signals (sigFormulaAssigned).
  - Compound ('Find Compound MF (MS2)'): the host runs it (it needs the
    ensemble's MS2) and hands back a FormulaAssignment; assigning one
    registers it (sigCompoundAssigned).
"""
from dataclasses import asdict

import numpy as np
from PyQt5 import QtWidgets, QtCore
from find_mfs import FormulaScorer, annotate_precursor
from find_mfs.spectra.halogen import HALOGEN_M2_OFFSET

from gui.resources.FormulaFinderWindow import Ui_Form
from core.utils.array_types import to_spec_arr
from core.data_structs.formula_assignment import FormulaAssignment
from core.formula.assign_formula import (
    results_to_assigned_candidates,
    search_provenance,
    to_formula_candidate,
)

from numpy.typing import NDArray
from configparser import ConfigParser
from typing import Literal, Optional

# By default, use the pre-shipped COCONUT GMM.
# TODO: Expose route to user for training their own GMM
SCORER = FormulaScorer()

class FormulaFinderDialog(
    QtWidgets.QWidget,
    Ui_Form,
):
    sigFormulaAssigned = QtCore.pyqtSignal(
        object  # find_mfs.FormulaCandidate (ion path)
    )
    # Placeholder source for ion-search assignments, which are never registered
    ION_SEARCH_SOURCE = 0
    sigCompoundSearchRequested = QtCore.pyqtSignal(dict)
    sigCompoundAssigned = QtCore.pyqtSignal(object)  # FormulaAssignment

    def __init__(
        self,
        parent=None,
        config: Optional["ConfigParser"] = None,
        modal: bool = False,
    ):
        super().__init__(parent)
        if modal and parent is not None:
            self.setWindowFlags(QtCore.Qt.Dialog)
            self.setWindowModality(QtCore.Qt.WindowModal)
        elif parent is not None:
            # Non-modal but still a separate window
            self.setWindowFlags(QtCore.Qt.Window)

        self.setupUi(self)

        self._connect_signals()
        self._setup_statusbar()

        # Per-search charge that overrides the shared param sheet without
        # persisting (the neutral-loss tool searches at charge 0).
        self.charge_override: Optional[int] = None

        # State stuff
        self.search_query: list[tuple[float, float]] = []
        # Which search produced the results on show (decides what Assign does)
        self._results_mode: Optional[Literal["ion", "compound"]] = None
        self.config = config
        # The find-mfs parameter sheet (element/counts/RDBE/octet + scoring
        # groups) and all [findmfs] persistence live in the promoted widget,
        # shared with EnsembleExtractionDialog.
        self.findMfsParams.set_config(config)

    def _connect_signals(self):
        self.btnAddSignal.clicked.connect(self.tableInput.add_row)

        self.btnRemoveSignal.clicked.connect(self.tableInput.remove_last_row)

        self.btnClearSignals.clicked.connect(self.tableInput.clear_rows)

        # Two search buttons: ion (synchronous) vs compound/MS2 (via controller).
        self.btnFindIonMF.clicked.connect(self.on_search_execute)
        self.btnFindCmpdMF.clicked.connect(self._request_compound_search)

        # Assigning routes depending on ion search or cmpd search
        self.formulaResults.sigAssignRequested.connect(self.on_assign_selected)

    def _setup_statusbar(self):
        self.statusbar = QtWidgets.QStatusBar()
        # self.statusbar.setMaximumHeight(15)  # pixels
        self.verticalLayout.addWidget(self.statusbar)

    def on_search_execute(self):
        """
        Called when user hits 'Find Ion MF': rank formulae for the selected
        signals (no MS2) with find-mfs `annotate_precursor`, using the shared
        parameter sheet.
        """
        self._retrieve_table_input()

        if not self.search_query:
            return

        params = self.findMfsParams.get_params()
        error = self.findMfsParams.validation_error()
        if error:
            # Don't leave the previous search's results looking current
            self._results_mode = None
            self.formulaResults.clear()
            self.statusbar.showMessage(f"Invalid parameters: {error}")
            return

        envelope: NDArray = np.array(self.search_query)
        search_mz = float(envelope[:, 0].min())  # the monoisotopic peak

        has_envelope = envelope.shape[0] > 1
        spec = to_spec_arr(envelope[:, 0], envelope[:, 1]) if has_envelope else None

        adduct = self.lineAdduct.text().strip() or None
        charge = (
            self.charge_override if self.charge_override is not None
            else params.charge
        )

        results = annotate_precursor(
            search_mz,
            adducts=[(adduct, charge)],
            scorer=SCORER,
            ms1_peaks=spec,
            **params.search_kwargs(),
        )

        assignment = FormulaAssignment(
            source_uuid=self.ION_SEARCH_SOURCE,
            source_kind='signal',
            candidates=results_to_assigned_candidates(
                results, ms2_weight=params.ms2_weight, top_n=len(results),
            ),
            precursor_mz=search_mz,
            charge=charge,
            adducts=[adduct] if adduct else None,
            params=asdict(params),
            **search_provenance(results),
        )

        self._results_mode = "ion"
        note = self._halogen_hint(assignment, envelope, charge)
        self.formulaResults.set_assignment(assignment, note=note)

        found = (
            f"Found {len(assignment.candidates)} formulae" if assignment.candidates
            else "No formulae found"
        )
        self.statusbar.showMessage(f"{found} for m/z {search_mz:.4f}")

    @staticmethod
    def _halogen_hint(
        assignment: FormulaAssignment,
        envelope: NDArray,
        charge: int,
    ) -> str:
        """Explain a negative halogen check that couldn't have been positive."""
        if assignment.halogen_detected is not False:
            return ""
        # Detection needs the M+2 peak; say so if it wasn't selected
        m2_mz = envelope[:, 0].min() + HALOGEN_M2_OFFSET / (abs(charge) or 1)
        if not np.any(np.abs(envelope[:, 0] - m2_mz) < 0.05):
            return "select the M+2 peak for halogen detection"
        return ""

    def populate_table(
        self,
        data: list[tuple[float, float]],
    ):
        """
        Populates the input table programmatically, given a
        list of mz values and a list of intensities
        """
        self.tableInput.populate_table(data)

    def populate_compound_results(
            self,
            assignment: "FormulaAssignment"
    ):
        """
        Public: called by the host after a compound (MS2) search.
        """
        self._results_mode = "compound"
        self.formulaResults.set_assignment(assignment)
        self.statusbar.showMessage(
            f"Compound (MS2): {len(assignment.candidates)} candidates — "
            f"select a row and hit Assign Selected"
        )

    def _retrieve_table_input(self):
        """
        Retrieves the user input from the table
        """
        user_input = self.tableInput.get_user_input()

        if user_input:
            self.search_query = user_input

        else:
            # Something went wrong; show warning
            self.statusbar.showMessage("Invalid search query")

            self.search_query = []

            return

    def _request_compound_search(self):
        """
        Ask the controller to run the MS2 compound assignment
        """
        self._retrieve_table_input()
        if not self.search_query:
            self.statusbar.showMessage(
                "Select the MS1 isotopologue group first"
            )
            return

        error = self.findMfsParams.validation_error()
        if error:
            self.statusbar.showMessage(f"Invalid parameters: {error}")
            return

        adduct = self.lineAdduct.text().strip() or None
        self.sigCompoundSearchRequested.emit({
            "ms1_signals": list(self.search_query),
            "adducts": [adduct] if adduct else None,
            "params": self.findMfsParams.get_params(),
        })

    def on_assign_selected(self, idx: int):
        """
        Commit candidate `idx` of the results on show: an ion search annotates
        the selected signals, a compound search registers the assignment.
        """
        assignment = self.formulaResults.assignment
        if assignment is None:
            return

        if self._results_mode == "compound":
            assignment.chosen_idx = idx
            assignment.chosen_by = 'user'
            self.sigCompoundAssigned.emit(assignment)
        else:
            self.sigFormulaAssigned.emit(
                to_formula_candidate(assignment.candidates[idx])
            )
        self.close()
