"""
Interactive formula search: the selected signals + the shared find-mfs
parameter sheet in, ranked candidates (FormulaResultsWidget) out.

The Ion combo is the adduct-assignment tool: "Unknown" or an IonType of
find-mfs's ION_VOCAB.

Two searches, both synchronous:
  - Ion ('Find Ion MF'): no MS2. Results are a transient (unregistered)
    FormulaAssignment, and assigning one annotates the selected signals
    (sigFormulaAssigned). Unknown searches the ion as observed (no adduct).
  - Compound ('Find Compound MF (MS2)'): needs the ensemble the envelope came
    from (`set_source`). Assigning one hands back the FormulaAssignment and the
    AdductNetwork it implies (sigCompoundAssigned); the host sets both. Unknown
    lets find-mfs's network decide the ion.

`FormulaFinderDialog.review` opens it on an ensemble's formula assignment
instead: the searched envelope frozen, the ion editable, re-search in place.
"""
from dataclasses import asdict

import numpy as np
from PyQt5 import QtWidgets, QtCore
from find_mfs import FormulaScorer, annotate_precursor
from find_mfs.spectra.halogen import HALOGEN_M2_OFFSET
from find_mfs.spectra.ions import ION_VOCAB, IonType

from gui.resources.FormulaFinderWindow import Ui_Form
from core.utils.array_types import to_spec_arr
from core.data_structs.formula_assignment import FormulaAssignment
from core.formula.assign_formula import (
    results_to_assigned_candidates,
    search_provenance,
    to_formula_candidate,
)
from core.formula.compound import ion_query, search_compound

from numpy.typing import NDArray
from configparser import ConfigParser
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.data_structs import Ensemble
    from core.interfaces.data_sources import EnsembleDataSource

# By default, use the pre-shipped COCONUT GMM.
# TODO: Expose route to user for training their own GMM
SCORER = FormulaScorer()


def ensemble_title(ensemble: 'Ensemble') -> str:
    """Window title naming an ensemble, e.g. 'Formula Assignment: S1 · m/z
    356.0126 · 4.21 min'."""
    return (
        f"Formula Assignment: {ensemble.injection.name} · "
        f"m/z {ensemble.base_mz:.4f} · {ensemble.peak_rt / 60:.2f} min"
    )


class FormulaFinderDialog(
    QtWidgets.QWidget,
    Ui_Form,
):
    sigFormulaAssigned = QtCore.pyqtSignal(
        object  # find_mfs.FormulaCandidate (ion path)
    )
    # Placeholder source for ion-search assignments, which are never registered
    ION_SEARCH_SOURCE = 0
    sigCompoundAssigned = QtCore.pyqtSignal(
        object,  # FormulaAssignment
        object,  # AdductNetwork, or None
    )

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
        self.comboIon.addItem("Unknown", None)
        for ion in ION_VOCAB:
            self.comboIon.addItem(ion.label, ion)

        self._connect_signals()
        self._setup_statusbar()

        # Per-search charge that overrides the shared param sheet without
        # persisting (the neutral-loss tool searches at charge 0).
        self.charge_override: Optional[int] = None

        # State stuff
        self.search_query: list[tuple[float, float]] = []
        # Which search produced the results on show (decides what Assign does)
        self._results_mode: Optional[Literal["ion", "compound"]] = None
        # The network the compound results on show imply
        self._compound_network = None
        # Where the selected envelope came from, for the compound search:
        # (ensemble, MS1 cofeature idxs, MS1 scan_num, MS2 override)
        self._source: Optional[tuple] = None
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
        self.btnFindCmpdMF.clicked.connect(self.on_compound_search_execute)

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

        ion = self.selected_ion() if self.charge_override is None else None
        if ion is not None:
            query_mz, adducts, _ = ion_query(search_mz, ion)
            charge = ion.charge
        else:
            # Unknown: the formula of the ion as observed
            charge = (
                self.charge_override if self.charge_override is not None
                else params.charge
            )
            query_mz, adducts = search_mz, [(None, charge)]

        results = annotate_precursor(
            query_mz,
            adducts=adducts,
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
            adducts=[ion.label] if ion else None,
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
            assignment: "FormulaAssignment",
            network=None,
    ):
        """Show a compound (MS2) search's results."""
        self._results_mode = "compound"
        self._compound_network = network
        self.formulaResults.set_assignment(assignment)
        ion = f" as {network.base_ion}" if network and network.base_ion else ""
        self.statusbar.showMessage(
            f"Compound (MS2){ion}: {len(assignment.candidates)} candidates — "
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

    @classmethod
    def review(
        cls,
        data_source: 'EnsembleDataSource',
        ensemble: 'Ensemble',
        parent=None,
        config: Optional[ConfigParser] = None,
    ) -> 'FormulaFinderDialog':
        """
        The finder on `ensemble`'s formula assignment: its envelope frozen,
        the current pick highlighted. Find Compound re-searches in place;
        Assign / Clear go through `data_source`, so every view refreshes.
        """
        dlg = cls(parent, config)
        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dlg.setWindowTitle(ensemble_title(ensemble))
        for widget in (dlg.btnAddSignal, dlg.btnRemoveSignal,
                       dlg.btnClearSignals, dlg.btnFindIonMF):
            widget.hide()
        dlg.tableInput.setEnabled(False)

        dlg._review = ensemble
        dlg.sigCompoundAssigned.connect(
            lambda a, network: data_source.update_ensemble(
                a.source_uuid, formula_assignment=a, adduct_network=network,
            )
        )
        dlg.formulaResults.sigClearRequested.connect(
            lambda: data_source.set_chosen_candidate(ensemble.uuid, None)
        )
        # Follow the registry: e.g. auto find-mfs re-run while this is open
        data_source.sigEnsembleAnnotated.connect(dlg._on_ensemble_annotated)
        dlg._show_review()
        return dlg

    def _on_ensemble_annotated(self, ensemble_uuid):
        if ensemble_uuid == self._review.uuid:
            self._show_review()

    def _show_review(self):
        ensemble = self._review
        assignment = ensemble.formula_assignment
        if assignment is None:
            self._results_mode = None
            self.formulaResults.clear("No formula assignment for this ensemble")
            return

        signals = []
        if assignment.envelope and assignment.scan_num is not None:
            spec = ensemble.get_spectrum(ms_level=1, scan_num=assignment.scan_num)
            signals = [(float(spec['mz'][i]), float(spec['intsy'][i]))
                       for i in assignment.envelope]
            self.populate_table(signals)
        self.set_source(
            ensemble if signals else None, assignment.envelope or (),
            assignment.scan_num,
            ensemble.composite_spectrum.ms2 if ensemble.is_dda else None,
        )

        network = ensemble.adduct_network
        self.comboIon.setCurrentIndex(max(0, self.comboIon.findText(
            network.base_ion if network and network.base_ion else "Unknown"
        )))
        self._results_mode = "compound"
        self._compound_network = network
        self.formulaResults.set_assignment(assignment, allow_clear=True)
        self.statusbar.showMessage(
            "" if signals else "No searched envelope recorded; nothing to re-search"
        )

    def selected_ion(self) -> Optional[IonType]:
        """The IonType picked in the Ion combo; None for Unknown."""
        return self.comboIon.currentData()

    def set_source(
        self,
        ensemble: Optional['Ensemble'],
        envelope: list[int] = (),
        scan_num: Optional[int] = None,
        ms2_override=None,
    ):
        """
        The ensemble the selected signals belong to (None: no compound search),
        their MS1 cofeature idxs and scan, and (DDA stopgap) the MS2 to use.
        """
        self._source = (
            (ensemble, list(envelope), scan_num, ms2_override)
            if ensemble is not None else None
        )

    def on_compound_search_execute(self):
        """
        'Find Compound MF (MS2)': rank compound formulae for the selected
        envelope with the ensemble's MS2, reading it as the picked ion.
        """
        self._retrieve_table_input()
        if not self.search_query:
            return
        if self._source is None:
            self.statusbar.showMessage(
                "Compound search needs an MS1 isotopologue group of an ensemble"
            )
            return
        error = self.findMfsParams.validation_error()
        if error:
            self.statusbar.showMessage(f"Invalid parameters: {error}")
            return

        ensemble, envelope, scan_num, ms2_override = self._source
        kwargs = {}
        if ms2_override is not None:
            kwargs = dict(ms2_spec=ms2_override, ms2_mode="selected_scan")

        self.statusbar.showMessage("Assigning compound formula (MS2)…")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            assignment, network = search_compound(
                ensemble, self.search_query, envelope, scan_num,
                self.selected_ion(), self.findMfsParams.get_params(), **kwargs,
            )
        except NotImplementedError:
            self.statusbar.showMessage(
                "Compound formula assignment is not yet supported for DDA."
            )
            return
        except ValueError as exc:   # find-mfs found no envelope at the selection
            self.statusbar.showMessage(str(exc))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self.populate_compound_results(assignment, network)

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
            self.sigCompoundAssigned.emit(assignment, self._compound_network)
        else:
            self.sigFormulaAssigned.emit(
                to_formula_candidate(assignment.candidates[idx])
            )
        self.close()
