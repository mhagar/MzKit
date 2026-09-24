"""
This tool is pretty crude still; lots to polish.
Rushed to make something useable
"""
import numpy as np
from PyQt5 import QtWidgets, QtCore
from find_mfs import FormulaFinder, FormulaScorer

from gui.resources.FormulaFinderWindow import Ui_Form
from core.utils.formula_formatting import format_formula_obj_to_html
from gui.dialogues.formula_finder.tables import HTMLDelegate
from core.utils.array_types import to_spec_arr

from molmass import Formula
from numpy.typing import NDArray
from configparser import ConfigParser
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from find_mfs import FormulaSearchResults, FormulaCandidate
    from core.data_structs.formula_assignment import FormulaAssignment

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
        self._setup_results_table()

        # Formula finder
        # TODO: expose element params to user
        self.finder = FormulaFinder()

        # State stuff
        self.search_query: list[tuple[float, float]] = []
        self.search_results: Optional[FormulaSearchResults] = None
        self._results_mode: Optional[Literal["ion", "compound"]] = None  # Whether mf search was for ion or compound
        self._compound_assignment: Optional["FormulaAssignment"] = None  # Whether a compound mf assignment was made
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

        # "Assign Selected" btn routes depending on ion search or cmpd search
        self.btnAssignSelected.clicked.connect(self.on_assign_selected)
        self.tableResults.doubleClicked.connect(self.on_assign_selected)

    def _setup_statusbar(self):
        self.statusbar = QtWidgets.QStatusBar()
        # self.statusbar.setMaximumHeight(15)  # pixels
        self.verticalLayout.addWidget(self.statusbar)

    def _setup_results_table(self):
        """
        Configure results tablewidget
        """
        # Apply custom delegates to table columns, necessary for
        # styling chemical formulae with subscripts
        self.tableResults.setItemDelegateForColumn(
            0,
            HTMLDelegate(self.tableResults),
        )

    def on_search_execute(self):
        """
        Called when user hits 'Find MFs' button
        """
        self._retrieve_table_input()

        if not self.search_query:
            return

        envelope: NDArray = np.array(self.search_query)
        search_mz = envelope[:, 0].min()  # Uses lowest m/z.. for now?

        mf_params, score_params = self._retrieve_params_from_ui()

        has_envelope = envelope.shape[0] > 1
        spec = to_spec_arr(envelope[:, 0], envelope[:, 1]) if has_envelope else None

        results = self.finder.find_formulae(
            mass=search_mz,
            # Perf-only prefilter; actual isotope scoring happens below via SCORER.score()
            isotope_prefilter=spec,
            **mf_params,
        )

        if len(results) == 0:
            self.search_results = None
            self._results_mode = "ion"
            self._compound_assignment = None
            self._populate_ion_results()
            self.statusbar.showMessage(
                f"No formulae found for m/z {search_mz}"
                + (" matching the isotope envelope" if has_envelope else "")
            )
            return

        SCORER.score(
            results,
            ms1_peaks=spec,
            precursor_mz=search_mz,
            mass_sigma_ppm=mf_params['error_ppm'] / 3,
            **score_params,
        )

        match self._retrieve_requested_sort():
            case "mass_error":
                self.search_results = results.sort_by_error()

            case "isotope_envelope":
                self.search_results = results.sort_by_iso_loglik()

            case "chemical_prior":
                self.search_results = results.sort_by_chem_logprior()

            case "posterior":
                self.search_results = results.sort_by_posterior()

        self._results_mode = "ion"
        self._compound_assignment = None
        self._populate_ion_results()

        self.statusbar.showMessage(
            f"Found {len(self.search_results)} formulae for m/z {search_mz}"
        )

    def populate_table(
        self,
        data: list[tuple[float, float]],
    ):
        """
        Populates the input table programmatically, given a
        list of mz values and a list of intensities
        """
        self.tableInput.populate_table(data)

    # Results table columns (10):
    #   0 Formula  1 Adduct  2 Error(ppm)  3 Error(Da)  4 RDBE
    #   5 Mass LL  6 Iso LL  7 Chem Prior  8 MS2 LL     9 Posterior

    def _insert_result_row(
            self,
            values: list[str]
    ):
        """
        Insert a row at the top
        (iterating results in reverse leaves the table in results order
         i.e. row index == results index).
        """
        self.tableResults.insertRow(0)
        for col_idx, text in enumerate(values):
            item = QtWidgets.QTableWidgetItem(text)
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.tableResults.setItem(0, col_idx, item)

    def _populate_ion_results(self):
        """
        Populate the table from ion-search results (find_mfs candidates).
        MS2 column is blank — the ion path has no MS2 term.
        """
        self.tableResults.setRowCount(0)
        if not self.search_results:
            return
        for c in self.search_results[::-1]:
            c: "FormulaCandidate"
            self._insert_result_row([
                format_formula_obj_to_html(c.formula),
                c.adduct or "",
                _fmt(c.error_ppm, ".2f"),
                _fmt(c.error_da, ".6f"),
                _fmt(c.rdbe, ".1f"),
                _fmt(c.mass_loglik, ".2f"),
                _fmt(c.iso_loglik, ".2f"),
                _fmt(c.chem_logprior, ".2f"),
                "",
                _fmt(c.log_posterior, ".2f"),
            ])

    def populate_compound_results(
            self,
            assignment: "FormulaAssignment"
    ):
        """
        Public: called by the controller after a compound (MS2) search.
        """
        self._compound_assignment = assignment
        self._results_mode = "compound"
        self.search_results = None

        self.tableResults.setRowCount(0)
        for c in assignment.candidates[::-1]:
            self._insert_result_row([
                format_formula_obj_to_html(Formula(c.formula_str)),
                c.adduct or "",
                _fmt(c.error_ppm, ".2f"),
                _fmt(c.error_da, ".6f"),
                _fmt(c.rdbe, ".1f"),
                _fmt(c.mass_loglik, ".2f"),
                _fmt(c.iso_loglik, ".2f"),
                _fmt(c.chem_logprior, ".2f"),
                _fmt(c.ms2_loglik, ".2f"),
                _fmt(c.log_posterior, ".2f"),
            ])
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

    def _retrieve_requested_sort(
        self,
    ) -> Literal["mass_error", "isotope_envelope", "chemical_prior", "posterior"]:
        """
        Retrieves state of 'sort by' combobox from UI
        """
        text = self.comboSortBy.currentText().lower()

        if "mass" in text:
            return "mass_error"

        if "isotope" in text:
            return "isotope_envelope"

        if "chemical" in text or "prior" in text:
            return "chemical_prior"

        if "posterior" in text:
            return "posterior"

        raise ValueError(
            f"Invalid combobox state: '{self.comboSortBy.currentText()}'. \n"
            f"Must contain 'mass', 'isotope', 'chemical', or 'posterior'"
        )

    def _retrieve_params_from_ui(self) -> tuple[dict, dict]:
        """
        Retrieves parameters from UI. The scoring/constraint controls live in the
        shared find-mfs param widget; the adduct is FormulaFinder-local.
        """
        mf_params = {
            "adduct": self.lineAdduct.text() or None,
            **self.findMfsParams.get_mf_params(),
        }

        score_params = self.findMfsParams.get_score_params()

        self._check_finder_element_set(
            self.findMfsParams.get_element_set_text()
        )

        return mf_params, score_params

    def _check_finder_element_set(
        self, element_set: Literal["CHNOPS", "CHNOPS + Halogens"]
    ):
        """
        Checks whether the FormulaFinder object needs to be re-instantiated
        (i.e. user has changed the element set)
        """
        element_set = {
            "CHNOPS": {
                "C",
                "H",
                "N",
                "O",
                "P",
                "S",
            },
            "CHNOPS + Halogens": {
                "C",
                "H",
                "N",
                "O",
                "P",
                "S",
                "F",
                "Br",
                "I",
                "Cl",
            },
        }[element_set]

        if element_set != self.finder.element_set:
            print(
                f"DEBUGGING: User requested element set {element_set},"
                f" but finder is using {self.finder.element_set}. "
                f"Reinstantiating."
            )

            self.finder = FormulaFinder(element_set)

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

        params = self._retrieve_compound_params()
        params["ms1_signals"] = list(self.search_query)
        self.sigCompoundSearchRequested.emit(params)

    def _retrieve_compound_params(self) -> dict:
        """
        Assemble kwargs for core.formula.query_from_signals + a top_n. The
        scoring/constraint controls come from the shared param widget; the
        adduct is FormulaFinder-local.
        """
        adduct = self.lineAdduct.text().strip() or None
        params = self.findMfsParams.get_compound_params()
        params["adducts"] = [adduct] if adduct else None
        return params

    def on_assign_selected(self):
        """
        Commit the highlighted row as
         either ion annotation or compound assignment,
        depending on which search produced the current results.
        """
        selected = self.tableResults.selectedItems()
        if not selected:
            return
        row = selected[0].row()

        if self._results_mode == "compound":
            if self._compound_assignment is None:
                return
            self._compound_assignment.chosen_idx = row
            self.sigCompoundAssigned.emit(self._compound_assignment)
            self.close()
            return

        # Ion path (unchanged)
        if not self.search_results:
            return
        self.sigFormulaAssigned.emit(self.search_results[row])
        self.close()


def _fmt(value, spec: str) -> str:
    """Format a numeric score term for the table, or '' when it is None."""
    if value is None:
        return ""
    return format(value, spec)
