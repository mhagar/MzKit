"""
This tool is pretty crude still; lots to polish.
Rushed to make something useable
"""
import numpy as np
from PyQt5 import QtWidgets, QtCore
from find_mfs import FormulaFinder, IsotopeMatchConfig, FormulaPrior

from gui.resources.FormulaFinderWindow import Ui_Form
from core.utils.config import save_config
from core.utils.formula_formatting import format_formula_obj_to_html
from gui.dialogues.formula_finder.tables import HTMLDelegate

from numpy.typing import NDArray
from configparser import ConfigParser
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from find_mfs import FormulaSearchResults, FormulaCandidate

# By default, use the pre-shipped COCONUT GMM.
# TODO: Expose route to user for training their own GMM
SCORER = FormulaPrior.default()

class FormulaFinderDialog(
    QtWidgets.QWidget,
    Ui_Form,
):
    sigFormulaAssigned = QtCore.pyqtSignal(
        object  # find_mfs.FormulaCandidate
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

        self._connect_signals()
        self._setup_statusbar()
        self._setup_results_table()

        # Formula finder
        # TODO: expose element params to user
        self.finder = FormulaFinder()

        # State stuff
        self.search_query: list[tuple[float, float]] = []
        self.search_results: Optional[FormulaSearchResults] = None
        self.config = config
        self._load_params_from_config()

        print("config:")
        print({section: dict(config[section]) for section in config.sections()})

    def _connect_signals(self):
        self.btnAddSignal.clicked.connect(self.tableInput.add_row)

        self.btnRemoveSignal.clicked.connect(self.tableInput.remove_last_row)

        self.btnClearSignals.clicked.connect(self.tableInput.clear_rows)

        self.btnSearch.clicked.connect(self.on_search_execute)

        self.btnConfigBox.clicked.connect(self.on_config_btn_pressed)

        self.btnAnnotateSelectedSignals.clicked.connect(self.annotate_selected_signals)

        self.tableResults.doubleClicked.connect(self.annotate_selected_signals)

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

        mf_params, isotope_params = self._retrieve_params_from_ui()

        iso_matching_config: Optional["IsotopeMatchConfig"] = None
        if envelope.shape[0] > 1:
            # Search query contains isotope envelopes
            iso_matching_config = IsotopeMatchConfig(
                envelope=envelope,
                **isotope_params,
            )

        results = self.finder.find_formulae(
            mass=search_mz,
            isotope_match=iso_matching_config,
            **mf_params,
        )

        SCORER.score_results(
            results=results,
            mass_sigma_ppm=mf_params['error_ppm']/3,
            isotope_sigma=isotope_params['minimum_rmse']/3,
        )

        match self._retrieve_requested_sort():
            case "mass_error":
                self.search_results = results.sort_by_error()

            case "envelope_rmse":
                self.search_results = results.sort_by_rmse()

            case "prior":
                self.search_results = results.sort_by_prior()

            case "posterior":
                self.search_results = results.sort_by_posterior()

        # self.search_results = results.sort_by_posterior()

        self._populate_results_table()

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

    def _populate_results_table(
        self,
    ):
        """
        Populates the table using a FormulaSearchResults object
        from `find_mfs`
        """
        self.tableResults.setRowCount(0)  # Delete all rows

        if not self.search_results:
            return

        for candidate in self.search_results[::-1]:
            candidate: "FormulaCandidate"

            self.tableResults.insertRow(0)
            for col_idx, text in [
                (0, f"{format_formula_obj_to_html(candidate.formula)}"),
                (1, f"{candidate.error_ppm:.2f}"),
                (2, f"{candidate.error_da:.6f}"),
                (3, f"{candidate.rdbe:.1f}"),
                (4, _get_intensity_rmse(candidate)),
                (5, f"{candidate.prior_score:.2f}"),
                (6, f"{candidate.posterior_score:.2f}"),
            ]:
                item = QtWidgets.QTableWidgetItem(
                    text,
                )

                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

                self.tableResults.setItem(
                    0,  # Row
                    col_idx,  # Col
                    item,
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
    ) -> Literal["mass_error", "envelope_rmse", "prior", "posterior"]:
        """
        Retrieves state of 'sort by' combobox from UI
        """
        if "mass" in self.comboSortBy.currentText().lower():
            return "mass_error"

        if "envelope" in self.comboSortBy.currentText().lower():
            return "envelope_rmse"

        if "prior" in self.comboSortBy.currentText().lower():
            return "prior"

        if "posterior" in self.comboSortBy.currentText().lower():
            return "posterior"

        raise ValueError(
            f"Invalid combobox state: '{self.comboSortBy.currentText()}'. \n"
            f"Must contain either 'mass' or 'envelope'"
        )

    def _retrieve_params_from_ui(self) -> tuple[dict, dict]:
        """
        Retrieves parameters from UI
        """
        mf_params = {
            "adduct": self.lineAdduct.text() or None,
            "charge": self.spinCharge.value(),
            "error_ppm": self.spinErrorPpm.value() * 3, # Search with 3x requested tol
            "error_da": self.spinErrorDa.value() * 3,  # This is readjusted in posterior score
            "min_counts": self.lineMinCounts.text(),
            "max_counts": self.lineMaxCounts.text(),
            "filter_rdbe": (
                self.spinRDBEMin.value(),
                self.spinRDBEMax.value(),
            ),
            "check_octet": self.checkOctet.isChecked(),
        }

        isotope_params = {
            "mz_tolerance_ppm": self.spinErrorPpmIsotopes.value(),
            "mz_tolerance_da": self.spinErrorDaIsotopes.value(),
            "minimum_rmse": self.spinMinIsotopeRMSE.value() / 100,  # Search w 3x requested tol
        }                                                          # This is readjusted in posterior score

        self._check_finder_element_set(self.comboElementSet.currentText())

        return mf_params, isotope_params

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

    def on_config_btn_pressed(
            self,
            button: QtWidgets.QAbstractButton,
    ):
        # Get which standard button was clicked
        standard_button = self.btnConfigBox.standardButton(button)

        match standard_button:
            case QtWidgets.QDialogButtonBox.StandardButton.Save:
                self._write_params_to_config()

            case QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults:
                # TODO: Implement this
                print("Not yet implemented :))")

            case QtWidgets.QDialogButtonBox.StandardButton.Reset:
                self._load_params_from_config()

    def annotate_selected_signals(
        self,
    ):
        """
        Called when user selects 'annotate selected signals', or double clicks a
        row in the results table.

        Emits a 'sigFormulaAssigned' QSignal
        """
        # Do nothing if no row is selected
        selected_rows = self.tableResults.selectedItems()
        if not selected_rows or not self.search_results:
            return

        formula_candidate = self.search_results[selected_rows[0].row()]

        self.sigFormulaAssigned.emit(formula_candidate)

        self.close()

    def _write_params_to_config(
        self,
    ):
        if not self.config:
            return

        self.config.set(
            section="findmfs", option="charge", value=str(self.spinCharge.value())
        )
        self.config.set(
            section="findmfs", option="error_ppm", value=str(self.spinErrorPpm.value())
        )
        self.config.set(
            section="findmfs", option="error_da", value=str(self.spinErrorDa.value())
        )
        self.config.set(
            section="findmfs", option="min_counts", value=str(self.lineMinCounts.text())
        )
        self.config.set(
            section="findmfs", option="max_counts", value=str(self.lineMaxCounts.text())
        )
        self.config.set(
            section="findmfs", option="min_rdbe", value=str(self.spinRDBEMin.value())
        )
        self.config.set(
            section="findmfs", option="max_rdbe", value=str(self.spinRDBEMax.value())
        )
        self.config.set(
            section="findmfs",
            option="check_octet",
            value=str(self.checkOctet.isChecked()),
        )

        # === Isotope Matching ===
        self.config.set(
            section="findmfs",
            option="min_isotope_rmse",
            value=str(self.spinMinIsotopeRMSE.value()),
        )
        self.config.set(
            section="findmfs",
            option="isotope_error_ppm",
            value=str(self.spinErrorPpmIsotopes.value()),
        )
        self.config.set(
            section="findmfs",
            option="isotope_error_da",
            value=str(self.spinErrorDaIsotopes.value()),
        )

        save_config(self.config)

    def _load_params_from_config(
        self,
    ):
        """
        Skips if not initialized with configparser argument
        """
        if not self.config:
            return

        self.spinCharge.setValue(self.config.getint("findmfs", "charge", fallback=0))

        self.spinErrorPpm.setValue(
            self.config.getfloat("findmfs", "error_ppm", fallback=0.0)
        )

        self.spinErrorDa.setValue(
            self.config.getfloat("findmfs", "error_da", fallback=0.0)
        )

        self.lineMinCounts.setText(
            self.config.get("findmfs", "min_counts", fallback="")
        )

        self.lineMaxCounts.setText(
            self.config.get("findmfs", "max_counts", fallback="")
        )

        self.spinRDBEMin.setValue(
            self.config.getfloat("findmfs", "min_rdbe", fallback=0.0)
        )

        self.spinRDBEMax.setValue(
            self.config.getfloat("findmfs", "max_rdbe", fallback=0.0)
        )

        self.checkOctet.setChecked(
            self.config.getboolean("findmfs", "check_octet", fallback=True)
        )

        self.spinMinIsotopeRMSE.setValue(
            self.config.getfloat("findmfs", "min_isotope_rmse", fallback=10)
        )

        self.spinErrorPpmIsotopes.setValue(
            self.config.getfloat("findmfs", "error_ppm", fallback=0.0)
        )

        self.spinErrorDaIsotopes.setValue(
            self.config.getfloat("findmfs", "error_da", fallback=0.1)
        )


def _get_intensity_rmse(candidate: "FormulaCandidate") -> str:
    """
    Helper; returns either a FormulaCandidate's isotope envelope rmse,
    or '' if no isotope matching was performed
    """
    if candidate.isotope_match_result is None:
        return ""

    return f"{candidate.isotope_match_result.intensity_rmse:.2f}"
