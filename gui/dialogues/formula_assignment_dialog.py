"""
Review (and change) the formula assignment registered for one ensemble.

Wraps a FormulaResultsWidget around the registered FormulaAssignment: the
current pick is highlighted, 'Assign Selected' accepts a different candidate,
'Clear Assignment' withdraws the choice (the display falls back to the top
candidate, marked '?'). Changes go through the data source, so every view
showing the formula refreshes.
"""
from typing import Optional, TYPE_CHECKING

from PyQt5 import QtCore, QtWidgets

from gui.widgets.formula_results_widget import FormulaResultsWidget

if TYPE_CHECKING:
    from core.data_structs import Ensemble
    from core.data_structs.uuid_types import EnsembleUUID
    from core.data_structs.formula_assignment import FormulaAssignment
    from core.interfaces.data_sources import EnsembleDataSource


def ensemble_title(ensemble: 'Ensemble') -> str:
    """Window title naming an ensemble, e.g. 'Formula Assignment: S1 · m/z
    356.0126 · 4.21 min'."""
    return (
        f"Formula Assignment: {ensemble.injection.name} · "
        f"m/z {ensemble.base_mz:.4f} · {ensemble.peak_rt / 60:.2f} min"
    )


class FormulaAssignmentDialog(QtWidgets.QDialog):
    def __init__(
        self,
        data_source: 'EnsembleDataSource',
        source_uuid: 'EnsembleUUID',
        *,
        title: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle(title or "Formula Assignment")
        self.resize(900, 500)

        self.data_source = data_source
        self.source_uuid = source_uuid

        self.results = FormulaResultsWidget(self)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.results)
        layout.addWidget(buttons)

        self.results.sigAssignRequested.connect(self._on_assign)
        self.results.sigClearRequested.connect(self._on_clear)

        # Follow the registry: e.g. auto find-mfs re-run while this is open
        data_source.sigEnsembleAnnotated.connect(self._on_ensemble_annotated)

        self._refresh()

    def _assignment(self) -> Optional['FormulaAssignment']:
        ensemble = self.data_source.get_ensemble(self.source_uuid)
        return ensemble.formula_assignment if ensemble is not None else None

    def _refresh(self) -> None:
        assignment = self._assignment()
        if assignment is None:
            self.results.clear("No formula assignment for this ensemble")
            return
        self.results.set_assignment(assignment, allow_clear=True)

    def _on_assign(self, idx: int) -> None:
        self.data_source.set_chosen_candidate(self.source_uuid, idx)
        self.accept()

    def _on_clear(self) -> None:
        # Stay open: the user may want to pick another candidate instead
        self.data_source.set_chosen_candidate(self.source_uuid, None)

    def _on_ensemble_annotated(self, ensemble_uuid: 'EnsembleUUID') -> None:
        if ensemble_uuid == self.source_uuid:
            self._refresh()
