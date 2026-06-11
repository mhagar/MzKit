from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from gui.resources.ProcessMonitorWindow import Ui_Form

from typing import Dict, Any, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from core.controllers.ProcessController import ProcessController

class ProcessMonitorWindow(
    QtWidgets.QWidget,
    Ui_Form,
):
    """
    Window for monitoring background processes
    """
    def __init__(
            self,
            process_controller: 'ProcessController'
    ):
        super().__init__()
        self.process_controller = process_controller
        self.process_outputs = {}  # Store process outputs for each process
        self.process_names = {}    # Store friendly names for processes

        self.setupUi(self)
        self.tableView.setModel(
            self.process_controller.model
        )
        self.tableView.resizeColumnsToContents()

        self.process_controller.process_signals.output_ready.connect(
            self.update_text_browser
        )

        # Cancel button: signals the selected process's cancel_event. Only
        # mzml_import and import_feature_table actually honour this today.
        self.pushButton.setText("Cancel Selected Process")
        self.pushButton.clicked.connect(self._cancel_selected)

    def _cancel_selected(self) -> None:
        """
        Request cancellation of whichever process row is selected
        """
        selection = self.tableView.selectionModel()
        if selection is None or not selection.hasSelection():
            return

        rows = selection.selectedRows()
        if not rows:
            # selectionBehavior is SelectRows but be defensive
            rows = [self.tableView.currentIndex()]

        model = self.process_controller.model
        for index in rows:
            if not index.isValid():
                continue
            # Column 0 is the process ID (stringified) — see ProcessTableModel
            pid_str = model.data(model.index(index.row(), 0), Qt.DisplayRole)
            try:
                pid = int(pid_str)
            except (TypeError, ValueError):
                continue
            self.process_controller.cancel_process(pid)


    def update_text_browser(
            self,
            process_id: int,
            level: str,
            msg: str,
    ):
        self.textBrowser.append(
            f"Process: {process_id} [{level}] {msg}"
        )