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


    def update_text_browser(
            self,
            process_id: int,
            level: str,
            msg: str,
    ):
        self.textBrowser.append(
            f"Process: {process_id} [{level}] {msg}"
        )