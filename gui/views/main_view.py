from PyQt5 import QtWidgets, QtCore, QtGui

from gui.resources.MainWindow import Ui_MainWindow

from enum import Enum, auto
from typing import Literal


class ContextMenuAction(Enum):
    """
    Currently used to enumerate actions in 'Injection Tree' context menu
    """
    SHOW_IN_CURRENT_WINDOW = auto()
    SHOW_IN_NEW_WINDOW = auto()


class MainView(
    QtWidgets.QMainWindow,
    Ui_MainWindow,
):
    sigImportFingerprintsRequested = QtCore.pyqtSignal()
    sigImportMzMLsRequested = QtCore.pyqtSignal()
    sigImportMetadataRequested = QtCore.pyqtSignal()
    sigImportAnalyteTableRequested = QtCore.pyqtSignal()
    sigShowSampleViewerRequested = QtCore.pyqtSignal(
        list,  #list[QModelIndex]
    )
    sigShowAnalyteTableRequested = QtCore.pyqtSignal(
        list, # list[QModelIndex]
    )
    sigSampleFilterChanged = QtCore.pyqtSignal(
        str,  # filter_text
        bool, # show_injections
        bool, # show_fingerprints
    )

    def __init__(
            self,
            parent=None,
    ) -> None:
        super().__init__(parent)
        self.setupUi(self)
        self._create_actions()
        self._configure_sample_listview()
        self._configure_analyte_table_listview()

    def set_sample_list_model(
        self,
        model_type: Literal['Sample', 'AnalyteTable'],
        model,
    ):
         match model_type:
            case 'Sample':
                self.listViewSamples.setModel(model)

            case 'AnalyteTable':
                self.listViewAnalyteTables.setModel(
                    model,
                )


    def _configure_sample_listview(self):
        self.listViewSamples.addAction(
            self.actionShowSelectedSamples
        )

        self.listViewSamples.doubleClicked.connect(
            self.actionShowSelectedSamples.trigger
        )

        self.listViewSamples.setContextMenuPolicy(
            QtCore.Qt.CustomContextMenu
        )

        self.listViewSamples.customContextMenuRequested.connect(
            self._show_samples_context_menu
        )


    def _configure_analyte_table_listview(self):
        self.listViewAnalyteTables.addAction(
            self.actionShowSelectedAnalyteTable
        )

        self.listViewAnalyteTables.doubleClicked.connect(
            self.actionShowSelectedAnalyteTable.trigger
        )

        self.listViewAnalyteTables.setContextMenuPolicy(
            QtCore.Qt.CustomContextMenu
        )

        self.listViewAnalyteTables.customContextMenuRequested.connect(
            self._show_analyte_table_context_menu
        )


    def _create_actions(self):
        """
        Creates actions that appear in context menu.
        These can't be made in Qt designer
        :return:
        """
        # Show selected sample in current window
        self.actionShowSelectedSamples = QtWidgets.QAction(
            "Show samples in viewer",
        )
        self.actionShowSelectedSamples.setShortcut(
            QtGui.QKeySequence(
                QtCore.Qt.CTRL +
                QtCore.Qt.SHIFT +
                QtCore.Qt.Key_O
            )
        )
        self.actionShowSelectedSamples.triggered.connect(
            self._on_trigger_show_samples
        )

        # Show selected analyte table
        self.actionShowSelectedAnalyteTable = QtWidgets.QAction(
            "Show analyte table in viewer",
        )
        self.actionShowSelectedAnalyteTable.triggered.connect(
            self._on_trigger_show_analyte_table,
        )


    def _on_trigger_import_mzmls(self) -> None:
        self.sigImportMzMLsRequested.emit()

    def _on_trigger_import_fingerprints(self) -> None:
        self.sigImportFingerprintsRequested.emit()

    def _on_trigger_import_metadata(self) -> None:
        self.sigImportMetadataRequested.emit()

    def _on_trigger_import_analyte_table(self) -> None:
        self.sigImportAnalyteTableRequested.emit()

    def _on_trigger_show_samples(self) -> None:
        if not self.listViewSamples.selectionModel().hasSelection():
            return

        self.sigShowSampleViewerRequested.emit(
            self.listViewSamples.selectedIndexes()
        )

    def _on_trigger_show_analyte_table(self) -> None:
        if not self.listViewAnalyteTables.selectionModel().hasSelection():
            return

        self.sigShowAnalyteTableRequested.emit(
            self.listViewAnalyteTables.selectedIndexes()
        )

    def _show_samples_context_menu(self) -> None:
        pass

    def _show_analyte_table_context_menu(self) -> None:
        pass

    def _on_filter_changed(self) -> None:
        """
        Called whenever user either types into 'Filter...' box under
        sample list, or when they check/uncheck the 'I'/'F' buttons
        """
        # Retrieve filter text
        filter_text: str = self.lineViewSampleFilter.text()

        # Retrieve button states
        injs_requested: bool = self.btnFilterInjection.isChecked()
        fps_requested: bool = self.btnFilterFPrint.isChecked()

        # Emit signal to controller
        self.sigSampleFilterChanged.emit(
            filter_text,
            injs_requested,
            fps_requested,
        )






