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
    sigImportFeatureTableRequested = QtCore.pyqtSignal()
    sigFilterAlignmentRequested = QtCore.pyqtSignal(
        list,  # list[QModelIndex]
    )
    sigExportAlignmentRequested = QtCore.pyqtSignal(
        list,  # list[QModelIndex]
    )
    sigLabelingRequested = QtCore.pyqtSignal(
        #
    )
    sigShowSampleViewerRequested = QtCore.pyqtSignal(
        list,  #list[QModelIndex]
    )
    sigShowAlignmentRequested = QtCore.pyqtSignal(
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
        self._configure_alignment_listview()

    def set_sample_list_model(
        self,
        model_type: Literal['Sample', 'Alignment'],
        model,
    ):
         match model_type:
            case 'Sample':
                self.listViewSamples.setModel(model)

            case 'Alignment':
                self.listViewAlignments.setModel(model)


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


    def _configure_alignment_listview(self):
        self.listViewAlignments.addAction(
            self.actionShowSelectedAlignment
        )

        self.listViewAlignments.doubleClicked.connect(
            self.actionShowSelectedAlignment.trigger
        )

        self.listViewAlignments.setContextMenuPolicy(
            QtCore.Qt.CustomContextMenu
        )

        self.listViewAlignments.customContextMenuRequested.connect(
            self._show_alignment_context_menu
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

        # Show selected alignment
        self.actionShowSelectedAlignment = QtWidgets.QAction(
            "Show alignment in viewer",
        )
        self.actionShowSelectedAlignment.triggered.connect(
            self._on_trigger_show_alignment,
        )

        # Filter alignment
        self.actionFilterAlignment = QtWidgets.QAction(
            "Filter Alignment...",
        )
        self.actionFilterAlignment.triggered.connect(
            self._on_trigger_filter_alignment,
        )

        # Export alignment as table
        self.actionExportAlignment = QtWidgets.QAction(
            "Export as Table...",
        )
        self.actionExportAlignment.triggered.connect(
            self._on_trigger_export_alignment,
        )


    def _on_trigger_import_mzmls(self) -> None:
        self.sigImportMzMLsRequested.emit()

    def _on_trigger_import_fingerprints(self) -> None:
        self.sigImportFingerprintsRequested.emit()

    def _on_trigger_import_metadata(self) -> None:
        self.sigImportMetadataRequested.emit()

    def _on_trigger_import_feature_table(self) -> None:
        self.sigImportFeatureTableRequested.emit()

    def _on_trigger_label_peaks(self) -> None:
        self.sigLabelingRequested.emit()

    def _on_trigger_show_samples(self) -> None:
        if not self.listViewSamples.selectionModel().hasSelection():
            return

        self.sigShowSampleViewerRequested.emit(
            self.listViewSamples.selectedIndexes()
        )

    def _on_trigger_show_alignment(self) -> None:
        if not self.listViewAlignments.selectionModel().hasSelection():
            return

        self.sigShowAlignmentRequested.emit(
            self.listViewAlignments.selectedIndexes()
        )

    def _show_samples_context_menu(self) -> None:
        pass

    def _on_trigger_filter_alignment(self) -> None:
        if not self.listViewAlignments.selectionModel().hasSelection():
            return

        self.sigFilterAlignmentRequested.emit(
            self.listViewAlignments.selectedIndexes()
        )

    def _on_trigger_export_alignment(self) -> None:
        if not self.listViewAlignments.selectionModel().hasSelection():
            return

        self.sigExportAlignmentRequested.emit(
            self.listViewAlignments.selectedIndexes()
        )

    def _show_alignment_context_menu(self, pos) -> None:
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.actionShowSelectedAlignment)
        menu.addAction(self.actionFilterAlignment)
        menu.addAction(self.actionExportAlignment)
        menu.exec_(self.listViewAlignments.mapToGlobal(pos))

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






