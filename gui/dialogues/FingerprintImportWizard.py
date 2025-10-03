"""
A QWizard for importing activity fingerprints.

Allows the user to choose which samples/descriptors to import
"""
from PyQt5 import QtWidgets, QtCore
import pandas as pd
import numpy as np

from gui.resources.FingerprintImportWizardWindow import Ui_Wizard
from core.data_structs.fingerprint import FingerprintImportParams

from pathlib import Path


class FingerprintImportWizard(
    QtWidgets.QWizard,
    Ui_Wizard,
):
    sigImportParamsGiven = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Page 1
        self.wizardPage1.registerField("csvPath*", self.lineEdit)
        self.wizardPage1.registerField("rowsSamples", self.radioRowSamples)

        self.currentIdChanged.connect(
            self._on_page_changed
        )

        # Monkey-patch isComplete method
        self.wizardPage1.isComplete = lambda: page1_is_complete(
            self.wizardPage1
        )

        # Page 2
        self.samples: list[str] = []
        self.descriptors: list[str] = []

    def validateCurrentPage(self):
        """
        Overrides wizard method - called when user tries to advance
        """
        match self.currentPage():
            case self.wizardPage1:
                path = self.field("csvPath")
                if not validate_csv_integrity(path):
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Invalid .csv file",
                        "The selected file is not a valid .csv, or cannot be read",
                    )
                    return False

        return super().validateCurrentPage()

    def _on_page_changed(
        self,
        page_id: int,
    ):
        """
        Handles population of page 2
        """
        print(f"page changed. page_id: {page_id}")
        match page_id:
            case 1:
                self.populateListWidgets()

    def _user_triggered_browse_btn(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Select .csv file containing activity fingerprints",
            filter="*.csv"
        )

        if not file:
            return

        self.lineEdit.setText(file)

    def populateListWidgets(self):
        df = pd.read_csv(
            self.field("csvPath"),
            index_col=0,
        )

        # Transpose if columns represent samples
        if not self.field("rowsSamples"):
            df = df.T
            # TODO: test whether reset column names needed

        # Populate sample listwidget
        self.listWidgetSamples.clear()
        for samplename in df.index:
            item = QtWidgets.QListWidgetItem(
                str(samplename)
            )
            item.setFlags(
                item.flags() | QtCore.Qt.ItemIsUserCheckable
            )
            item.setCheckState(QtCore.Qt.Checked)

            self.listWidgetSamples.addItem(item)
            self.samples.append(samplename)

        # Populate descriptor listwidget
        self.listWidgetDescriptors.clear()
        for descriptorname in df.columns:
            item = QtWidgets.QListWidgetItem(
                str(descriptorname)
            )
            item.setFlags(
                item.flags() | QtCore.Qt.ItemIsUserCheckable
            )
            item.setCheckState(QtCore.Qt.Checked)

            self.listWidgetDescriptors.addItem(item)
            self.descriptors.append(descriptorname)

        pass

    def _selectAll(self):
        match self.sender().objectName():
            case 'toolButtonAllSamples':
                widget: QtWidgets.QListWidget = self.listWidgetSamples
            case 'toolButtonAllDescriptors':
                widget: QtWidgets.QListWidget = self.listWidgetDescriptors
            case _:
                return

        for i in range(widget.count()):
            widget.item(i).setCheckState(QtCore.Qt.Checked)

    def _selectNone(self):
        match self.sender().objectName():
            case 'toolButtonNoneSamples':
                widget: QtWidgets.QListWidget = self.listWidgetSamples
                pass
            case 'toolButtonNoneDescriptors':
                widget: QtWidgets.QListWidget = self.listWidgetDescriptors
            case _:
                return

        for i in range(widget.count()):
            widget.item(i).setCheckState(QtCore.Qt.Unchecked)

    def get_selected_samples(self) -> list[str]:
        samplenames: list[str] = []
        for i in range(
            self.listWidgetSamples.count()
        ):
            item = self.listWidgetSamples.item(i)

            if item.checkState() != QtCore.Qt.Checked:
                continue

            samplenames.append(
                item.data(
                    QtCore.Qt.DisplayRole
                )
            )

        return samplenames

    def get_selected_descriptors(self) -> list[str]:
        descriptors: list[str] = []
        for i in range(
            self.listWidgetDescriptors.count()
        ):
            item = self.listWidgetDescriptors.item(i)

            if item.checkState() != QtCore.Qt.Checked:
                continue

            descriptors.append(
                item.data(
                    QtCore.Qt.DisplayRole
                )
            )

        return descriptors

    def accept(self):
        """
        This runs when the user clicks 'Finish'
        :return:
        """
        self.sigImportParamsGiven.emit(
            FingerprintImportParams(
                sample_names = self.get_selected_samples(),
                descriptors = self.get_selected_descriptors(),
                csv_path = Path(self.field('csvPath')),
                samples_in_rows = self.field('rowsSamples'),
            )
        )
        super().accept()

def page1_is_complete(
    page_self: QtWidgets.QWizardPage
) -> bool:
    path = page_self.field("csvPath")

    return validate_csv_path(path)

def validate_csv_path(
    path: str,
) -> bool:
    """
    Confirms whether a path is a .csv file that exists, and then
    whether it is valid (i.e. contains a fully formed table)
    :param path:
    :return:
    """
    filepath = Path(path)
    if not filepath.exists() or not filepath.is_file():
        return False

    if not filepath.suffix.lower() == '.csv':
        return False

    return True

def validate_csv_integrity(
    path: str,
) -> bool:
    """
    Uses pandas to check whether a .csv file is valid
    :param path:
    :return:
    """
    try:
        df = pd.read_csv(
            path,
            index_col=0,
        )

        # Check that df has something in it
        if not ( len(df.columns) > 0 and len(df) > 0 ):
            return False

        # Check that all entries are numeric values
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) != len(df.columns):
            return False

        return True

    except Exception:
        return False