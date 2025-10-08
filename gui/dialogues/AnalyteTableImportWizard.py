"""
A QWizard for importing an analyte table and a metadata table

Allows the user to choose which fields to import, and
define which column contains sample names
"""
from PyQt5 import QtWidgets, QtCore
import pandas as pd
import numpy as np

from gui.resources.AnalyteTableImportWizardWindow import Ui_Wizard

from pathlib import Path
from typing import Literal, Optional


class AnalyteTableImportWizard(
    QtWidgets.QWizard,
    Ui_Wizard,
):
    sigImportParamsGiven = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Page 1
        self.wizardPage1.registerField(
            "csvAnalyteTablePath*", self.lineAnalyteTablePath,
        )

        # Page 3
        self.wizardPage3.registerField(
            "csvMetadataTablePath*", self.lineMetadataTablePath,
        )

        self.currentIdChanged.connect(
            self._on_page_changed
        )

        # Monkey-patch isComplete method
        self.wizardPage1.isComplete = lambda: page_is_complete(
            page_self=self.wizardPage1,
            page_num=1,
        )

        self.wizardPage3.isComplete = lambda: page_is_complete(
            page_self=self.wizardPage3,
            page_num=3,
        )

    def validateCurrentPage(self):
        """
        Overrides wizard method - called when user tries to advance
        """
        match self.currentPage():
            case self.wizardPage1:
                path = self.field("csvAnalyteTablePath")
                if not validate_csv_integrity(path):
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Invalid .csv file",
                        "The selected file is not a valid .csv, or cannot be read. \n"
                        "All values except headers must be numeric, and there must be"
                        "columns labelled 'm/z' and 'rt'.",
                    )
                    return False

            case self.wizardPage2:
                analyte_id_column = self.get_selected_analyte_id_column()
                if not analyte_id_column:
                    return False

            case self.wizardPage3:
                path = self.field("csvMetadataTablePath")
                return True

            case self.wizardPage4:
                analyte_id_column = self.get_selected_analyte_id_column_in_metadata_table()
                if not analyte_id_column:
                    return False

        return super().validateCurrentPage()

    def _on_page_changed(
        self,
        page_id: int,
    ):
        """
        Handles population of page widgets
        """
        print(f"page changed. page_id: {page_id}")
        match page_id:
            case 1:
                self.populateListWidgets(
                    list_widget_type='analytes'
                )
            case 3:
                 self.populateListWidgets(
                    list_widget_type='metadata'
                )

    def _user_triggered_browse_btn(self):
        sender = self.sender().objectName()
        opts = {
            'pushAnalyteTablePath':
                {
                    'file': 'analyte',
                    'lineEdit': self.lineAnalyteTablePath,
                },
           'pushMetadataTablePath':
                {
                    'file': 'metadata',
                    'lineEdit': self.lineMetadataTablePath,
                },
        }[sender]

        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption=f"Select .csv file containing {opts['file']} table",
            filter="*.csv"
        )

        if not file:
            return

        opts['lineEdit'].setText(file)

    def populateListWidgets(
        self,
        list_widget_type: Literal['analytes', 'metadata'],
    ):
        widgets = {
            'analytes': {
                'path': self.field("csvAnalyteTablePath"),
                'columns' : self.listWidgetAnalyteTableColumns,
                'descriptors': self.listWidgetAnalyteTableSamples
            },
             'metadata': {
                'path': self.field("csvMetadataTablePath"),
                'columns' : self.listWidgetMetadataTableColumns,
                'descriptors': self.listWidgetFields,
            },
        }[list_widget_type]

        df = pd.read_csv(
            widgets['path'],
        )

        col_listwidget: QtWidgets.QListWidget = widgets['columns']

        col_listwidget.clear()
        for column in df.columns:
            if column == 'm/z' or column == 'rt':
                continue

            item = QtWidgets.QListWidgetItem(
                str(column)
            )
            col_listwidget.addItem(item)

        desc_listwidget: QtWidgets.QListWidget = widgets['descriptors']
        for field in df.columns:
            if field == 'm/z' or field == 'rt':
                continue
            item = QtWidgets.QListWidgetItem(
                str(field)
            )
            item.setFlags(
                item.flags() | QtCore.Qt.ItemIsUserCheckable
            )
            item.setCheckState(QtCore.Qt.Checked)

            desc_listwidget.addItem(item)


    def _selectAll(self):
        match self.sender().objectName():
            case 'toolButtonAllSamples':
                widget: QtWidgets.QListWidget = self.listWidgetAnalyteTableSamples
            case 'toolButtonAllFields':
                widget: QtWidgets.QListWidget = self.listWidgetFields
            case _:
                return

        for i in range(widget.count()):
            widget.item(i).setCheckState(QtCore.Qt.Checked)

    def _selectNone(self):
        match self.sender().objectName():
            case 'toolButtonNoneSamples':
                widget: QtWidgets.QListWidget = self.listWidgetAnalyteTableSamples
            case 'toolButtonNoneFields':
                widget: QtWidgets.QListWidget = self.listWidgetFields
            case _:
                return

        for i in range(widget.count()):
            widget.item(i).setCheckState(QtCore.Qt.Unchecked)

    def get_selected_analyte_id_column(self) -> Optional[str]:
        if not self.listWidgetAnalyteTableColumns.currentItem():
            return None

        return self.listWidgetAnalyteTableColumns.currentItem().data(
            QtCore.Qt.DisplayRole
        )

    def get_selected_analyte_id_column_in_metadata_table(self) -> Optional[str]:
        if not self.listWidgetMetadataTableColumns.currentItem():
            return None

        return self.listWidgetMetadataTableColumns.currentItem().data(
            QtCore.Qt.DisplayRole
        )

    def get_selected_samples(self) -> list[str]:
        samples: list[str] = []
        for i in range(
            self.listWidgetAnalyteTableSamples.count()
        ):
            item = self.listWidgetAnalyteTableSamples.item(i)

            if item.checkState() != QtCore.Qt.Checked:
                continue

            samples.append(
                item.data(
                    QtCore.Qt.DisplayRole
                )
            )

        return samples

    def get_selected_fields(self) -> list[str]:
        fields: list[str] = []
        for i in range(
            self.listWidgetFields.count()
        ):
            item = self.listWidgetFields.item(i)

            if item.checkState() != QtCore.Qt.Checked:
                continue

            fields.append(
                item.data(
                    QtCore.Qt.DisplayRole
                )
            )

        return fields

    def accept(self):
        """
        This runs when the user clicks 'Finish'
        :return:
        """
        self.sigImportParamsGiven.emit(
            {
                'analyte_table_csv_filepath': self.field('csvAnalyteTablePath'),
                'analyte_id_column': self.get_selected_analyte_id_column(),
                'sample_columns': self.get_selected_samples(),
                'metadata_table_csv_filepath': self.field('csvMetadataTablePath'),
                'metadata_analyte_id_column': self.get_selected_analyte_id_column_in_metadata_table(),
                'field_columns': self.get_selected_fields(),
            }
        )
        super().accept()

def page_is_complete(
    page_self: QtWidgets.QWizardPage,
    page_num: int,
) -> bool:
    """Called when user clicks 'next'"""
    match page_num:
        case 1:
            path = page_self.field("csvAnalyteTablePath")
        case 3:
            path = page_self.field("csvMetadataTablePath")
        case _:
            raise ValueError(
                f"Invalid page_num: {page_num}"
            )

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

        # Check that m/z and rt are present in columns
        for _ in ['rt', 'm/z']:
            if _ not in df.columns:
                return False

        return True

    except Exception:
        return False