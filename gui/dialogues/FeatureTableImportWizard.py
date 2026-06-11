"""
QWizard for importing a feature table CSV.

Page 1: select a CSV file
Page 2: pick m/z, RT, and (optional) ID columns, set parameters.
        Supports regex extraction from column values and RT unit conversion.
"""
import re

from PyQt5 import QtWidgets, QtCore
import pandas as pd

from gui.resources.FeatureTableImportWizard import Ui_Wizard

from pathlib import Path
from typing import Optional

from core.cli.import_feature_table import (
    FeatureCoordinate,
    FeatureTableImportParams,
)


class FeatureTableImportWizard(
    QtWidgets.QWizard,
    Ui_Wizard,
):
    sigImportParamsGiven = QtCore.pyqtSignal(
        object,  # list[FeatureCoordinate]
        object,  # FeatureTableImportParams
    )

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self._df: Optional[pd.DataFrame] = None

        self.wizardPage1.registerField(
            "csvPath*", self.lineEditCsvPath,
        )

        self.currentIdChanged.connect(self._on_page_changed)

    def _on_page_changed(self, page_id: int):
        if page_id == 1:
            self._populate_column_combos()

    def _user_triggered_browse_btn(self):
        print(
            'hi. _user_triggered_browse_btn()'
        )
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Select feature table CSV",
            filter="CSV files (*.csv *.tsv *.txt)",
        )
        if not filepath:
            return

        self.lineEditCsvPath.setText(filepath)

    def validateCurrentPage(self):
        if self.currentPage() == self.wizardPage1:
            path = self.field("csvPath")
            if not _validate_csv_path(path):
                QtWidgets.QMessageBox.warning(
                    self, "Invalid file",
                    "Please select a valid .csv, .tsv, or .txt file.",
                )
                return False

            if not self._try_load_csv(path):
                QtWidgets.QMessageBox.warning(
                    self, "Error reading file",
                    "Could not parse the selected file as a table.",
                )
                return False

        if self.currentPage() == self.wizardPage2:
            # Validate regex patterns if provided
            for line_edit, label in [
                (self.lineEditMzRegex, "m/z"),
                (self.lineEditRtRegex, "RT"),
            ]:
                pattern = line_edit.text().strip()
                if pattern:
                    try:
                        re.compile(pattern)
                    except re.error as e:
                        QtWidgets.QMessageBox.warning(
                            self, "Invalid regex",
                            f"Invalid {label} regex pattern:\n{e}",
                        )
                        return False

        return super().validateCurrentPage()

    def _try_load_csv(self, filepath: str) -> bool:
        try:
            path = Path(filepath)
            sep = '\t' if path.suffix.lower() in ('.tsv', '.txt') else ','
            self._df = pd.read_csv(filepath, sep=sep)
            return len(self._df) > 0 and len(self._df.columns) > 0
        except Exception:
            self._df = None
            return False

    def _populate_column_combos(self):
        if self._df is None:
            return

        columns = list(self._df.columns)

        for combo in (self.comboMzColumn, self.comboRtColumn, self.comboIdColumn):
            combo.clear()

        self.comboIdColumn.addItem("(none)")

        for col in columns:
            self.comboMzColumn.addItem(col)
            self.comboRtColumn.addItem(col)
            self.comboIdColumn.addItem(col)

        # Auto-select likely columns
        for i, col in enumerate(columns):
            col_lower = col.lower().strip()
            if col_lower in ('mz', 'm/z', 'mz_mean', 'mz_avg'):
                self.comboMzColumn.setCurrentIndex(i)
            if col_lower in ('rt', 'rt_mean', 'rt_avg', 'retention_time'):
                self.comboRtColumn.setCurrentIndex(i)

    def accept(self):
        if self._df is None:
            return

        mz_col = self.comboMzColumn.currentText()
        rt_col = self.comboRtColumn.currentText()
        id_col = self.comboIdColumn.currentText()
        mz_regex = self.lineEditMzRegex.text().strip() or None
        rt_regex = self.lineEditRtRegex.text().strip() or None
        rt_in_minutes = self.comboRtUnit.currentIndex() == 1

        features: list[FeatureCoordinate] = []
        for _, row in self._df.iterrows():
            try:
                mz = _extract_numeric(row[mz_col], mz_regex)
                rt = _extract_numeric(row[rt_col], rt_regex)
            except (ValueError, TypeError):
                continue

            if mz is None or rt is None:
                continue

            if rt_in_minutes:
                rt *= 60.0

            analyte_id = ""
            if id_col != "(none)":
                analyte_id = str(row[id_col])

            features.append(FeatureCoordinate(
                mz=mz, rt=rt, analyte_id=analyte_id,
            ))

        if not features:
            QtWidgets.QMessageBox.warning(
                self, "No features",
                "Could not parse any features from the selected columns.",
            )
            return

        params = FeatureTableImportParams(
            rt_window=self.spinRtWindow.value(),
            mz_window=self.spinMzWindow.value(),
            pregroup=self.checkPreGroup.isChecked(),
        )

        self.sigImportParamsGiven.emit(features, params)
        super().accept()


def _extract_numeric(
    value,
    regex_pattern: Optional[str],
) -> Optional[float]:
    """
    Extract a numeric value from a cell.

    If regex_pattern is provided, apply it to the string
    representation of the value and use the first capture
    group (or the whole match if no groups). Otherwise,
    convert the value directly to float.
    """
    if regex_pattern is None:
        return float(value)

    text = str(value)
    match = re.search(regex_pattern, text)
    if match is None:
        return None

    # Use first capture group if present, else whole match
    extracted = match.group(1) if match.lastindex else match.group(0)
    return float(extracted)


def _validate_csv_path(path: str) -> bool:
    filepath = Path(path)
    if not filepath.exists() or not filepath.is_file():
        return False
    if filepath.suffix.lower() not in ('.csv', '.tsv', '.txt'):
        return False
    return True
