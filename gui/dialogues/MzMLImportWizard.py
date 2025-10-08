from PyQt5 import QtWidgets, QtCore, QtGui

from gui.resources.MzMLImportWizard import Ui_Wizard
from core.data_structs.scan_array import ScanArrayParameters
from core.utils.config import save_config

import re
from configparser import ConfigParser
from pathlib import Path
from typing import Optional


class MzMLImportWizard(
    QtWidgets.QWizard,
    Ui_Wizard,
):
    sigImportParamsGiven = QtCore.pyqtSignal(
        list,  #list[str] (Paths)
        list,       # list[str] (Samplenames)
        object,     # ScanArrayParameters (MS1)
        object,     # ScanArrayParameters (MS2, optional)
    )

    def __init__(
        self,
        config: ConfigParser = None
    ):
        super().__init__()
        self.setupUi(self)
        self.config = config

        self.currentIdChanged.connect(
            self._on_page_changed
        )

        # State handling
        self._filepaths: list[Path] = []
        self._samplenames: list[str] = []
        self._regex_pattern: str = ""

        self.filename_model = QtGui.QStandardItemModel()
        self.listViewFilename.setModel(
            self.filename_model,
        )

        self.samplename_model = QtGui.QStandardItemModel()
        self.listViewSamplename.setModel(
            self.samplename_model
        )

        # Field registration

        # Register spinboxes
        self.setDefaultProperty(
            'QDoubleSpinBox',
           'value',
           QtWidgets.QDoubleSpinBox.valueChanged,
        )
        fields: tuple[tuple[str, any], ...] = (
            # Name,     Object
            ("ms1_tol", self.spinMS1Tolerance),
            ("ms1_gap", self.spinMS1ScanGap),
            ("ms1_intsy", self.spinMS1Intsy),
            ("ms2_tol", self.spinMS2Tolerance),
            ("ms2_gap", self.spinMS2ScanGap),
            ("ms2_intsy", self.spinMS2Intsy),
        )

        for name, obj in fields:
            self.wizardPage3.registerField(
                name, obj
            )
            obj.valueChanged.emit(
                obj.value()
            )

        self.checkBox.setChecked(True)

    def validateCurrentPage(self):
        """
        Overrides wizard method - called when user tries to advance
        """
        match self.currentPage():
            case self.wizardPage1:
                # Verify that all the paths in textbox are valid
                if not self._validate_textbox():
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Invalid .mzML file path",
                        "One of the selected files is not a valid .mzML path"
                    )
                    return False

            case self.wizardPage2:
                # Verify that all extracted samplenames are valid
                if not self._validate_sample_names():
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Invalid sample name",
                        "One of the extracted sample names is invalid "
                        "(i.e. did not match pattern, or is a duplicate)"
                    )
                    return False

        return super().validateCurrentPage()


    def _on_page_changed(
        self,
        page_id: int,
    ) -> None:
        """
        Mostly just handles population of page 2
        """
        match page_id:
            case 0:
                pass

            case 1:
                # Populate model to preview samplename extraction stuff
                self.load_file_model()

                # Pre-load previously used regex pattern if in config
                if not self.config:
                    return

                last_pattern = self.config.get(
                    section='regex',
                    option='pattern',
                    fallback='',
                )

                self.lineEditRegex.setText(last_pattern)


    # *** PAGE 1: Filepath selection ***
    def _on_click_open_file_browser(self):
        """
        Triggered when user clicks 'open file browser'.
        Appends to the textbox file paths selected by user.
        :return:
        """
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            parent=self,
            caption="Select .mzML files to import",
            filter="*.mzML"
        )

        if not files:
            return

        for file in files:
            self.plainTextEdit.appendPlainText(
                file
            )


    def _validate_textbox(self) -> bool:
        lines = self.plainTextEdit.toPlainText().splitlines()

        for line in lines:
            filepath = Path(line)
            if not filepath:
                return False

            if not filepath.exists():
                return False

            if filepath.suffix.lower() != '.mzml':
                return False

        return True


    # *** PAGE 2: Sample name extraction ***
    def load_file_model(self) -> None:
        """
        Loads the filepaths specified by user into a model
        :return:
        """
        self._filepaths: list[Path] = []
        self.filename_model.clear()

        lines = list(
            set(
                self.plainTextEdit.toPlainText().splitlines()
            )
        )  # Discards duplicate filepaths

        for line in lines:
            filepath = Path(line)
            self._filepaths.append(filepath)

            item = QtGui.QStandardItem(filepath.name)
            self.filename_model.appendRow(item)

        self._update_sample_names()


    def _update_sample_names(self) -> None:
        self.samplename_model.clear()

        regex_pattern = self.lineEditRegex.text()
        if regex_pattern:
            self._regex_pattern = regex_pattern

        for filepath in self._filepaths:
            sample_name = apply_regex(
                filepath.name, regex_pattern
            )

            # Create item w appropriate styling
            item = QtGui.QStandardItem(
                sample_name or "<No match>"
            )

            if sample_name is None:
                item.setForeground(
                    QtGui.QBrush(
                        QtGui.QColor(128, 128, 128),
                    )
                )

                item.setFont(
                    QtGui.QFont(
                        "", -1, 1, True,  # Italic
                    )
                )

            self.samplename_model.appendRow(
                item
            )


    def get_sample_names(self) -> list[str]:
        """
        Returns a list of extracted sample names
        :return:
        """
        regex_pattern = self._regex_pattern
        self._samplenames: list[str] = []

        for filepath in self._filepaths:
            sample_name = apply_regex(filepath.name, regex_pattern)
            self._samplenames.append(
                sample_name
            )

        return self._samplenames


    def _validate_sample_names(self) -> bool:
        """
        Checks whether the regex produces valid results
        for all filenames

        i.e. returns a result for each filename, and each result
        is unique
        :return:
        """
        sample_names = self.get_sample_names()

        # Check if any samples failed to match
        if None in sample_names:
            return False

        # Check for duplicates
        if len(set(sample_names)) != len(sample_names):
            return False

        return True

    # *** PAGE 3: MS Parameters ***

    def _synchronize_ms2_params(
        self,
        sync: bool,
    ):
        """
        Whether to match the spinners for MS2 with MS1
        (i.e. if the user wants to just use the same parameters for both)
        :return:
        """
        fields: tuple[
            tuple[QtWidgets.QSpinBox, QtWidgets.QSpinBox],
            ...
        ] = (
            # MS1,     MS2
            (self.spinMS1Tolerance, self.spinMS2Tolerance),
            (self.spinMS1ScanGap, self.spinMS2ScanGap),
            (self.spinMS1Intsy, self.spinMS2Intsy),
        )

        # self.spinMS1Tolerance.valueChanged.connect(
        #     self.spinMS2Tolerance.setValue
        # )

        for field_pair in fields:
            ms1_param, ms2_param = field_pair

            if sync:
                ms1_param.valueChanged.connect(
                    ms2_param.setValue
                )

            else:
                ms1_param.valueChanged.disconnect(
                    ms2_param.setValue
                )

    # *** COMPLETION ***
    def accept(self):
        ms1_params = ScanArrayParameters(
            ms_level=1,
            mz_tolerance=self.field('ms1_tol'),
            scan_gap_tolerance=self.field('ms1_gap'),
            min_intsy=self.field('ms1_intsy'),
            scan_nums=None,
        )

        ms2_params = None
        if self.groupBoxMS2.isChecked():
            ms2_params = ScanArrayParameters(
                ms_level=2,
                mz_tolerance=self.field('ms2_tol'),
                scan_gap_tolerance=self.field('ms2_gap'),
                min_intsy=self.field('ms2_intsy'),
                scan_nums=None,
            )

        filepaths = [str(x) for x in self._filepaths]
        samplenames = self._samplenames

        self.sigImportParamsGiven.emit(
            filepaths,
            samplenames,
            ms1_params,
            ms2_params,
        )

        # Save the regex pattern
        if self.config:
            if not self.config.has_section('regex'):
                self.config.add_section('regex')

            self.config.set(
                section='regex',
                option='pattern',
                value=self._regex_pattern,
            )

            save_config(self.config)

        super().accept()


def apply_regex(
    filename: str,
    pattern: str,
) -> Optional[str]:
    """
    Applies regex pattern to filename to extract samplename

    Returns None if pattern is invalid or no match found
    :param filename:
    :param pattern:
    :return:
    """
    if not pattern.strip():
        return None

    try:
        hit = re.search(
            pattern,
            filename,
        )

        if not hit:
            return None

        if hit.groups():
            # If regex has capture groups, use first one
            return hit.group(1)

        else:
            # Otherwise return full match
            return hit.group(0)

    except re.error:
        # Invalid regex pattern
        return None





