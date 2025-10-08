"""
A QWizard for linking activity fingerprints, injections, and feature tables

For now, links them based on containing the fingerprint samplename.
Will implement a customizable regex-based system later
"""

from PyQt5 import QtWidgets, QtCore

from gui.resources.SampleMatchingWizardWindow import Ui_Wizard
from core.utils.sample_matching import SampleMatchingParams, find_groups, Grouping

from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from core.data_structs.injection import Injection
    from core.data_structs.fingerprint import Fingerprint
    from core.data_structs.feature_table import FeatureTable


class SampleMatchingWizard(
    QtWidgets.QWizard,
    Ui_Wizard,
):
    sigSampleMatchingParams = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self._injections: Optional[list['Injection']] = None
        self._fingerprints: Optional[list['Fingerprint']] = None
        self._featuretable: Optional['FeatureTable'] = None


    def setDataToLink(
        self,
        injections: Optional[list['Injection']] = None,
        fingerprints: Optional[list['Fingerprint']] = None,
        featuretable: Optional['FeatureTable'] = None,
    ):
        self._injections = injections
        self._fingerprints = fingerprints
        self._featuretable = featuretable

        self.populateTableWidget()


    def group_by_sample_name(self) -> list[Grouping]:
        """
        Cosmetic - for previewing purposes

        Implements the same algorithm as core.utils.sample_matching,
        but doesn't actually apply anything
        :return:
        """
        groupings = find_groups(
            injections=self._injections,
            fingerprints=self._fingerprints,
            feature_table=self._featuretable,
        )

        return groupings


    def populateTableWidget(self):
        """
        Populates the TableWidget to preview how the data will be linked
        :return:
        """
        linkages = self.group_by_sample_name()

        self.tableWidget.setRowCount(len(linkages))
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setHorizontalHeaderLabels(
            [
                'Activity fingerprint',
                'Injection',
                'Feature list',
            ]
        )

        # Populate table
        for row_idx, (fp, inj, ft) in enumerate(linkages):
            fp = fp.samplename if fp is not None else ""
            self.tableWidget.setItem(
                row_idx,
                0,
                QtWidgets.QTableWidgetItem(fp)
            )

            inj = inj.filename if inj is not None else ""
            self.tableWidget.setItem(
                row_idx,
                1,
                QtWidgets.QTableWidgetItem(inj)
            )

            ft = ft if ft is not None else ""
            self.tableWidget.setItem(
                row_idx,
                2,
                QtWidgets.QTableWidgetItem(ft)
            )


    def accept(self):
        """
        This runs when user clicks 'Finish'
        :return:
        """
        self.sigSampleMatchingParams.emit(
            SampleMatchingParams()
        )
        super().accept()