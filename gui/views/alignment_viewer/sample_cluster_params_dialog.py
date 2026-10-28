"""
Non-modal dialog for the Alignment Viewer's sample clustering parameters.

One page per method (the method combo; see `SampleClusterParams`,
core/cli/cluster_samples.py):
 - Sample composition: Jaccard of analyte presence, optionally over MS2
   groups cut from the analyte clustering (the Analyte Clustering dialog's
   settings; computed on demand). Saved to the `[sample_clustering]`
   config section.
 - Imported sample lineage: ranks per sample name from a CSV. Kept for the
   session only, so each session starts on composition.

Clustering is recomputed on "Apply".
"""
import csv
from pathlib import Path
from typing import Optional

from PyQt5 import QtCore, QtWidgets

from core.cli.cluster_samples import (
    COMPOSITION, LINEAGE, SAMPLE_CLUSTERING_SECTION, SampleClusterParams,
    sample_cluster_params_to_config,
)
from core.utils.config import load_config, save_config
from gui.resources.SampleClusterParamsDialog import Ui_Dialog
from gui.views.alignment_viewer.cluster_params_dialog import LINKAGE_METHODS

# Combo entries, in stackMethod page order
METHODS = (
    ("Sample composition", COMPOSITION),
    ("Imported sample lineage", LINEAGE),
)

Ranks = dict[str, tuple[str, ...]]


class SampleClusterParamsDialog(QtWidgets.QDialog, Ui_Dialog):
    # SampleClusterParams, ranks (None unless LINEAGE)
    sigApply = QtCore.pyqtSignal(object, object)

    def __init__(
        self,
        params: SampleClusterParams,
        ranks: Optional[Ranks] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Sample Clustering")

        self._ranks = ranks
        self._ranks_path: Optional[str] = None
        self._sample_names: set[str] = set()

        for text, method in METHODS:
            self.comboMethod.addItem(text, method)
        self.comboLinkage.addItems(LINKAGE_METHODS)
        self.checkMs2Grouping.setToolTip(
            "Count analytes whose MS2 dendrogram merges at or above this similarity"
            " as one, i.e. samples comprising analogues score as similar. "
        )
        self.checkMs2Grouping.toggled.connect(self.spinMs2Threshold.setEnabled)
        self.comboMethod.currentIndexChanged.connect(
            self.stackMethod.setCurrentIndex
        )
        self.btnLoadLineage.clicked.connect(self._load_lineage)

        box = QtWidgets.QDialogButtonBox
        self.buttonBox.button(box.Apply).clicked.connect(self._apply)
        self.buttonBox.button(box.Apply).setDefault(True)
        self.buttonBox.button(box.RestoreDefaults).clicked.connect(
            lambda: self.set_params(SampleClusterParams(method=self._method()))
        )

        self.set_params(params)
        self._update_lineage_labels()

    def set_sample_names(self, names: set[str]):
        """The shown alignment's sample names, for the lineage match report."""
        self._sample_names = names
        self._update_lineage_labels()

    def _method(self) -> str:
        return self.comboMethod.currentData()

    def set_params(self, params: SampleClusterParams):
        self.comboMethod.setCurrentIndex(self.comboMethod.findData(params.method))
        self.checkMs2Grouping.setChecked(params.ms2_grouping)
        self.spinMs2Threshold.setValue(params.ms2_threshold)
        self.spinMs2Threshold.setEnabled(params.ms2_grouping)
        self.comboLinkage.setCurrentIndex(
            max(self.comboLinkage.findText(params.linkage_method), 0)
        )

    def params(self) -> SampleClusterParams:
        return SampleClusterParams(
            method=self._method(),
            ms2_grouping=self.checkMs2Grouping.isChecked(),
            ms2_threshold=self.spinMs2Threshold.value(),
            linkage_method=self.comboLinkage.currentText(),
        )

    def _load_lineage(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load sample lineage", "", "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            with open(path, newline='') as f:
                rows = list(csv.reader(f))[1:]   # skip header
        except (OSError, UnicodeDecodeError, csv.Error) as e:
            QtWidgets.QMessageBox.warning(self, "Load sample lineage", str(e))
            return
        self._ranks = {row[0]: tuple(row[1:]) for row in rows if row}
        self._ranks_path = path
        self._update_lineage_labels()

    def _update_lineage_labels(self):
        if self._ranks is None:
            self.labelLineagePath.setText("<i>No lineage loaded</i>")
            self.labelLineageReport.setText("")
            return
        self.labelLineagePath.setText(
            Path(self._ranks_path).name if self._ranks_path else "Loaded lineage"
        )
        if self._ranks_path:
            self.labelLineagePath.setToolTip(self._ranks_path)
        matched = len(self._sample_names & self._ranks.keys())
        self.labelLineageReport.setText(
            f"{len(self._ranks)} rows; {matched} of {len(self._sample_names)} "
            "samples matched by name (unmatched samples are unrelated to all)."
        )

    def _apply(self):
        params = self.params()
        if params.method == LINEAGE and self._ranks is None:
            QtWidgets.QMessageBox.warning(
                self, "No lineage", "Load a sample lineage CSV first.",
            )
            return
        config = load_config()
        sample_cluster_params_to_config(config, params)
        save_config(config, [SAMPLE_CLUSTERING_SECTION])
        self.sigApply.emit(params, self._ranks if params.method == LINEAGE else None)
