"""
Non-modal dialog for the Alignment Viewer's cluster-mode parameters
(`ClusterParams`, core/cli/cluster_analytes.py). Unlike the display
settings, edits aren't live: clustering is recomputed on "Cluster".
Applied parameters are saved to the `[analyte_clustering]` config section.
"""
from PyQt5 import QtCore, QtWidgets

from core.cli.cluster_analytes import (
    CLUSTERING_SECTION,
    ClusterParams,
    cluster_params_to_config,
)
from core.utils.config import load_config, save_config


LINKAGE_METHODS = ('average', 'complete', 'single', 'weighted')


class ClusterParamsDialog(QtWidgets.QDialog):
    sigApply = QtCore.pyqtSignal(object)  # ClusterParams

    def __init__(self, params: ClusterParams, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analyte Clustering (MS2 modified cosine)")

        self.spinTolerance = QtWidgets.QDoubleSpinBox()
        self.spinTolerance.setRange(0.0001, 1.0)
        self.spinTolerance.setDecimals(4)
        self.spinTolerance.setSingleStep(0.005)

        self.spinMzPower = QtWidgets.QDoubleSpinBox()
        self.spinMzPower.setRange(0.0, 5.0)
        self.spinMzPower.setSingleStep(0.5)

        self.spinIntsyPower = QtWidgets.QDoubleSpinBox()
        self.spinIntsyPower.setRange(0.0, 5.0)
        self.spinIntsyPower.setSingleStep(0.5)

        self.spinMinMatches = QtWidgets.QSpinBox()
        self.spinMinMatches.setRange(0, 100)

        self.comboLinkage = QtWidgets.QComboBox()
        self.comboLinkage.addItems(LINKAGE_METHODS)

        form = QtWidgets.QFormLayout()
        form.addRow("m/z tolerance", self.spinTolerance)
        form.addRow("m/z power", self.spinMzPower)
        form.addRow("Intensity power", self.spinIntsyPower)
        form.addRow("Min. matched peaks", self.spinMinMatches)
        form.addRow("Linkage", self.comboLinkage)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.RestoreDefaults
            | QtWidgets.QDialogButtonBox.Close
        )
        btn_apply = buttons.addButton(
            "Cluster", QtWidgets.QDialogButtonBox.AcceptRole,
        )
        btn_apply.clicked.connect(self._apply)
        buttons.rejected.connect(self.close)
        buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(
            lambda: self.set_params(ClusterParams())
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

        self.set_params(params)

    def set_params(self, params: ClusterParams):
        self.spinTolerance.setValue(params.tolerance)
        self.spinMzPower.setValue(params.mz_power)
        self.spinIntsyPower.setValue(params.intensity_power)
        self.spinMinMatches.setValue(params.min_matched_peaks)
        idx = self.comboLinkage.findText(params.linkage_method)
        self.comboLinkage.setCurrentIndex(max(idx, 0))

    def params(self) -> ClusterParams:
        return ClusterParams(
            tolerance=self.spinTolerance.value(),
            mz_power=self.spinMzPower.value(),
            intensity_power=self.spinIntsyPower.value(),
            min_matched_peaks=self.spinMinMatches.value(),
            linkage_method=self.comboLinkage.currentText(),
        )

    def _apply(self):
        params = self.params()
        config = load_config()
        cluster_params_to_config(config, params)
        save_config(config, [CLUSTERING_SECTION])
        self.sigApply.emit(params)
