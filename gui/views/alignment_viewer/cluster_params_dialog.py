"""
Non-modal dialog for the Alignment Viewer's cluster-mode parameters.

Two similarity sources:
 - Modified cosine (`ClusterParams`, core/cli/cluster_analytes.py); applied
   parameters are saved to the `[analyte_clustering]` config section.
 - An imported matrix (core/cli/import_similarity.py): "Export MS2" writes
   the consensus .mgf (Export Table) is scored externally, "Import
   matrix…" reads the scores back.
   Imported matrices are held here for the session only. The import report
   and score histogram preview the transform for the current alignment.

Unlike the display settings, edits aren't live: clustering is recomputed on
"Cluster".
"""
from typing import Optional, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from core.cli.cluster_analytes import (
    CLUSTERING_SECTION,
    ClusterParams,
    cluster_params_to_config,
)
from core.cli.import_similarity import (
    ImportParams,
    ImportedClusterSpec,
    RawMatrix,
    read_similarity_csv,
    transform,
)
from core.utils.config import load_config, save_config
from gui.resources.ClusterParamsDialog import Ui_Dialog

if TYPE_CHECKING:
    from core.data_structs.alignment import EnsembleAlignment


LINKAGE_METHODS = ('average', 'complete', 'single', 'weighted')

class ClusterParamsDialog(QtWidgets.QDialog, Ui_Dialog):
    # ClusterParams | ImportedClusterSpec, RawMatrix | None
    sigApply = QtCore.pyqtSignal(object, object)

    def __init__(self, params: ClusterParams, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Analyte Clustering")

        self._alignment: Optional['EnsembleAlignment'] = None
        self._matrices: dict[int, RawMatrix] = {}

        self.comboSource.addItem("Modified cosine", None)
        self.comboLinkage.addItems(LINKAGE_METHODS)
        self.spinTolerance.setMinimum(0.0001)
        self.spinFloor.setMaximum(1.0)
        self.btnImport.setToolTip(
            "Import a square similarity/distance matrix (CSV/TSV) whose "
            "header row and first column hold the MZKIT_ANALYTE ids from "
            "the exported consensus .mgf"
        )
        self.spinFloor.setToolTip(
            "Scores below this count as unrelated (0)."
        )
        self.labelMatrixInfo.setWordWrap(True)
        self.labelReport.setWordWrap(True)
        self.labelReport.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )

        self._setup_histogram()
        self._setup_buttons()

        # Debounce the (O(n^2)) preview while spinboxes are being dragged
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._update_preview)

        self.comboSource.currentIndexChanged.connect(self._on_source_changed)
        self.btnImport.clicked.connect(self._import_matrix)
        self.checkDistance.toggled.connect(self._update_distance_tooltips)
        self.checkBidirectional.toggled.connect(
            self.spinBidirThreshold.setEnabled
        )
        for spin in (self.spinClipMin, self.spinClipMax,
                     self.spinBidirThreshold, self.spinFloor):
            spin.valueChanged.connect(self._preview_timer.start)
        for check in (self.checkDistance, self.checkBidirectional):
            check.toggled.connect(self._preview_timer.start)

        self.set_params(params)
        self.set_import_params(ImportParams())
        self._update_distance_tooltips(False)
        self._on_source_changed()

    def _setup_histogram(self):
        self.histogram = pg.PlotWidget(background=None)
        self.histogram.setMenuEnabled(False)
        self.histogram.setMouseEnabled(False, False)
        self.histogram.hideButtons()
        self.histogram.getPlotItem().hideAxis('left')
        self.histogram.setXRange(0, 1, padding=0.02)
        self.histogram.setToolTip(
            "Pair scores after rescaling/symmetrizing; the line is the floor"
        )
        self._bars = pg.BarGraphItem(x=[], height=[], width=0.05)
        self._floor_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen('r', width=1),
        )
        self.histogram.addItem(self._bars)
        self.histogram.addItem(self._floor_line)
        layout = QtWidgets.QVBoxLayout(self.histogramPlaceholder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.histogram)

    def _setup_buttons(self):
        box = QtWidgets.QDialogButtonBox
        self.buttonBox.button(box.Apply).setText("Cluster")
        self.buttonBox.button(box.Apply).clicked.connect(self._apply)
        self.buttonBox.button(box.Cancel).setText("Close")
        self.buttonBox.button(box.RestoreDefaults).clicked.connect(
            self._restore_defaults
        )

    # -- state --------------------------------------------------------------

    def set_alignment(self, alignment: Optional['EnsembleAlignment']):
        """The alignment the preview / report is computed against."""
        self._alignment = alignment
        self._update_preview()

    def set_params(self, params: ClusterParams):
        self.spinTolerance.setValue(params.tolerance)
        self.spinMzPower.setValue(params.mz_power)
        self.spinIntsyPower.setValue(params.intensity_power)
        self.spinMinMatches.setValue(params.min_matched_peaks)
        idx = self.comboLinkage.findText(params.linkage_method)
        self.comboLinkage.setCurrentIndex(max(idx, 0))

    def set_import_params(self, params: ImportParams):
        self.checkDistance.setChecked(params.is_distance)
        self.spinClipMin.setValue(params.clip_min)
        self.spinClipMax.setValue(params.clip_max)
        self.checkBidirectional.setChecked(params.bidirectional)
        self.spinBidirThreshold.setValue(params.bidirectional_threshold)
        self.spinBidirThreshold.setEnabled(params.bidirectional)
        self.spinFloor.setValue(params.floor)

    def params(self) -> ClusterParams:
        return ClusterParams(
            tolerance=self.spinTolerance.value(),
            mz_power=self.spinMzPower.value(),
            intensity_power=self.spinIntsyPower.value(),
            min_matched_peaks=self.spinMinMatches.value(),
            linkage_method=self.comboLinkage.currentText(),
        )

    def import_params(self) -> ImportParams:
        return ImportParams(
            is_distance=self.checkDistance.isChecked(),
            clip_min=self.spinClipMin.value(),
            clip_max=self.spinClipMax.value(),
            bidirectional=self.checkBidirectional.isChecked(),
            bidirectional_threshold=self.spinBidirThreshold.value(),
            floor=self.spinFloor.value(),
        )

    def matrices(self) -> list[RawMatrix]:
        """The matrices imported this session."""
        return list(self._matrices.values())

    def _current_matrix(self) -> Optional[RawMatrix]:
        uid = self.comboSource.currentData()
        return None if uid is None else self._matrices.get(uid)

    # -- slots --------------------------------------------------------------

    def _on_source_changed(self, *_):
        self.stackSource.setCurrentIndex(
            int(self._current_matrix() is not None)
        )
        self._update_preview()

    def _update_distance_tooltips(self, is_distance: bool):
        if is_distance:
            self.spinClipMin.setToolTip("Distance of identical spectra")
            self.spinClipMax.setToolTip(
                "Distance at (and beyond) which analytes are fully unrelated"
            )
        else:
            self.spinClipMin.setToolTip(
                "Scores at or below this become 0 (unrelated)"
            )
            self.spinClipMax.setToolTip("Scores at or above this become 1")

    def _import_matrix(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Similarity Matrix", "",
            "Matrix files (*.csv *.tsv *.txt);;All files (*)",
        )
        if not path:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            matrix = read_similarity_csv(path)
        except (OSError, ValueError) as e:
            matrix, error = None, e
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if matrix is None:
            QtWidgets.QMessageBox.warning(self, "Import failed", str(error))
            return

        self._matrices[matrix.uid] = matrix
        self.comboSource.addItem(matrix.name, matrix.uid)
        self.comboSource.setCurrentIndex(self.comboSource.count() - 1)

    def _update_preview(self):
        """Matrix info, import report and histogram for the current matrix."""
        matrix = self._current_matrix()
        if matrix is None:
            return
        n = len(matrix.ids)
        info = f"<b>{matrix.name}</b> ({n}×{n})"
        if matrix.path is not None:
            info += f"<br><small>{matrix.path}</small>"
        self.labelMatrixInfo.setText(info)

        if self._alignment is None:
            self.labelReport.setText("<i>No alignment shown</i>")
            self._set_histogram(None, None)
            return
        try:
            report = transform(
                matrix, self._alignment, self.import_params(),
            ).report
        except ValueError as e:
            self.labelReport.setText(f"<span style='color:#c33'>{e}</span>")
            self._set_histogram(None, None)
            return

        text = report.format_text().replace('\n', '<br>')
        if report.n_matched == 0:
            text = (
                "<span style='color:#c33'>No matrix ids match this "
                "alignment's analytes. Was the matrix scored on an .mgf "
                "exported from this (or a parent) alignment?</span><br>"
                + text
            )
        self.labelReport.setText(text)
        self._set_histogram(report.histogram, report.bin_edges)

    def _set_histogram(self, counts, edges):
        if counts is None or not np.any(counts):
            self._bars.setOpts(x=[], height=[])
        else:
            centers = 0.5 * (edges[:-1] + edges[1:])
            # Log-ish scale: unrelated pairs usually dwarf the rest
            self._bars.setOpts(
                x=centers, height=np.log1p(counts),
                width=0.9 * (edges[1] - edges[0]),
            )
        self._floor_line.setValue(self.spinFloor.value())

    def _restore_defaults(self):
        if self.stackSource.currentIndex() == 0:
            self.set_params(ClusterParams())
        else:
            self.set_import_params(ImportParams())

    def _apply(self):
        params = self.params()
        matrix = self._current_matrix()
        import_params = self.import_params()
        if matrix is not None and import_params.clip_max <= import_params.clip_min:
            QtWidgets.QMessageBox.warning(
                self, "Invalid range", "Clip max must exceed clip min.",
            )
            return

        config = load_config()
        cluster_params_to_config(config, params)
        save_config(config, [CLUSTERING_SECTION])

        if matrix is None:
            self.sigApply.emit(params, None)
            return
        spec = ImportedClusterSpec(
            matrix_uid=matrix.uid,
            matrix_name=matrix.name,
            params=import_params,
            linkage_method=params.linkage_method,
        )
        self.sigApply.emit(spec, matrix)
