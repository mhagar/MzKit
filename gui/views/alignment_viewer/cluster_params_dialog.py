"""
Non-modal dialog for the Alignment Viewer's cluster-mode parameters.

Similarity sources (the source combo):
 - "Compute modified cosine" / "Compute MS2DeepScore" (`ClusterParams`,
   core/cli/cluster_analytes.py); applied parameters (and the linkage) are
   saved to the `[analyte_clustering]` config section. MS2DeepScore has no
   parameters, so it shows the cosine page disabled. MainController reuses a
   registered matrix computed the same way, else computes and registers one.
 - Every registered SimilarityMatrix sharing analytes with the current
   alignment (computed earlier, or imported via "Import matrix…" from a
   matrix scored externally on the exported consensus .mgf). Picking one
   clusters it without recomputing; its transform (`ImportParams`) is
   editable, previewed by the import report and score histogram.

Unlike the display settings, edits aren't live: clustering is recomputed on
"Cluster".
"""
from typing import Optional, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from core.cli.cluster_analytes import (
    CLUSTERING_SECTION,
    MODIFIED_COSINE,
    MS2DEEPSCORE,
    ClusterParams,
    cluster_params_to_config,
)
from core.cli.import_similarity import read_similarity_csv, transform
from core.data_structs.similarity_matrix import (
    ImportParams, SimilarityMatrix, initial_transform,
)
from core.utils.config import load_config, save_config
from gui.resources.ClusterParamsDialog import Ui_Dialog

if TYPE_CHECKING:
    from core.data_structs.alignment import EnsembleAlignment
    from gui.views.alignment_viewer.data_source import AlignmentViewerDataSource


LINKAGE_METHODS = ('average', 'complete', 'single', 'weighted')

class ClusterParamsDialog(QtWidgets.QDialog, Ui_Dialog):
    # ClusterParams (compute) | SimilarityMatrix (registered), linkage method
    sigApply = QtCore.pyqtSignal(object, str)

    def __init__(
        self,
        data_source: 'AlignmentViewerDataSource',
        params: ClusterParams,
        linkage_method: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Analyte Clustering")

        self.data_source = data_source
        self._alignment: Optional['EnsembleAlignment'] = None

        self._populate_sources()
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
        self.btnDelete.clicked.connect(self._delete_matrix)
        self.btnDelete.setToolTip("Delete the selected similarity matrix")
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
        idx = self.comboLinkage.findText(linkage_method)
        self.comboLinkage.setCurrentIndex(max(idx, 0))
        self.set_import_params(ImportParams())
        self._update_distance_tooltips(False)
        self._on_source_changed()

        data_source.subscribe_to_changes(
            self._populate_sources, self._populate_sources,
            change_type='Matrix',
        )

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
        self._populate_sources()
        self._update_preview()

    def _populate_sources(self, *_):
        """Compute entries + the registry's matrices for this alignment."""
        current = self.comboSource.currentData()
        with QtCore.QSignalBlocker(self.comboSource):
            self.comboSource.clear()
            self.comboSource.addItem("Compute modified cosine", MODIFIED_COSINE)
            self.comboSource.addItem("Compute MS2DeepScore", MS2DEEPSCORE)
            analytes = self._alignment.analytes if self._alignment else []
            for matrix in self.data_source.matrices_for_analytes(
                a.uuid for a in analytes
            ):
                self.comboSource.addItem(matrix.name, matrix.uuid)
                self.comboSource.setItemData(
                    self.comboSource.count() - 1,
                    _provenance_text(matrix), QtCore.Qt.ToolTipRole,
                )
            self.comboSource.setCurrentIndex(
                max(self.comboSource.findData(current), 0)
            )
        if self.comboSource.currentData() != current:
            self._on_source_changed()

    def set_params(self, params: ClusterParams):
        self.spinTolerance.setValue(params.tolerance)
        self.spinMzPower.setValue(params.mz_power)
        self.spinIntsyPower.setValue(params.intensity_power)
        self.spinMinMatches.setValue(params.min_matched_peaks)
        self.comboSource.setCurrentIndex(int(params.scoring == MS2DEEPSCORE))

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
            scoring=(MS2DEEPSCORE if self.comboSource.currentData() == MS2DEEPSCORE
                     else MODIFIED_COSINE),
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

    def _current_matrix(self) -> Optional[SimilarityMatrix]:
        data = self.comboSource.currentData()
        if data in (MODIFIED_COSINE, MS2DEEPSCORE):
            return None
        return self.data_source.get_matrix(data)

    # -- slots --------------------------------------------------------------

    def _on_source_changed(self, *_):
        matrix = self._current_matrix()
        self.stackSource.setCurrentIndex(int(matrix is not None))
        self.btnDelete.setEnabled(matrix is not None)
        if matrix is not None:
            self.set_import_params(matrix.default_transform)
        self.pageCosine.setEnabled(self.comboSource.currentData() != MS2DEEPSCORE)
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

        # Only matrices sharing analytes with the alignment are offered (and kept)
        if self._alignment is None or transform(
            matrix, self._alignment, matrix.default_transform,
        ).report.n_matched == 0:
            QtWidgets.QMessageBox.warning(
                self, "Import failed",
                "No matrix ids match the shown alignment's analytes. Was the "
                "matrix scored on an .mgf exported from this (or a parent) "
                "alignment?",
            )
            return

        self.data_source.register_matrix(matrix)   # repopulates the combo
        self.comboSource.setCurrentIndex(self.comboSource.findData(matrix.uuid))

    def _delete_matrix(self):
        matrix = self._current_matrix()
        if matrix is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Delete matrix",
            f"Delete the similarity matrix '{matrix.name}'? Clusterings by it "
            "are dropped; a computed one can be recomputed.",
        )
        if answer == QtWidgets.QMessageBox.Yes:
            self.data_source.remove_matrix(matrix.uuid)   # repopulates the combo

    def _update_preview(self):
        """Matrix info, import report and histogram for the current matrix."""
        matrix = self._current_matrix()
        if matrix is None:
            return
        n = len(matrix.ids)
        info = f"<b>{matrix.name}</b> ({n}×{n})"
        info += f"<br><small>{_provenance_text(matrix)}</small>"
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
        matrix = self._current_matrix()
        if matrix is None:
            self.set_params(ClusterParams(scoring=self.params().scoring))
        else:
            self.set_import_params(
                initial_transform(matrix.provenance.get('source'))
            )

    def _apply(self):
        params = self.params()
        matrix = self._current_matrix()
        import_params = self.import_params()
        if matrix is not None and import_params.clip_max <= import_params.clip_min:
            QtWidgets.QMessageBox.warning(
                self, "Invalid range", "Clip max must exceed clip min.",
            )
            return

        linkage_method = self.comboLinkage.currentText()
        config = load_config()
        cluster_params_to_config(config, params, linkage_method)
        save_config(config, [CLUSTERING_SECTION])

        if matrix is None:
            self.sigApply.emit(params, linkage_method)
            return
        # The matrix keeps the transform last clustered with (saved in the .mzk)
        matrix.default_transform = import_params
        self.sigApply.emit(matrix, linkage_method)


def _provenance_text(matrix: SimilarityMatrix) -> str:
    provenance = matrix.provenance
    if matrix.is_imported:
        return f"Imported from {provenance.get('path', '?')}"
    params = provenance.get('params') or {}
    return "Computed" + (
        " (" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")"
        if params else ""
    )
