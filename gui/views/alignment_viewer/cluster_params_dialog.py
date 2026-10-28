"""
Non-modal dialog for the Alignment Viewer's analyte clustering parameters.

One page per method (the method combo):
 - MS2DeepScore (the default) / Modified cosine: compute a matrix
   (`ClusterParams`, core/cli/cluster_analytes.py). Applied cosine
   parameters (and the linkage) are saved to the `[analyte_clustering]`
   config section; the method isn't, so each session starts on
   MS2DeepScore. MainController reuses a registered matrix computed the same
   way, else computes and registers one.
 - Imported matrix: a registered imported SimilarityMatrix sharing analytes
   with the current alignment (via "Import…", from a matrix scored
   externally on the exported consensus .mgf), with its editable value
   transform (`ImportParams`).

Below the pages, "Scores" shows the matrix the page resolves to (once
computed): its score histogram, the import report and the score floor,
which can be typed or dragged on the histogram. Applying stores the floor
(and an imported matrix's transform) as the matrix's `default_transform`.

Edits aren't live: clustering is recomputed on "Apply".
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
    scoring_name,
    similarity_provenance,
)
from core.cli.import_similarity import read_similarity_csv, transform
from core.data_structs.similarity_matrix import (
    IMPORTED, ImportParams, SimilarityMatrix, initial_transform,
)
from core.utils.config import load_config, save_config
from gui.resources.ClusterParamsDialog import Ui_Dialog

if TYPE_CHECKING:
    from core.data_structs.alignment import EnsembleAlignment
    from gui.views.alignment_viewer.data_source import AlignmentViewerDataSource


LINKAGE_METHODS = ('average', 'complete', 'single', 'weighted')
# Combo entries, in stackMethod page order
METHODS = (
    ("MS2DeepScore", MS2DEEPSCORE),
    ("Modified cosine", MODIFIED_COSINE),
    ("Imported matrix", IMPORTED),
)

class ClusterParamsDialog(QtWidgets.QDialog, Ui_Dialog):
    # ClusterParams (compute) | SimilarityMatrix (imported), linkage method
    sigApply = QtCore.pyqtSignal(object, str)

    def __init__(
        self,
        data_source: 'AlignmentViewerDataSource',
        source: 'ClusterParams | int',
        linkage_method: str,
        parent=None,
    ):
        """:param source: the viewer's current source (params, or a matrix uuid)."""
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Analyte Clustering")

        self.data_source = data_source
        self._alignment: Optional['EnsembleAlignment'] = None
        # The matrix whose transform the widgets were last loaded from
        self._loaded: Optional[SimilarityMatrix] = None

        for text, method in METHODS:
            self.comboMethod.addItem(text, method)
        self.comboLinkage.addItems(LINKAGE_METHODS)
        self.btnImport.setToolTip(
            "Import a square similarity/distance matrix (CSV/TSV) whose "
            "header row and first column hold the MZKIT_ANALYTE ids from "
            "the exported alignment .mgf file"
        )
        self.btnDelete.setToolTip("Delete the selected imported matrix")
        self.spinFloor.setToolTip(
            "Scores below this count as unrelated (0). Drag the red line "
            "on the histogram to set it."
        )

        self._setup_histogram()
        self._setup_buttons()

        # Debounce the (O(n^2)) preview while spinboxes / the line are dragged
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._update_preview)

        self.comboMethod.currentIndexChanged.connect(self._on_method_changed)
        self.comboMatrix.currentIndexChanged.connect(self._sync_matrix)
        self.btnImport.clicked.connect(self._import_matrix)
        self.btnDelete.clicked.connect(self._delete_matrix)
        self.checkDistance.toggled.connect(self._update_distance_tooltips)
        self.checkBidirectional.toggled.connect(
            self.spinBidirThreshold.setEnabled
        )
        # Cosine params pick out a different computed matrix
        for spin in (self.spinTolerance, self.spinMzPower,
                     self.spinIntsyPower, self.spinMinMatches):
            spin.valueChanged.connect(self._sync_matrix)
        for spin in (self.spinClipMin, self.spinClipMax,
                     self.spinBidirThreshold, self.spinFloor):
            spin.valueChanged.connect(self._preview_timer.start)
        for check in (self.checkDistance, self.checkBidirectional):
            check.toggled.connect(self._preview_timer.start)

        params = source if isinstance(source, ClusterParams) else ClusterParams()
        self.set_params(params)
        self._populate_matrices()
        if not isinstance(source, ClusterParams):
            self._set_method(IMPORTED)
            self.comboMatrix.setCurrentIndex(
                max(self.comboMatrix.findData(source), 0)
            )
        idx = self.comboLinkage.findText(linkage_method)
        self.comboLinkage.setCurrentIndex(max(idx, 0))
        self.set_import_params(initial_transform(IMPORTED))
        self._update_distance_tooltips(False)
        self._on_method_changed()

        data_source.subscribe_to_changes(
            self._populate_matrices, self._populate_matrices,
            change_type='Matrix',
        )

    def _setup_histogram(self):
        self.histogram = pg.PlotWidget(background=None)
        self.histogram.setMenuEnabled(False)
        self.histogram.setMouseEnabled(False, False)
        self.histogram.hideButtons()
        self.histogram.getPlotItem().hideAxis('left')
        self.histogram.setXRange(0, 1, padding=0.02)
        self._bars = pg.BarGraphItem(x=[], height=[], width=0.05)
        self._floor_line = pg.InfiniteLine(
            angle=90, movable=True, bounds=(0.0, 1.0),
            pen=pg.mkPen('r', width=2), hoverPen=pg.mkPen('r', width=4),
        )
        self._floor_line.setCursor(QtCore.Qt.SizeHorCursor)
        self.histogram.addItem(self._bars)
        self.histogram.addItem(self._floor_line)
        self._floor_line.sigPositionChanged.connect(
            lambda line: self.spinFloor.setValue(line.value())
        )
        self.spinFloor.valueChanged.connect(self._floor_line.setValue)
        self.histogram.scene().sigMouseClicked.connect(self._on_histogram_clicked)
        layout = QtWidgets.QVBoxLayout(self.histogramPlaceholder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.histogram)

    def _setup_buttons(self):
        box = QtWidgets.QDialogButtonBox
        self.buttonBox.button(box.Apply).clicked.connect(self._apply)
        self.buttonBox.button(box.Apply).setDefault(True)
        self.buttonBox.button(box.RestoreDefaults).clicked.connect(
            self._restore_defaults
        )

    # -- state --------------------------------------------------------------

    def set_alignment(self, alignment: Optional['EnsembleAlignment']):
        """The alignment the preview / report is computed against."""
        self._alignment = alignment
        self._populate_matrices()
        self._sync_matrix()

    def _method(self) -> str:
        return self.comboMethod.currentData()

    def _set_method(self, method: str):
        self.comboMethod.setCurrentIndex(self.comboMethod.findData(method))

    def _populate_matrices(self, *_):
        """The registry's imported matrices sharing analytes with this alignment."""
        current = self.comboMatrix.currentData()
        with QtCore.QSignalBlocker(self.comboMatrix):
            self.comboMatrix.clear()
            analytes = self._alignment.analytes if self._alignment else []
            for matrix in self.data_source.matrices_for_analytes(
                a.uuid for a in analytes
            ):
                if not matrix.is_imported:
                    continue
                self.comboMatrix.addItem(matrix.name, matrix.uuid)
                self.comboMatrix.setItemData(
                    self.comboMatrix.count() - 1,
                    _provenance_text(matrix), QtCore.Qt.ToolTipRole,
                )
            self.comboMatrix.setCurrentIndex(
                max(self.comboMatrix.findData(current), 0)
            )
        self._sync_matrix()   # a computed matrix may have landed / gone

    def set_params(self, params: ClusterParams):
        self.spinTolerance.setValue(params.tolerance)
        self.spinMzPower.setValue(params.mz_power)
        self.spinIntsyPower.setValue(params.intensity_power)
        self.spinMinMatches.setValue(params.min_matched_peaks)
        if self._method() != IMPORTED:
            self._set_method(params.scoring)

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
            scoring=(MODIFIED_COSINE if self._method() == MODIFIED_COSINE
                     else MS2DEEPSCORE),
        )

    def _transform(self, matrix: SimilarityMatrix) -> ImportParams:
        """The transform the widgets describe for `matrix`."""
        if not matrix.is_imported:   # computed scores: only the floor is editable
            return matrix.default_transform._replace(floor=self.spinFloor.value())
        return ImportParams(
            is_distance=self.checkDistance.isChecked(),
            clip_min=self.spinClipMin.value(),
            clip_max=self.spinClipMax.value(),
            bidirectional=self.checkBidirectional.isChecked(),
            bidirectional_threshold=self.spinBidirThreshold.value(),
            floor=self.spinFloor.value(),
        )

    def _current_matrix(self) -> Optional[SimilarityMatrix]:
        """The matrix the current page resolves to (None: not computed / none imported)."""
        if self._method() == IMPORTED:
            data = self.comboMatrix.currentData()
            return None if data is None else self.data_source.get_matrix(data)
        if self._alignment is None:
            return None
        return self.data_source.find_matrix(
            similarity_provenance(self.params()), self._alignment,
        )

    # -- slots --------------------------------------------------------------

    def _on_method_changed(self, *_):
        self.stackMethod.setCurrentIndex(self.comboMethod.currentIndex())
        self._sync_matrix()

    def _sync_matrix(self, *_):
        """Load the current matrix's transform when it changes, and preview it."""
        matrix = self._current_matrix()
        self.btnDelete.setEnabled(matrix is not None)
        self.groupScoreFilter.setEnabled(matrix is not None)
        if matrix is not None and matrix is not self._loaded:
            if matrix.is_imported:
                self.set_import_params(matrix.default_transform)
            else:
                self.spinFloor.setValue(matrix.default_transform.floor)
        self._loaded = matrix
        self._update_preview()

    def _on_histogram_clicked(self, event):
        if not self.groupScoreFilter.isEnabled():
            return
        view_box = self.histogram.getPlotItem().getViewBox()
        x = view_box.mapSceneToView(event.scenePos()).x()
        self.spinFloor.setValue(min(max(x, 0.0), 1.0))

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
        self.comboMatrix.setCurrentIndex(self.comboMatrix.findData(matrix.uuid))

    def _delete_matrix(self):
        matrix = self._current_matrix()
        if matrix is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Delete matrix",
            f"Delete the similarity matrix '{matrix.name}'? Clusterings by it "
            "are dropped.",
        )
        if answer == QtWidgets.QMessageBox.Yes:
            self.data_source.remove_matrix(matrix.uuid)   # repopulates the combo

    def _update_preview(self):
        """Matrix info, import report and histogram for the current matrix."""
        matrix = self._current_matrix()
        if matrix is None:
            self.labelReport.setText("")
            self._set_histogram(None, None)
            return

        try:
            report = transform(
                matrix, self._alignment, self._transform(matrix),
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

    def _restore_defaults(self):
        if self._method() == IMPORTED:
            self.set_import_params(initial_transform(IMPORTED))
        else:
            self.set_params(ClusterParams(scoring=self._method()))
            self.spinFloor.setValue(initial_transform(self._method()).floor)

    def _apply(self):
        matrix = self._current_matrix()
        if matrix is None and self._method() == IMPORTED:
            QtWidgets.QMessageBox.warning(
                self, "No matrix", "Import a similarity matrix first.",
            )
            return
        if matrix is not None:
            transform_params = self._transform(matrix)
            if transform_params.clip_max <= transform_params.clip_min:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid range", "Clip max must exceed clip min.",
                )
                return

        params = self.params()
        linkage_method = self.comboLinkage.currentText()
        config = load_config()
        cluster_params_to_config(config, params, linkage_method)
        save_config(config, [CLUSTERING_SECTION])

        if matrix is not None:
            # The matrix keeps the transform last clustered with (saved in the .mzk)
            matrix.default_transform = transform_params
        if matrix is not None and matrix.is_imported:
            self.sigApply.emit(matrix, linkage_method)
        else:
            self.sigApply.emit(params, linkage_method)


def _provenance_text(matrix: SimilarityMatrix) -> str:
    provenance = matrix.provenance
    if matrix.is_imported:
        return f"Imported from {provenance.get('path', '?')}"
    params = provenance.get('params') or {}
    return "Computed" + (
        " (" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")"
        if params else ""
    )
