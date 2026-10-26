"""
Test tool: predict an MS/MS spectrum from SMILES with GLACIER (core/utils/glacier.py).
Runs on the GUI thread.
"""
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtSvg import QSvgWidget

from core.utils import glacier
from core.utils.array_types import to_spec_arr
from core.utils.render_molecule import smiles_to_svg
from gui.widgets.MSPlotWidget import MSPlotWidget


def merge_peaks(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse GLACIER's raw rows to unique m/z (4 dp, max intensity), base peak = 1.
    Also returns, per peak, the index of the raw row that gave it that intensity.
    """
    rows = np.flatnonzero(spec[:, 1] > 0)
    mz, inv = np.unique(np.round(spec[rows, 0], 4), return_inverse=True)
    order = np.lexsort((-spec[rows, 1], inv))  # by peak, most intense row first
    src = rows[order[np.r_[True, inv[order][1:] != inv[order][:-1]]]]
    intsy = spec[src, 1]
    return mz, intsy / intsy.max() if len(intsy) else intsy, src


class GlacierDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GLACIER MS/MS prediction")
        self.resize(900, 550)

        self.smiles_edit = QtWidgets.QLineEdit()
        self.smiles_edit.setPlaceholderText("SMILES")

        self.ce_spin = QtWidgets.QDoubleSpinBox()
        self.ce_spin.setRange(-1, 300)
        self.ce_spin.setSpecialValueText("Unknown")  # shown at the minimum; predicted as NaN
        self.ce_spin.setSuffix(" eV")
        self.ce_spin.setValue(30)

        self.adduct_combo = QtWidgets.QComboBox()
        self.instrument_combo = QtWidgets.QComboBox()
        if glacier.model_available():
            meta = glacier.load_model().meta
            # ms-pred lists every adduct twice ('[M+H]+' and '[M+H]1+')
            self.adduct_combo.addItems([a for a in meta["ion2mass"] if "]1" not in a])
            self.instrument_combo.addItems(list(meta["instrument2onehot_pos"]))

        self.predict_btn = QtWidgets.QPushButton("Predict MS2")
        self.predict_btn.clicked.connect(self.predict)
        self.status_label = QtWidgets.QLabel()

        form = QtWidgets.QFormLayout()
        form.addRow("SMILES", self.smiles_edit)
        form.addRow("Collision energy", self.ce_spin)
        form.addRow("Adduct", self.adduct_combo)
        form.addRow("Instrument", self.instrument_combo)
        form.addRow(self.predict_btn, self.status_label)

        self.plot = MSPlotWidget()
        self.plot.sigMSSignalClicked.connect(self.show_fragment)
        self.structure = QSvgWidget()
        self.structure.setFixedHeight(250)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.plot, stretch=1)
        layout.addWidget(self.structure)

        # Set by predict(): parent SMILES and, per plotted peak, its fragment's atom mask
        self.smiles = ""
        self.peak_frags = np.zeros((0, 0), bool)

        if not glacier.model_available():
            self.predict_btn.setEnabled(False)
            self.status_label.setText(glacier.missing_model_message())

    def predict(self):
        smiles = self.smiles_edit.text().strip()
        ce = np.nan if self.ce_spin.value() == self.ce_spin.minimum() else self.ce_spin.value()
        try:
            pred = glacier.load_model().predict(
                [smiles], np.array([ce]), [self.adduct_combo.currentText()],
                [self.instrument_combo.currentText()],
            )[0]
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "GLACIER", str(e))
            return
        mz, intsy, src = merge_peaks(pred["spec"])
        self.smiles, self.peak_frags = smiles, pred["frag"][src]
        self.show_structure()
        self.plot.setSpectrumArray(to_spec_arr(mz, intsy))
        if len(mz):
            self.plot.pi.vb.setRange(xRange=(0, mz.max() * 1.1), yRange=(0, 1.05))
        self.status_label.setText(f"{len(mz)} peaks")

    def show_fragment(self, signal: tuple[int, float]):
        """Clicked peak (spec_idx, mz): highlight its (most intense) fragment in the parent."""
        self.show_structure(np.flatnonzero(self.peak_frags[signal[0]]).tolist())

    def show_structure(self, highlight_atoms: list[int] | None = None):
        svg, _ = smiles_to_svg(self.smiles, highlight_atoms)
        self.structure.load(QtCore.QByteArray(svg.encode()))
        self.structure.renderer().setAspectRatioMode(QtCore.Qt.KeepAspectRatio)
