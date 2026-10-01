"""
Non-modal dialog for editing the feature-map's RenderParams. Every edit is
applied live (via sigParamsChanged) so the effect is visible immediately.
"""
from dataclasses import fields

from PyQt5 import QtCore, QtWidgets

from gui.widgets.alignment_plot.params import RenderParams


# field name -> (label, minimum, maximum, decimals, step); ints ignore decimals
_FIELD_SPECS: dict[str, tuple[str, float, float, int, float]] = {
    'intsy_floor':       ("Intensity floor",        1.0, 1e12, 0, 1e4),
    'intsy_ceiling':     ("Intensity ceiling",      1.0, 1e12, 0, 1e6),
    'width_min':         ("Strip width min (s)",    0.0, 600.0, 2, 0.5),
    'width_max':         ("Strip width max (s)",    0.0, 600.0, 2, 1.0),
    'height_min':        ("Strip height min (lane)", 0.0, 1.0, 3, 0.01),
    'height_max':        ("Strip height max (lane)", 0.0, 1.0, 3, 0.01),
    'stagger_count':     ("Stagger rows",           1, 50, 0, 1),
    'stagger_step':      ("Stagger step (lane)",    0.0, 1.0, 3, 0.01),
    'line_opacity_min':  ("Line opacity min",       0, 255, 0, 5),
    'line_opacity_max':  ("Line opacity max",       0, 255, 0, 5),
    'cluster_width_scale': ("Cluster-mode width scale", 0.0, 1.0, 4, 0.005),
}


class RenderParamsDialog(QtWidgets.QDialog):
    sigParamsChanged = QtCore.pyqtSignal(object)  # RenderParams

    def __init__(self, params: RenderParams, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Alignment Display Settings")
        self._params = params.copy()
        self._editors: dict[str, QtWidgets.QAbstractSpinBox] = {}

        form = QtWidgets.QFormLayout()
        for f in fields(RenderParams):
            label, lo, hi, decimals, step = _FIELD_SPECS[f.name]
            if f.type in (int, 'int'):
                editor = QtWidgets.QSpinBox()
                editor.setRange(int(lo), int(hi))
                editor.setSingleStep(int(step))
            else:
                editor = QtWidgets.QDoubleSpinBox()
                editor.setRange(lo, hi)
                editor.setDecimals(decimals)
                editor.setSingleStep(step)
            editor.setKeyboardTracking(False)  # apply on commit, not per keystroke
            editor.setValue(getattr(self._params, f.name))
            editor.valueChanged.connect(
                lambda value, name=f.name: self._on_value_changed(name, value)
            )
            self._editors[f.name] = editor
            form.addRow(label, editor)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.RestoreDefaults
            | QtWidgets.QDialogButtonBox.Close
        )
        buttons.rejected.connect(self.close)
        buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(
            self._restore_defaults
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _on_value_changed(self, name: str, value):
        setattr(self._params, name, value)
        self.sigParamsChanged.emit(self._params.copy())

    def _restore_defaults(self):
        self._params = RenderParams()
        for name, editor in self._editors.items():
            with QtCore.QSignalBlocker(editor):
                editor.setValue(getattr(self._params, name))
        self.sigParamsChanged.emit(self._params.copy())
