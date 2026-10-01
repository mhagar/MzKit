import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt

from typing import Optional, TYPE_CHECKING

from core.data_structs.alignment import AlignmentParams
from core.cli.align_ensembles import (
    ALIGNMENT_SECTION,
    alignment_params_from_config,
    alignment_params_to_config,
)
from core.utils.config import save_config, load_default_config

if TYPE_CHECKING:
    from configparser import ConfigParser


class AlignmentParamsDialog(QtWidgets.QDialog):
    """
    Temporary modal dialog for entering cross-sample AlignmentParams by
    hand. This is a stopgap until a proper alignment configuration UI is
    built — see the TODO in MainController._handle_align_ensembles_request.

    Initial values come from the ``[alignment]`` config section (config is
    the single source of truth), and the Save/Reset/Restore Defaults buttons
    persist / reload that section, mirroring EnsembleExtractionDialog.
    """

    # (attribute, label, minimum, maximum, decimals, single-step)
    _FIELDS = (
        ('rt_tolerance', 'RT tolerance (s)', 0.0, 10000.0, 2, 1.0),
        ('mz_tolerance', 'm/z tolerance', 0.0, 10.0, 4, 0.001),
        ('ms1_similarity_threshold', 'MS1 similarity threshold',
         0.0, 1.0, 3, 0.05),
        ('ms2_similarity_threshold', 'MS2 similarity threshold',
         0.0, 1.0, 3, 0.05),
        ('ms1_weight', 'MS1 weight', 0.0, 1.0, 3, 0.05),
        ('ms2_weight', 'MS2 weight', 0.0, 1.0, 3, 0.05),
    )

    def __init__(
        self,
        parent=None,
        config: Optional['ConfigParser'] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Alignment Parameters")
        self.setModal(True)

        self.config = config

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self._spinboxes: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for attr, label, minimum, maximum, decimals, step in self._FIELDS:
            spinbox = QtWidgets.QDoubleSpinBox()
            spinbox.setRange(minimum, maximum)
            spinbox.setDecimals(decimals)
            spinbox.setSingleStep(step)
            self._spinboxes[attr] = spinbox
            form.addRow(label, spinbox)

        layout.addLayout(form)

        # Config persistence buttons (Save/Reset/Restore Defaults), mirroring
        # EnsembleExtractionDialog. Only shown when a config is available.
        if config is not None:
            self.btnConfigBox = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Save
                | QtWidgets.QDialogButtonBox.Reset
                | QtWidgets.QDialogButtonBox.RestoreDefaults
            )
            self.btnConfigBox.clicked.connect(self._on_config_btn_pressed)
            layout.addWidget(self.btnConfigBox)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Populate every control from [alignment] (falling back to the
        # NamedTuple's own defaults when no config is supplied).
        self._apply_params(
            alignment_params_from_config(config)
            if config is not None
            else AlignmentParams()
        )

    def _apply_params(self, params: 'AlignmentParams') -> None:
        for attr, spinbox in self._spinboxes.items():
            spinbox.setValue(getattr(params, attr))

    def params(self) -> 'AlignmentParams':
        """Return an AlignmentParams built from the current field values."""
        return AlignmentParams(**{
            attr: spinbox.value()
            for attr, spinbox in self._spinboxes.items()
        })

    def _on_config_btn_pressed(
        self,
        button: QtWidgets.QAbstractButton,
    ) -> None:
        standard_button = self.btnConfigBox.standardButton(button)
        match standard_button:
            case QtWidgets.QDialogButtonBox.StandardButton.Save:
                if self.config is not None:
                    alignment_params_to_config(self.config, self.params())
                    save_config(self.config, [ALIGNMENT_SECTION])
            case QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults:
                self._apply_params(
                    alignment_params_from_config(load_default_config())
                )
            case QtWidgets.QDialogButtonBox.StandardButton.Reset:
                if self.config is not None:
                    self._apply_params(
                        alignment_params_from_config(self.config)
                    )

    @classmethod
    def get_params(
        cls,
        parent=None,
        config: Optional['ConfigParser'] = None,
    ) -> 'AlignmentParams | None':
        """
        Show the dialog modally. Returns the entered AlignmentParams, or
        None if the user cancelled.
        """
        dialog = cls(parent=parent, config=config)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None
        return dialog.params()


class FingerprintDisplayMenu(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlags(  # For pop-up style behaviour
            Qt.Popup | Qt.FramelessWindowHint
        )

        self.setMaximumSize(
            QtCore.QSize(400, 200)
        )

        layout = QtWidgets.QVBoxLayout(self)
        # self.plot_widget = pg.PlotWidget()
        # self.plot_widget.setFixedSize(200, 300)

        self.graphics_layout_widget = pg.GraphicsLayoutWidget()

        self.colorbar = pg.ColorBarItem(
            values=(-1, 1),
            colorMap='CET-D1A',
            orientation='horizontal',
            limits=(-1.1, 1.1),
            rounding=0.1,
            width=100,
        )
        self.graphics_layout_widget.addItem(
            self.colorbar
        )

        layout.addWidget(
            self.graphics_layout_widget, # type: ignore
        )
