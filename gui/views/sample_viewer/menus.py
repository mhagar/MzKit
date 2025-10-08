import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt


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


class EnsembleExtractionSettingsMenu(QtWidgets.QWidget):
    """
    A drop-down menu containing settings for ensemble extraction
    """
    sigSettingsChanged = QtCore.pyqtSignal(
        dict,  # { ms1_corr_threshold: float,
    )          #   ms2_corr_threshold: float,
               #   min_intsy: float,
               #   use_rel_intsy: bool, }

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlags(  # For pop-up style behaviour
            Qt.Popup | Qt.FramelessWindowHint
        )

        self.setMaximumSize(
            QtCore.QSize(400, 200)
        )

        layout = QtWidgets.QVBoxLayout(self)

        # *** Correlation threshold spinners ***
        layout.addWidget(
            QtWidgets.QLabel(
                "MS1 correlation threshold:"
            )
        )
        self.spinnerMS1Threshold = QtWidgets.QDoubleSpinBox()
        self.spinnerMS1Threshold.setSingleStep(0.1)
        self.spinnerMS1Threshold.setMinimum(0.01)
        self.spinnerMS1Threshold.setMaximum(0.999)
        self.spinnerMS1Threshold.setValue(0.8)
        self.spinnerMS1Threshold.valueChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.spinnerMS1Threshold
        )


        layout.addWidget(
            QtWidgets.QLabel(
                "MS2 correlation threshold:"
            )
        )
        self.spinnerMS2Threshold = QtWidgets.QDoubleSpinBox()
        self.spinnerMS2Threshold.setSingleStep(0.1)
        self.spinnerMS2Threshold.setMinimum(0.01)
        self.spinnerMS2Threshold.setMaximum(0.999)
        self.spinnerMS2Threshold.setValue(0.8)
        self.spinnerMS2Threshold.valueChanged.connect(
            self.param_changed

        )
        layout.addWidget(
            self.spinnerMS2Threshold
        )

        # *** Minimum Intensity Spinner
        layout.addWidget(
            QtWidgets.QLabel(
                "Minimum intensity:"
            )
        )
        self.spinnerMinIntsy = QtWidgets.QDoubleSpinBox()
        self.spinnerMinIntsy.setMinimum(1)
        self.spinnerMinIntsy.setMaximum(1e9)
        self.spinnerMinIntsy.setStepType(
                QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType
        )
        self.spinnerMinIntsy.setValue(2000)
        self.spinnerMinIntsy.valueChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.spinnerMinIntsy
        )

        # *** Relative intensities checkbox ***
        self.checkRelativeIntensity = QtWidgets.QCheckBox()
        self.checkRelativeIntensity.setText(
            "Use relative signal intensity"
        )
        self.checkRelativeIntensity.setChecked(True)
        self.checkRelativeIntensity.toggled.connect(
            self.param_changed
        )

        layout.addWidget(
            self.checkRelativeIntensity
        )


    def param_changed(self):
        self.sigSettingsChanged.emit(
            self.get_params()
        )


    def get_params(self) -> dict:
        """
        Returns a dict containing the parameters
        as defined in this menu
        """
        return {
            'ms1_corr_threshold': float(self.spinnerMS1Threshold.value()),
            'ms2_corr_threshold': float(self.spinnerMS2Threshold.value()),
            'min_intsy': float(self.spinnerMinIntsy.value()),
            'use_rel_intsy': self.checkRelativeIntensity.isChecked(),
        }