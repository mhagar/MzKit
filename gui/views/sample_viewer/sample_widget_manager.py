from PyQt5 import QtCore, QtWidgets

from core.utils.config import load_config
from gui.widgets.SampleWidget import SampleWidget

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from gui.views.sample_viewer.plot_stack import SampleStackView
    from gui.views.sample_viewer.model import SampleViewerItemModel
    from core.data_structs import (
        Sample, SampleUUID,
        Injection, Analyte, AnalyteID,
        FeaturePointer,
        ScanArray,
    )
    from configparser import ConfigParser

class SampleWidgetManager(QtCore.QObject):
    """
    Manages lifecycle of SampleWidget instances
    """
    def __init__(
        self,
        plot_stack: 'SampleStackView',
        container_layout: QtWidgets.QVBoxLayout,
    ):
        super().__init__()

        self.config = load_config()
        self._parent = plot_stack
        self._layout = container_layout
        self._model: Optional['SampleViewerItemModel'] = None
        self._uuid_to_widget: dict['SampleUUID', 'SampleWidget'] = {}
        self._samples_per_window: int = 1

    def set_model(
        self,
        model: 'SampleViewerItemModel',
    ):
        self._model = model

    def set_samples_per_window(
        self,
        num: int,
    ):
        """
        Adjusts widget height so that `num` widgets are shown
        per window. Does nothing if `num` is already the same as the
        current value
        """
        if num == self._samples_per_window:
            return

        self._samples_per_window = num

        hide_axes_threshold = self.config.getint(
            section='sampleviewer',
            option='hide_axis_threshold_px',
            fallback=50,
        )
        for widget in self.get_all_widgets().values():
            widget: 'SampleWidget'

            target_height = self._parent.viewport().height() // num
            widget.setFixedHeight(target_height)

            widget.chromPlotWidget.showAxes(  # Hide axes of widget is small
                target_height > hide_axes_threshold
            )

            widget.fprintPlotWidget.showAxis(
                'bottom',
                target_height > hide_axes_threshold
            )

    def create_widget(
        self,
        uuid: 'SampleUUID',
        row_idx: int,
    ) -> 'SampleWidget':
        sample = self._model.getSample(uuid)
        widget = SampleWidget()

        widget.setName(sample.name)
        widget.setSampleUuid(uuid)

        position = self._calculate_position_for_row(row_idx)
        self._layout.insertWidget(position, widget)

        self._uuid_to_widget[uuid] = widget

        return widget

    def _calculate_position_for_row(
        self,
        row_idx: int
    ) -> int:
        """
        Find where in the layout a widget at model row_idx should go
        """
        position = 0

        for i in range(row_idx):
            item = self._model.item(i)

            if not item:
                continue

            uuid = item.data(role=self._model.UuidRole)

            if uuid in self._uuid_to_widget:
                position += 1

        return position

    def remove_widget(
        self,
        uuid: 'SampleUUID',
    ):
        """
        Remove and delete a widget
        """
        if uuid not in self._uuid_to_widget:
            return

        widget = self._uuid_to_widget[uuid]
        self._layout.removeWidget(widget)
        widget.deleteLater()

        del self._uuid_to_widget[uuid]

    def remove_all_widgets(
        self,
    ):
        """
        Removes/deletes all widgets
        """
        for uuid, widget in self.get_all_widgets().items():
            self._layout.removeWidget(widget)
            widget.deleteLater()

        self._uuid_to_widget = {}

    def get_widget(
        self,
        uuid: 'SampleUUID',
    ) -> Optional['SampleWidget']:
        return self._uuid_to_widget.get(uuid)

    def get_all_widgets(self) -> dict['SampleUUID', SampleWidget]:
        return self._uuid_to_widget

