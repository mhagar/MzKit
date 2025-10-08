"""
Plot Stack Widget for displaying
samples based on the contents of an
SampleViewer's model
"""
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QStandardItem
from PyQt5.QtCore import Qt, QPointF
import pyqtgraph as pg
import numpy as np


from gui.views.sample_viewer.tools import (
    ToolType, XICMode, ToolStage, ExtractionMode,
)
from gui.widgets.ChromPlotWidget import ChromViewBox
from gui.widgets.SampleWidget import SampleWidget
from gui.utils.ms_arrays import strip_empty_values

from typing import Optional, Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from gui.views.sample_viewer.model import SampleViewerItemModel
    from core.data_structs import (
        SampleUUID,
        Fingerprint,
        Injection,
        FeaturePointer,
    )
    from gui.views.sample_viewer.tools import ToolStateListener
    

class SampleStackView(
    QtWidgets.QScrollArea,
):
    """
    Must implement ToolStateListener protocol
    """
    sigChromatogramHovered = QtCore.pyqtSignal(
        object, # UUID; Note: specifying `int` converts to 32bit (bad)
            QPointF,
    )

    sigSelectionMade = QtCore.pyqtSignal()
    sigConfigurationMade = QtCore.pyqtSignal()


    def __init__(
        self,
        parent=None,
        model=None,
    ):
        """
        :param model:
        :param parent:
        """
        super().__init__(parent)
        self.model: Optional['SampleViewerItemModel'] = None
        self._uuid_to_samplewidget: dict['SampleUUID', SampleWidget] = {}

        # Local copy of tool state
        self._tool_type: ToolType = ToolType.NONE
        self._tool_stage: ToolStage = ToolStage.IDLE
        self._xic_mode: XICMode = XICMode.NONE
        self.current_extraction_mode: ExtractionMode = ExtractionMode.NONE
        self.current_extraction_range: Optional[tuple[float, float]] = None

        self.current_ms_level: Literal[1, 2] = 1
        self._currently_hovered_uuid: Optional[int] = None

        # Container for stacked plots
        self.container_widget = QtWidgets.QWidget()
        self.container_layout = QtWidgets.QVBoxLayout(
            self.container_widget
        )
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.setWidget(self.container_widget)
        self.setWidgetResizable(True)

        if model:
            self.setModel(model)

        # For IDE type-checking:
        self: 'ToolStateListener'

    def setModel(
        self,
        model: 'SampleViewerItemModel',
    ):
        if self.model:
            self.model.disconnect()

        self.model = model

        # Connect to model signals
        self.model.itemChanged.connect(
            self.on_item_changed
        )
        self.model.rowsInserted.connect(
            self.on_rows_inserted
        )
        self.model.rowsRemoved.connect(
            self.on_rows_removed
        )
        self.model.rowsMoved.connect(
            self.on_rows_moved
        )

        # Rebuild plots
        self.rebuild_plots()

    def set_extraction_range(
        self,
        extraction_range: tuple[float, float]
    ):
        self.current_extraction_range = extraction_range
        self.update_chroms_arrays()

    def set_selected_scan(
        self,
        selected_uuid: 'SampleUUID',
        scan_num: int,
        ms_level: int,
    ):
        """
        Called when user selects a scan in the chromatogram.

        Draws a selection indicator
        """
        for uuid, sample_widget in self._uuid_to_samplewidget.items():

            if uuid != selected_uuid:
                # Clear selection indicator
                sample_widget.setSelectionIndicatorVisible(False)

                # Move on
                continue

            sample_widget = self._uuid_to_samplewidget[uuid]

            # Get rt
            scan_array = self.model.getInjection(uuid).get_scan_array(ms_level)
            rt = scan_array.rt_arr[scan_num]

            sample_widget.setSelectionIndicator(
                xpos=rt
            )
            sample_widget.setSelectionIndicatorVisible(True)


    def rebuild_plots(self):
        """
        Rebuild sample plots
         (deletes everything and starts over)
        """
        # Clear existing plots
        self.clear_plots()

        # Add new plots
        for row_idx in range(0, self.model.rowCount()):

            sample_uuid = self.model.getSampleUuidAtRow(row_idx)
            if not sample_uuid:
                continue

            self.create_sample_widget_at_row(
                sample_uuid=sample_uuid,
                row_idx=row_idx,
            )

        self.update_chroms_arrays()
        self.update_fprint_arrays()

        print("Defaulting to link_y = True; expose to user!!")
        self.link_chrom_widget_axes(
            link_x=True,
            link_y=True,
        )

    def update_chroms_arrays(
        self
    ):
        """
        Iterate over all visible chrom_widgets and update the
        displayed chromatogram
        :return:
        """
        for uuid, sample_widget in self._uuid_to_samplewidget.items():
            injection: 'Injection' = self.model.getInjection(uuid)

            if not injection:
                continue

            scan_array = injection.get_scan_array(
                ms_level=self.current_ms_level
            )
            chrom_array: np.array
            match self._xic_mode:
                case XICMode.NONE:
                    chrom_array = scan_array.get_bpc()

                case XICMode.BPC:
                    chrom_array = scan_array.get_bpc(
                        mz_range=self.current_extraction_range,
                    )

                case XICMode.XIC:
                    chrom_array = scan_array.get_xic(
                        mz_range=self.current_extraction_range,
                    )

                case _:
                    raise ValueError(
                        f"Invalid chromatogram type specified: "
                        f"{self.current_extraction_mode}"
                    )

            sample_widget.setChromArray(
                strip_empty_values(chrom_array), # Remove rt == 0
            )

    def update_chrom_highlights(
        self,
        highlights: list[tuple['SampleUUID', 'FeaturePointer']]
    ):
        """
        Given a list of (SampleUUID, FeaturePointer) tuples,
        updates the appropriate plots to display a little "highlight trace".

        These are intended to be transient
        """
        for uuid, ftr_ptr in highlights:
            sample_widget = self._uuid_to_samplewidget[uuid]

            chrom = ftr_ptr.get_chrom_array(
                scan_array=self.model.getInjection(uuid).get_scan_array(
                    ms_level=self.current_ms_level,
                )
            )

            sample_widget.addHighlight(
                chrom=chrom,
                replace=True,
            )

    def clear_chrom_highlights(
        self,
        uuid: 'SampleUUID'
    ):
        """
        Removes highlights from selected sample
        """
        sample_widget = self._uuid_to_samplewidget.get(uuid)
        if sample_widget:
            sample_widget.clearHighlights()

    def update_fprint_arrays(self):
        """
        Iterate over all visible SampleWidgets and update the
        displayed fingerprint
        :return:
        """
        for uuid, sample_widget in self._uuid_to_samplewidget.items():
            fprint: 'Fingerprint' = self.model.getFingerprint(uuid)

            if not fprint:
                continue

            sample_widget.setFprintArray(
                array=fprint.array,
                descriptors=fprint.descriptors,
            )

    def clear_plots(self):
        for uuid, sample_widget in self._uuid_to_samplewidget.items():
            sample_widget.deleteLater()

        self._uuid_to_samplewidget = {}

    def create_sample_widget_at_row(
        self,
        sample_uuid: 'SampleUUID',
        row_idx: int,
    ):
        """
        Create and add a SampleWidget for this item.
        """
        sample_widget: SampleWidget = self.create_sample_widget(
            uuid=sample_uuid
        )

        self._uuid_to_samplewidget[sample_uuid] = sample_widget

        position = self._get_layout_position_for_row(row_idx)
        self.container_layout.insertWidget(
            position,
            sample_widget,  # type: ignore
        )

    def create_sample_widget(
        self,
        uuid: 'SampleUUID',
    ) -> SampleWidget:
        """
        Given an Sample UUID, returns a SampleWidget
        """
        sample = self.model.getSample(uuid)

        sample_widget = SampleWidget()
        sample_widget.setName(sample.name)
        sample_widget.setSampleUuid(uuid)

        self.connect_sample_widget_signals(sample_widget)

        return sample_widget

    def connect_sample_widget_signals(
        self,
        sample_widget: SampleWidget,
    ):
        """
        Given a SampleWidget, connects all the appropriate Qt signals/slots
        :param sample_widget:
        """
        sample_widget.sigChromatogramHovered.connect(
            self.on_chromatogram_hover
        )

        sample_widget.sigChromatogramLeaved.connect(
            self.on_chromatogram_leave
        )

        sample_widget.sigChromatogramClicked.connect(
            self.on_chromatogram_click
        )

        sample_widget.sigFPrintHovered.connect(
            self.on_fprint_hovered
        )

    def link_chrom_widget_axes(
        self,
        link_x: bool,
        link_y: bool,
    ):
        """
        Iterates over the chrom_widgets and links their axes together
        :return:
        """
        previous_chrom_vb: Optional[ChromViewBox] = None
        previous_fprint_vb: Optional[pg.ViewBox] = None
        for uuid, sample_widget in self._uuid_to_samplewidget.items():
            if not previous_chrom_vb:
                previous_chrom_vb = sample_widget.chromPlotWidget.pi.vb
                previous_fprint_vb = sample_widget.fprintPlotWidget.pi.vb
                continue

            vb_chrom: ChromViewBox = sample_widget.chromPlotWidget.pi.vb
            vb_fprint: pg.ViewBox = sample_widget.fprintPlotWidget.pi.vb
            if link_x:
                vb_chrom.setXLink(previous_chrom_vb)
                vb_fprint.setXLink(previous_fprint_vb)

            if link_y:
                vb_chrom.setYLink(previous_chrom_vb)
                vb_fprint.setYLink(previous_fprint_vb)

    def show_feature_pointer_trace(
        self,
        feature_pointer: 'FeaturePointer',
        sample_uuid: 'SampleUUID',
    ) -> None:
        """
        Given a FeaturePointer and a SampleUUID, updates
        that sample's widget to show an overlay of the FeaturePointer
        """
        sample_widget = self._uuid_to_samplewidget.get(sample_uuid)
        if not sample_widget:
            return

        scan_array = self.model.getInjection(sample_uuid).get_scan_array(
            ms_level=self.current_ms_level
        )

        rt = feature_pointer.get_retention_times(scan_array)
        intsy = feature_pointer.get_intensity_values(scan_array)

    def on_chromatogram_hover(
        self,
        uuid: 'SampleUUID',
        pos: QPointF,
    ):
        """
        Called when user hovers over a chromPlotWidget.
        Behaves differently depending on self.current_tooltype

        :param uuid: UUID of injection that's hovered
        :param pos: Position of mouse in scene coordinates
        """

        self._currently_hovered_uuid = uuid

        match self._tool_type:
            case ToolType.NONE:
                return

            case ToolType.GETSPECTRUM:
                # Get the ChromPlotWidget being hovered?
                sample_widget: SampleWidget = self._uuid_to_samplewidget[uuid]

                # Move slide selector in the hovered chromatogram
                sample_widget.setSliderSelector(
                    xpos=pos.x()
                )
                sample_widget.setSliderSelectorVisible(True)

                # Set mouse cursor to be pointing hand
                self.setCursor(QtCore.Qt.PointingHandCursor)

                # Emit signal
                self.sigChromatogramHovered.emit(
                    uuid,
                    pos,
                )

    def on_chromatogram_click(
        self,
        uuid: int,
    ):
        """
        Called when user clicks on a chromPlotWidget.
        Behaves differently depending on self.current_tooltype
        :param uuid: UUID of injection that was clicked
        :return:
        """
        match self._tool_type:
            case ToolType.NONE:
                return

            case ToolType.GETSPECTRUM:
                # Emit information about selected spectrum
                # TODO
                self.unsetCursor()
                self.sigSelectionMade.emit()

    def on_chromatogram_leave(
        self,
        uuid: 'SampleUUID',
    ):
        """
        Called when mouse stops hovering on a ChromPlotWidget
        :param uuid:
        :return:
        """
        self._uuid_to_samplewidget[uuid].setSliderSelectorVisible(False)

        if self._currently_hovered_uuid == uuid:
            self._currently_hovered_uuid = None
            self.setCursor(QtCore.Qt.ArrowCursor)

    def on_fprint_hovered(
        self,
        hovered_inj_uuid: int,
        idx: int,
    ):
        """
        Called when user hovers over a fingerprint

        :param hovered_inj_uuid:
        :param idx:
        :return:
        """

        # Update label in *all* Fprint plots
        for uuid, sample_widget in self._uuid_to_samplewidget.items():
            fprint: 'Fingerprint' = self.model.getSample(uuid).fingerprint

            if not fprint:
                continue
                
            value = fprint.array[idx]
            descriptor = fprint.descriptors[idx]

            sample_widget.setFprintLabel(
                f"{value:.2f}\t{descriptor}"
            )

    def on_item_changed(
        self,
        item: Optional['QStandardItem'],
    ):
        if not item:
            return

        uuid: 'SampleUUID' = item.data(
            role=self.model.UuidRole
        )
        if uuid not in self._uuid_to_samplewidget:
            return

        # Toggle visibility based on check state
        self._uuid_to_samplewidget[uuid].setVisible(
            item.checkState() == Qt.Checked
        )

    def on_rows_inserted(
        self,
        parent,
        first: int,
        last: int,
    ):
        """
        Determines whether a row insertion corresponds to a new
        sample being added, or just rearranged.

        If new sample added, appends a widget for it without repopulating stack

        If a move occured, repopulates stack (expensive!)
        # TODO: optimize this some day
        """
        # Check if any of the 'new' items already exist (i.e. are moves)
        new_samples: list[tuple[int, 'SampleUUID']] = []

        for row_idx in range(first, last + 1):
            sample_uuid = self.model.getSampleUuidAtRow(row_idx)

            if not sample_uuid:
                # This means there is nothing at row_idx, i.e. sample moved
                continue

            if sample_uuid not in self._uuid_to_samplewidget:
                # This sample is new to viewer, therefore this is an INSERTION
                new_samples.append(
                    (row_idx, sample_uuid)
                )

        # No new samples found. This was a move. Repopulate stack
        if not new_samples:
            self.rebuild_plots()
            return

        # Handle insertions
        for row_idx, sample_uuid in new_samples:
            self.create_sample_widget_at_row(
                sample_uuid=sample_uuid,
                row_idx=row_idx,
            )

        self.update_chroms_arrays()
        self.update_fprint_arrays()
        self._relink_axes_if_needed()

    def on_rows_removed(
        self,
        parent,
        first: int,
        last: int,
    ):
        # Iterate deleted rows backwards to avoid idx issues
        for row_idx in range(last, first - 1, -1):
            uuid_to_remove: Optional[ 'SampleUUID' ] = None
            for uuid, widget in self._uuid_to_samplewidget.items():
                if self._widget_was_at_row(widget, row_idx):
                    uuid_to_remove = uuid
                    break

            if uuid_to_remove:
                widget = self._uuid_to_samplewidget[uuid_to_remove]
                self.container_layout.removeWidget(widget)

                widget.deleteLater()

                del self._uuid_to_samplewidget[uuid_to_remove]

        self._relink_axes_if_needed()

    def on_rows_moved(self):
        # TODO: This isn't called by QStandardItem model. Meant for proxy models
        # Get all current widgets
        widgets: list[QtWidgets.QWidget | None] = []
        for i in range(self.model.rowCount()):
            item = self.model.item(i)
            if item:
                uuid = item.data(role=self.model.UuidRole)

                if uuid in self._uuid_to_samplewidget:
                    widgets.append(
                        self._uuid_to_samplewidget[uuid]
                    )
                else:
                    widgets.append(None)  # Placeholder for missing widgets

        # Reorder the layout
        for i, widget in enumerate(widgets):
            if not widget:  # Skip missing
                continue

            self.container_layout.removeWidget(widget)
            self.container_layout.insertWidget(i, widget)

    def on_tool_type_changed(
        self,
        new_tool: ToolType,
    ) -> None:
        """
        Called when user switches tools
        """
        self._tool_type = new_tool

        match new_tool:
            case ToolType.NONE:
                # Reset spectrum selectors (hide)
                for _, sample_widget in self._uuid_to_samplewidget.items():
                    sample_widget.setSliderSelectorVisible(False)

    def on_tool_stage_changed(self, stage: ToolStage) -> None:
        pass

    def on_xic_mode_changed(self, mode: XICMode) -> None:
        self._xic_mode = mode
        self.update_chroms_arrays()
   
    def on_ms_level_changed(
        self,
        ms_level: Literal[1, 2]
    ):
        self.current_ms_level = ms_level
        self.update_chroms_arrays()

    def on_fprints_toggled(
        self,
        show_fprints: bool,
    ):
        """
        Called when user toggles fingerprints

        :param show_fprints:
        :return:
        """
        for _, sample_widget in self._uuid_to_samplewidget.items():
            sample_widget.fprintPlotWidget.setVisible(
                show_fprints
            )

    def link_colorbar_to_fprint_plots(
        self,
        colorbaritem: pg.ColorBarItem,
    ):
        for _, chrom_widget in self._uuid_to_samplewidget.items():
            colorbaritem.setImageItem(
                chrom_widget.fprintPlotWidget.ImageItem
            )


    def _get_layout_position_for_row(
        self,
        row_idx: int
    ) -> int:
        """
        Find where in the layout a widget at model row_idx should go
        """
        position = 0
        for i in range(row_idx):
            item = self.model.item(i)

            if not item:
                continue

            uuid = item.data(role=self.model.UuidRole)

            if uuid in self._uuid_to_samplewidget:
                position += 1

        return position


    def _relink_axes_if_needed(self):
        """
        Placeholder, just calls regular axes linking func for now
        """
        self.link_chrom_widget_axes(
            link_x=True,
            link_y=True,
        )


    def _widget_was_at_row(
        self,
        widget: SampleWidget,
        row: int,
    ):
        pass

    def show_scan_window_selector(
        self,
        uuid: 'SampleUUID',
        bounds: tuple[float, float],
        display_arr: Optional[np.ndarray] = None,
    ):
        """
        Shows a 'scan window selector' that can be
        used to define the duration with which a user
        would like to extract an Ensemble

        :param uuid: Sample UUID
        :param bounds: tuple(start, end) in rt
        :param display_arr: (Optional) a chromarray to draw while user selects
        """
        sample_widget: SampleWidget = self._uuid_to_samplewidget[uuid]

        sample_widget.addWindowSelector(
            bounds=bounds,
            display_arr=display_arr,
        )

    def clear_scan_window_selector(self):
        """
        Removes all the graphics generated by .show_scan_window_selector()
        """
        for uuid, sample_widget in self._uuid_to_samplewidget.items():
            sample_widget.clearWindowSelector()

    def get_selected_scan_window(
        self,
        uuid: 'SampleUUID'
    ) -> tuple[float, float]:
        """
        Returns a tuple (start, end) defining whatever region is currently
        highlighted by the scan window selector.

        If no scan window selector is present, returns (0, 0)
        """
        sample_widget = self._uuid_to_samplewidget[uuid]
        return sample_widget.getWindowSelectorBounds()




