"""
Plot Stack Widget for displaying
samples based on the contents of an
SampleViewer's model
"""
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QStandardItem
from PyQt5.QtCore import Qt, QPointF
import pyqtgraph as pg


from gui.views.sample_viewer.sample_widget_manager import SampleWidgetManager
from gui.views.sample_viewer.chromatogram_data_manager import ChromatogramDataManager
from gui.views.sample_viewer.ensemble_ui_manager import EnsembleUIManager
from gui.views.sample_viewer.tools import (
    ToolType, XICMode, ToolStage, ExtractionMode,
)
from gui.widgets.SampleWidget import SampleWidget

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

    # Ensemble peak interaction signals
    sigEnsemblePeakHovered = QtCore.pyqtSignal(
        object,  # SampleUUID
        object,  # EnsembleUUID
        QPointF,
    )
    sigEnsemblePeakClicked = QtCore.pyqtSignal(
        object,  # SampleUUID
        object,  # EnsembleUUID
        int,     # MouseButton
    )


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

        # Local copy of tool state
        self._tool_type: ToolType = ToolType.NONE
        self._tool_stage: ToolStage = ToolStage.IDLE

        self._currently_hovered_uuid: Optional[int] = None

        self._show_ensembles: bool = True

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

        # SampleWidgetManager
        self.sample_wdgt_mgr: SampleWidgetManager = SampleWidgetManager(
            container_layout=self.container_layout,
        )

        # ChromatogramDataManager
        self.chrom_mgr: ChromatogramDataManager = ChromatogramDataManager(
            model=self.model,
            widget_manager=self.sample_wdgt_mgr,
        )

        # EnsembleUIManager
        self.ensemble_ui_mgr: EnsembleUIManager = EnsembleUIManager(
            model=self.model,
            widget_manager=self.sample_wdgt_mgr,
        )

        # For IDE type-checking:
        # self: 'ToolStateListener'

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

        self.sample_wdgt_mgr.set_model(model)
        self.chrom_mgr.set_model(model)
        self.ensemble_ui_mgr.set_model(model)

        # Rebuild plots
        self.rebuild_plots()

    def set_extraction_range(
        self,
        extraction_range: tuple[float, float]
    ):
        self.chrom_mgr.set_extraction_range(extraction_range)

    def on_spectrum_selected(
        self,
        selected_uuid: 'SampleUUID',
        ms_level: Literal[1, 2],
        scan_num: int,
    ):
        """
        Draws an indicator in the appropriate widget, showing where
        the scan was selected
        """
        # Clear all indicators
        for sample_uuid, widget in self.sample_wdgt_mgr.get_all_widgets().items():

            # If this is the requested widget, draw indicator
            if sample_uuid == selected_uuid:
                # Get rt
                rt = self.model.getInjection(
                    sample_uuid
                ).get_scan_array(
                    ms_level
                ).rt_arr[scan_num]

                widget.setSelectionIndicator(xpos=rt)
                widget.setSelectionIndicatorVisible(True)

            # Otherwise hide indicator
            else:
                widget.setSelectionIndicatorVisible(False)

    def rebuild_plots(self):
        """
        Rebuild sample plots
         (deletes everything and starts over)
        """
        # Clear existing plots
        self.sample_wdgt_mgr.remove_all_widgets()

        # Add new plots
        for row_idx in range(0, self.model.rowCount()):
            sample_uuid = self.model.getSampleUuidAtRow(row_idx)
            if not sample_uuid:
                continue

            widget = self.sample_wdgt_mgr.create_widget(
                uuid=sample_uuid,
                row_idx=row_idx,
            )

            self.connect_sample_widget_signals(widget)
            self._populate_widget_elements(sample_uuid)

        # self.chrom_mgr.update_all_plots()
        # self.ensemble_ui_mgr.display_ensembles_for_all_samples(
        #     ms_level=self.chrom_mgr.get_ms_level(),
        # )

    def refresh_plot(
        self,
        sample_uuid: 'SampleUUID'
    ):
        self._populate_widget_elements(sample_uuid)

    def refresh_all_plots(self):
        """
        Updates the plots but does not delete them
        """
        for sample_uuid in self.sample_wdgt_mgr.get_all_widgets():
            self.refresh_plot(sample_uuid)


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

        # Ensemble peak interaction signals
        sample_widget.sigEnsemblePeakHovered.connect(
            self.on_ensemble_peak_hovered
        )
        sample_widget.sigEnsemblePeakLeaved.connect(
            self.on_ensemble_peak_leaved
        )
        sample_widget.sigEnsemblePeakClicked.connect(
            self.on_ensemble_peak_clicked
        )

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
                sample_widget: SampleWidget = self.sample_wdgt_mgr.get_widget(uuid)

                # Move slide selector in the hovered chromatogram
                sample_widget.setSliderSelector(
                    xpos=pos.x()
                )
                sample_widget.setSliderSelectorVisible(True)

                # Set mouse cursor to be pointing hand
                self.setCursor(QtCore.Qt.PointingHandCursor)  # type: ignore

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
        self.sample_wdgt_mgr.get_widget(uuid).setSliderSelectorVisible(False)

        if self._currently_hovered_uuid == uuid:
            self._currently_hovered_uuid = None
            self.setCursor(QtCore.Qt.ArrowCursor)  # type: ignore

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
        for uuid, sample_widget in self.sample_wdgt_mgr.get_all_widgets().items():
            fprint: 'Fingerprint' = self.model.getSample(uuid).fingerprint

            if not fprint:
                continue
                
            value = fprint.array[idx]
            descriptor = fprint.descriptors[idx]

            sample_widget.setFprintLabel(
                f"{value:.2f}\t{descriptor}"
            )

    def on_ensemble_peak_hovered(
        self,
        sample_uuid: 'SampleUUID',
        ensemble_uuid: 'EnsembleUUID',
        pos: QPointF,
    ):
        """
        Called when user hovers over an ensemble peak.
        Changes cursor and propagates signal upward.
        """
        # Set cursor to pointing hand to indicate clickability
        self.setCursor(QtCore.Qt.PointingHandCursor)

        # Propagate signal
        self.sigEnsemblePeakHovered.emit(
            sample_uuid,
            ensemble_uuid,
            pos,
        )

    def on_ensemble_peak_leaved(
        self,
        sample_uuid: 'SampleUUID',
    ):
        """Reset cursor when leaving peak"""
        self.unsetCursor()

    def on_ensemble_peak_clicked(
        self,
        sample_uuid: 'SampleUUID',
        ensemble_uuid: 'EnsembleUUID',
        button: int,
    ):
        """
        Handle peak clicks - propagate to top level for action handling
        """
        self.sigEnsemblePeakClicked.emit(
            sample_uuid,
            ensemble_uuid,
            button,
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
        widget = self.sample_wdgt_mgr.get_widget(uuid)
        if not widget:
            return

        # Toggle visibility based on check state
        widget.setVisible(
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

            if not self.sample_wdgt_mgr.get_widget(sample_uuid):
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

            widget = self.sample_wdgt_mgr.create_widget(
                uuid=sample_uuid,
                row_idx=row_idx,
            )

            self._populate_widget_elements(sample_uuid)
            self.connect_sample_widget_signals(widget)

        # self.chrom_mgr.update_all_plots()

    def on_rows_removed(
        self,
        parent,
        first: int,
        last: int,
    ):
        # Iterate deleted rows backwards to avoid idx issues
        for row_idx in range(last, first - 1, -1):
            uuid_to_remove: Optional[ 'SampleUUID' ] = None
            for uuid, widget in self.sample_wdgt_mgr.get_all_widgets().items():
                if self._widget_was_at_row(widget, row_idx):
                    uuid_to_remove = uuid
                    break

            if uuid_to_remove:
                widget = self.sample_wdgt_mgr.get_widget(uuid_to_remove)
                self.container_layout.removeWidget(widget)

                widget.deleteLater()

                self.sample_wdgt_mgr.remove_widget(uuid_to_remove)

        self.chrom_mgr.link_chrom_widget_axes()

    def on_rows_moved(self):
        # TODO: This isn't called by QStandardItem model. Meant for proxy models
        # Get all current widgets
        widgets: list[QtWidgets.QWidget | None] = []
        for i in range(self.model.rowCount()):
            item = self.model.item(i)
            if item:
                uuid = item.data(role=self.model.UuidRole)

                widget = self.sample_wdgt_mgr.get_widget(uuid)
                if widget:
                    widgets.append(
                        widget
                    )
                else:
                    widgets.append(None)  # Placeholder for missing widgets

        # Reorder the layout
        for i, widget in enumerate(widgets):
            if not widget:  # Skip missing
                continue

            self.container_layout.removeWidget(widget)
            self.container_layout.insertWidget(i, widget)

    def _populate_widget_elements(
        self,
        sample_uuid: 'SampleUUID'
    ):
        """
        Reads current settings/parameters and updates the widget
        accordingly
        """
        widget = self.sample_wdgt_mgr.get_widget(sample_uuid)
        if not widget:
            return

        # Update plot
        self.chrom_mgr.update_chromatogram(sample_uuid)
        self.chrom_mgr.update_fingerprint(sample_uuid)

        # Update ensembles based on current settings
        match self._show_ensembles:
            case True:
                self.ensemble_ui_mgr.display_ensembles_for_sample(
                    uuid=sample_uuid,
                    ms_level=self.chrom_mgr.get_ms_level(),
                )
            case False:
                self.ensemble_ui_mgr.clear_ensembles_for_sample(
                    sample_uuid
                )

        # Placeholders:
        # if self._show_ensemble_labels..
        # if self._ensemble_filter:
        #   self.ensemble_ui_mgr.apply_filter(...)

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
                for _, sample_widget in self.sample_wdgt_mgr.get_all_widgets().items():
                    sample_widget.setSliderSelectorVisible(False)

    def on_tool_stage_changed(self, stage: ToolStage) -> None:
        pass

    def on_xic_mode_changed(self, mode: XICMode) -> None:
        # self._xic_mode = mode
        self.chrom_mgr.set_xic_mode(mode)

    def on_fprints_toggled(
        self,
        show_fprints: bool,
    ):
        """
        Called when user toggles fingerprints

        :param show_fprints:
        :return:
        """
        for _, sample_widget in self.sample_wdgt_mgr.get_all_widgets().items():
            sample_widget.fprintPlotWidget.setVisible(
                show_fprints
            )

    def on_show_ensembles_toggled(
        self,
        show_ensembles: bool,
    ):
        self._show_ensembles = show_ensembles
        self.refresh_all_plots()

    def link_colorbar_to_fprint_plots(
        self,
        colorbaritem: pg.ColorBarItem,
    ):
        for _, chrom_widget in self.sample_wdgt_mgr.get_all_widgets().items():
            colorbaritem.setImageItem(
                chrom_widget.fprintPlotWidget.ImageItem
            )

    def _widget_was_at_row(
        self,
        widget: SampleWidget,
        row: int,
    ):
        pass
