
from gui.widgets.CustomAxisItems import IntsyAxisItem
from gui.widgets.TextOverlay import TextOverlay
from gui.utils.ms_arrays import zero_pad_arrays
from core.utils.arrays import find_closest_point
from core.utils.array_types import SpectrumArray
from gui.views.sample_viewer.tools import (
    ToolType, ToolStage, XICMode,
)

import numpy as np
import pyopenms as oms
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRectF

from typing import Optional
import uuid


class ClickableTextItem(pg.TextItem):
    """
    `pg.TextItem` that calls `on_click(button)` when clicked. Used to
    make annotation labels (anchored labels, ion-annotation labels,
    delta-bracket labels) interactive without subclassing every
    annotation-graphic composite. Defaults to right-button only so
    left-click view manipulation still works.
    """
    def __init__(self, *args, on_click=None, accepted_buttons=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_click = on_click
        if accepted_buttons is None:
            accepted_buttons = QtCore.Qt.RightButton
        self.setAcceptedMouseButtons(accepted_buttons)
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, ev):
        if self._on_click is not None:
            self.setCursor(QtCore.Qt.PointingHandCursor)
        super().hoverEnterEvent(ev)

    def hoverLeaveEvent(self, ev):
        self.unsetCursor()
        super().hoverLeaveEvent(ev)

    def mousePressEvent(self, ev):
        if self._on_click is not None:
            self._on_click(int(ev.button()))
            ev.accept()
            return
        super().mousePressEvent(ev)


class IonAnnotationGraphic:
    """
    Manages graphical items for an ion annotation: a text label
    and a theoretical isotope envelope overlaid as stick bars.
    """
    _ENVELOPE_COLOR = (180, 80, 80, 120)

    def __init__(
        self,
        mz_values: np.ndarray,
        intsy_values: np.ndarray,
        spec_idxs: list[int],
        text: str,
        envelope: np.ndarray,
        on_click=None,
    ):
        # Text label at monoisotopic peak
        self.label = create_textitem(
            text=text.replace('\n', '<br>'),
            pos=(mz_values[spec_idxs[0]], intsy_values[spec_idxs[0]]),
            level=1.5,
            on_click=on_click,
        )

        # Theoretical envelope bars
        self.envelope_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color=self._ENVELOPE_COLOR, width=3),
            connect='pairs',
        )

        self._set_envelope_data(
            mz_values, intsy_values, spec_idxs, envelope,
        )

    def _set_envelope_data(
        self,
        mz_values: np.ndarray,
        intsy_values: np.ndarray,
        spec_idxs: list[int],
        envelope: np.ndarray,
    ):
        """
        Scale the theoretical envelope to match the experimental
        monoisotopic intensity, then build zero-padded stick data.
        """
        if envelope.size == 0 or len(spec_idxs) == 0:
            self.envelope_curve.setData(x=[], y=[])
            return

        # envelope is [[mz, rel_intsy], ...] with monoisotopic = 1.0
        mono_intsy = intsy_values[spec_idxs[0]]
        scaled_intsy = envelope[:, 1] * mono_intsy

        # Zero-pad for stick plot (connect='pairs')
        n = len(envelope)
        x = np.zeros(n * 2)
        y = np.zeros(n * 2)
        x[0::2] = envelope[:, 0]
        x[1::2] = envelope[:, 0]
        y[0::2] = scaled_intsy
        # y[1::2] already 0

        self.envelope_curve.setData(x=x, y=y)

    def add_to_plot(self, plotitem: 'pg.PlotItem'):
        plotitem.addItem(self.envelope_curve)
        plotitem.addItem(self.label)

    def remove_from_plot(self, plotitem: 'pg.PlotItem'):
        plotitem.removeItem(self.envelope_curve)
        plotitem.removeItem(self.label)


class DeltaMzBracket:
    """
    Manages graphical items for a bracket annotation connecting two
    MS signals and displaying the delta m/z label.
    """
    def __init__(
        self,
        mz_a: float,
        intsy_a: float,
        mz_b: float,
        intsy_b: float,
        text: str,
        on_click=None,
    ):
        self.curve = pg.PlotCurveItem(
            pen=pg.mkPen(color=(80, 80, 255), width=1.5),
            connect='all',
        )
        label_html = (
            f'<div style="text-align: center; color: rgb(80, 80, 255);">'
            f'{text}</div>'
        )
        if on_click is not None:
            self.label = ClickableTextItem(
                html=label_html, anchor=(0.5, -0.05), on_click=on_click,
            )
        else:
            self.label = pg.TextItem(html=label_html, anchor=(0.5, -0.05))
        self.label = center_textitem(self.label)
        self.update_positions(mz_a, intsy_a, mz_b, intsy_b)

    def update_positions(
        self,
        mz_a: float,
        intsy_a: float,
        mz_b: float,
        intsy_b: float,
    ):
        max_intsy = max(intsy_a, intsy_b)
        x = np.array([mz_a, mz_a, mz_b, mz_b])
        y = np.array([intsy_a, max_intsy, max_intsy, intsy_b])
        self.curve.setData(x=x, y=y)
        self.label.setPos((mz_a + mz_b) / 2, max_intsy)

    def update_text(self, text: str):
        self.label.setHtml(
            f'<div style="text-align: center;">{text}</div>'
        )

    def add_to_plot(self, plotitem: 'pg.PlotItem'):
        plotitem.addItem(self.curve)
        plotitem.addItem(self.label)

    def remove_from_plot(self, plotitem: 'pg.PlotItem'):
        plotitem.removeItem(self.curve)
        plotitem.removeItem(self.label)


class MSPlotWidget(pg.PlotWidget):
    """
    Custom PlotWidget with sensible behaviour for
    zooming/panning mass spectra
    """
    sigSpectrumArrayChanged = QtCore.pyqtSignal()
    sigExtractionRegionChanged = QtCore.pyqtSignal(
        tuple,  # region (min, max)
    )

    # Used by sample_viewer to generate transient xics
    sigMSSignalHovered = QtCore.pyqtSignal(tuple)  # tuple[spec_idx, mz_float]
    sigMSSignalClicked = QtCore.pyqtSignal(tuple) # tuple[spec_idx, mz_float]
    sigMSpectrumLeaved = QtCore.pyqtSignal()

    # Used by tool manager
    sigSelectionMade = QtCore.pyqtSignal()
    sigConfigurationMade = QtCore.pyqtSignal(dict)

    # Forwarded from MSLabelManager.sigAnnotationClicked.
    # Args: (plot_id: str, button: int).
    sigAnnotationClicked = QtCore.pyqtSignal(str, int)


    def __init__(self, *args, **kwargs):
        super(MSPlotWidget, self).__init__(
            *args,
            **kwargs,
            plotItem=MSPlotItem(
                plot_widget=self,
                axisItems={
                    'left': IntsyAxisItem('left'),
                    # 'bottom': MzAxisItem('bottom'),
                }
            ),
        )
        # self.setBackground(None)
        self.pi : MSPlotItem = self.getPlotItem()

        self.sigSpectrumArrayChanged.connect(
            self.pi.updateSpectrumPlot
        )

        # Forward annotation-click signal up from the label manager.
        self.pi.label_manager.sigAnnotationClicked.connect(
            self.sigAnnotationClicked
        )

        # TODO: Expose to user!!
        self.pi.vb.setLimits(
            xMin=0,
            xMax=2000,
        )
        self.pi.vb.setXRange(
            min=0,
            max=1000,
        )
        self.pi.vb.setYRange(
            min=0,
            max=1e7,
        )

        # 'Region selector' used to make XICs/BPCs
        self.region_selector = pg.LinearRegionItem(
            values=(200,300),
            pen=pg.mkPen('grey'),
        )
        self.region_selector.sigRegionChanged.connect(
            self.on_extraction_region_changed
        )
        self.addItem(
            self.region_selector
        )

        # Markers used to indicate selected MS signals
        #   { spec_idx: targetitem }
        self.signal_markers: dict[int, pg.TargetItem] = {}

        # Floating label
        self.floating_label = QtWidgets.QLabel(self)
        self.floating_label.setStyleSheet(
            "QLabel { color: black; background-color: rgba(225, 225, 225, 128) }"
        )  # TODO: experiment w translucent background
        self.floating_label.move(60, 5)
        self.floating_label.hide()
        self.floating_label.raise_()

        # BPC intensity readout (top-right corner)
        self.bpc_label = QtWidgets.QLabel(self)
        self.bpc_label.setStyleSheet(
            "QLabel { color: black; background-color: rgba(225, 225, 225, 128); "
            "padding: 1px 4px; }"
        )
        self.bpc_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.bpc_label.hide()
        self.bpc_label.raise_()

        # Set default extraction mode to be NONE
        self.on_xic_mode_changed(XICMode.NONE)

        # Local copy of tool state
        self._tool_type: 'ToolType' = ToolType.NONE
        self._tool_stage: 'ToolStage' = ToolStage.IDLE
        self._xic_mode: 'XICMode' = XICMode.NONE

        # Hovered MS signal
        self.hovered_ms_signal: Optional[tuple[int, float]] = None

    def update_label(
        self,
        text: str,
    ):
        self.floating_label.setText(text)
        self.floating_label.raise_()

        # Force a refresh of the widget stack
        self.floating_label.setParent(None)
        self.floating_label.setParent(self)
        self.floating_label.show()

    def update_bpc_label(
        self,
        text: str,
    ):
        """
        Sets the top-right BPC intensity readout. Empty text hides it.
        """
        if not text:
            self.bpc_label.hide()
            return
        self.bpc_label.setText(text)
        self.bpc_label.adjustSize()
        self._reposition_bpc_label()
        self.bpc_label.show()
        self.bpc_label.raise_()

    def _reposition_bpc_label(self):
        margin = 8
        x = self.width() - self.bpc_label.width() - margin
        self.bpc_label.move(max(0, x), 5)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # resizeEvent fires once from inside pyqtgraph's __init__ before our
        # own __init__ has run — guard against the attribute not yet existing.
        label = self.__dict__.get('bpc_label')
        if label is not None and label.isVisible():
            self._reposition_bpc_label()


    def add_signal_marker(
        self,
        spec_idx: int,
    ):
        """
        Adds a signal marker to the plot, based on the currently
        displayed SpectrumArray
        :param spec_idx:
        :return:
        """
        spectrum_array = self.pi.spectrum_array

        if spectrum_array is None:
            return

        mz = spectrum_array['mz'][spec_idx]
        intsy = spectrum_array['intsy'][spec_idx]

        signal_marker = pg.TargetItem(
            pos=(mz, intsy),
            movable=False,
        )

        self.signal_markers[spec_idx] = signal_marker
        self.pi.addItem(signal_marker)


    def remove_signal_marker(
        self,
        spec_idx: int,
    ):
        signal_marker = self.signal_markers.get(spec_idx)
        if signal_marker:
            self.pi.removeItem(signal_marker)
            del self.signal_markers[spec_idx]


    def clear_signal_markers(self):
        for _, signal_marker in self.signal_markers.items():
            self.pi.removeItem(signal_marker)

        self.signal_markers.clear()


    # Anchored label API methods
    def add_anchored_label(
        self,
        spec_idx: int,
        text: str,
        label_id: Optional[str] = None,
    ) -> str:
        """
        Adds an anchored label at the specified spectrum array index.

        :param spec_idx: Index into the current spectrum_array
        :param text: Text content for the label (e.g., chemical formula)
        :param label_id: Optional unique identifier. Auto-generated if not provided.
        :return: The label_id of the created label
        """
        return self.pi.label_manager.add_anchored_label(
            spec_idx,
            text,
            label_id,
        )

    def remove_anchored_label(
        self,
        label_id: str,
    ) -> bool:
        """
        Removes an anchored label by its ID.

        :param label_id: The unique identifier of the label to remove
        :return: True if label was found and removed, False otherwise
        """
        return self.pi.label_manager.remove_anchored_label(label_id)

    def update_anchored_label(
        self,
        label_id: str,
        text: Optional[str] = None,
        spec_idx: Optional[int] = None,
    ) -> bool:
        """
        Updates an existing anchored label's text and/or position.

        :param label_id: The unique identifier of the label to update
        :param text: New text content (if None, keeps existing text)
        :param spec_idx: New spectrum index (if None, keeps existing position)
        :return: True if label was found and updated, False otherwise
        """
        return self.pi.label_manager.update_anchored_label(label_id, text, spec_idx)

    def clear_anchored_labels(self) -> None:
        """
        Removes all anchored labels from the plot.
        """
        self.pi.label_manager.clear_anchored_labels()

    # Ion annotation API methods
    def add_ion_annotation(
        self,
        spec_idxs: list[int],
        text: str,
        envelope: np.ndarray,
        annot_id: Optional[str] = None,
    ) -> str:
        return self.pi.label_manager.add_ion_annotation(
            spec_idxs, text, envelope, annot_id,
        )

    def remove_ion_annotation(self, annot_id: str) -> bool:
        return self.pi.label_manager.remove_ion_annotation(annot_id)

    def clear_ion_annotations(self) -> None:
        self.pi.label_manager.clear_ion_annotations()

    # Delta bracket API methods
    def add_delta_bracket(
        self,
        spec_idx_a: int,
        spec_idx_b: int,
        text: str,
        bracket_id: Optional[str] = None,
    ) -> str:
        return self.pi.label_manager.add_delta_bracket(
            spec_idx_a, spec_idx_b, text, bracket_id,
        )

    def remove_delta_bracket(self, bracket_id: str) -> bool:
        return self.pi.label_manager.remove_delta_bracket(bracket_id)

    def clear_delta_brackets(self) -> None:
        self.pi.label_manager.clear_delta_brackets()


    def on_extraction_region_changed(
        self,
    ):
        self.sigExtractionRegionChanged.emit(
            self.region_selector.getRegion(),
        )


    def move_region_selector(
        self,
        region: Optional[tuple[float, float]] = None,
        center: Optional[tuple[float, float]] = None,
    ):
        """
        Moves the selector window to either `region` or `center`.

        Will use `region` if given. Otherwise, will use `center`.
        Raises an error if neither arguments are provided

        :param region: tuple(start, end)
        :param center: tuple(center, window)
        :return:
        """
        if not region and not center:
            raise ValueError(
                "Neither arguments region nor center were given"
            )

        if region:
            self.region_selector.setRegion(
                region
            )

        elif center:
            start = center[0] - center[1]
            end = center[0] + center[1]
            self.region_selector.setRegion(
                (start, end)
            )

        self.on_extraction_region_changed()


    def on_xic_mode_changed(
        self,
        mode: XICMode
    ) -> None:
        match mode:
            case XICMode.NONE:
                # Hide and disabe region selector
                self.region_selector.setVisible(False)

            case XICMode.XIC:
                self.region_selector.setVisible(True)
                # self.on_extraction_region_changed()  # Manually trigger on init

            case XICMode.BPC:
                self.region_selector.setVisible(True)
                # self.on_extraction_region_changed()  # Manually trigger on init

        self._xic_mode = mode


    def on_tool_type_changed(
        self,
        tool_type: 'ToolType',
    ) -> None:
        self._tool_type = tool_type


    def on_tool_stage_changed(
        self,
        stage: ToolStage,
    ) -> None:
        match stage:
            case ToolStage.SELECTING:
                pass

            case ToolStage.CONFIGURING:
                pass

            case ToolStage.READY_TO_EXECUTE:
                pass

            case ToolStage.IDLE:
                pass

        self._tool_stage = stage

    def setSpectrumPlotPen(
        self,
        *args,
        **kwargs,
    ):
        """
        Wrapper around the setPen() func for the PlotDataItem used to
        display spectra
        """
        self.pi.spectrum_plot.setPen(*args, **kwargs)

    def setSpectrumArray(
        self,
        spectrum_array: SpectrumArray,
    ):
        self.pi.setSpectrumArray(spectrum_array)


    def MSSignalHovered(
        self,
        spectrum_idx: Optional[int],
        mz: Optional[ float ],
    ):
        """
        Called when user hovers near an MS signal.

        Sets self.hovered_ms_signal to (spectrum_idx, mz),
        or None if the user is no longer hovering
        """
        match self._tool_type:
            case ToolType.NONE:
                self.unsetCursor()

            case ToolType.GETCOMPOUND:
                self.setCursor(
                    QtCore.Qt.PointingHandCursor
                )

            case ToolType.GETXIC:
                self.setCursor(
                    QtCore.Qt.CrossCursor
                )

        if spectrum_idx is not None:
            self.hovered_ms_signal = (spectrum_idx, mz)
            self.sigMSSignalHovered.emit(self.hovered_ms_signal)

        else:
            self.hovered_ms_signal = None
            self.sigMSpectrumLeaved.emit()
            self.unsetCursor()


    def MSSignalClicked(
        self,
    ):
        self.sigMSSignalClicked.emit(
            self.hovered_ms_signal
        )

        match self._tool_type:
            case ToolType.GETCOMPOUND:
                print("Todo: implement GETCOMPOUND")
                self.sigSelectionMade.emit()

            case ToolType.GETXIC:
                self.move_region_selector(
                    center=(
                        self.hovered_ms_signal[1],  # mz
                        0.4,                        # Window (Da)
                    )
                )

                # Exits SELECTION mode
                self.sigSelectionMade.emit()


    def mousePressEvent(
        self,
        QMouseEvent: QtGui.QMouseEvent,
    ):
        super().mousePressEvent(QMouseEvent)

        if self.hovered_ms_signal:
            self.MSSignalClicked()


class MSPlotItem(pg.PlotItem):
    """
    Custom PlotItem implementing features such as
     - snapping hover detection
     - MS-tailored auto-panning behaviour
     - MS signal labeling that avoids crowding/overlap
    """
    def __init__(
            self,
            plot_widget: MSPlotWidget,
            *args,
            **kargs,
    ):
        super(MSPlotItem, self).__init__(
            *args,
            **kargs,
            viewBox=MSViewBox(
                defaultPadding=0.0
            )
        )
        self.plot_widget = plot_widget

        # Back-reference so the ViewBox can delegate auto-scaling to us
        # (data-driven, see scaleViewboxToSpectrumArray()).
        self.vb._plot_item = self

        # Spectrum PlotDataItem
        # TODO: Optimize w bespoke QGraphicsItem
        self.spectrum_plot: pg.PlotDataItem = pg.PlotDataItem(
            connect='pairs',
        )
        self.addItem(
            self.spectrum_plot,
        )

        # Mass bin indicator PlotDataItem
        self.mass_bin_markers: pg.PlotDataItem = pg.PlotDataItem(
            symbol='x',
        )
        self.addItem(
            self.mass_bin_markers,
        )

        # Text overlay (Upper left corner)
        self.text_overlay = TextOverlay(text="", offset=(50, -10))
        self.text_overlay.setParentItem(self)

        # Label Manager (Used for overlap detection)
        self.label_manager = MSLabelManager(self)

        # Vertical cursor line, snaps to nearest/hovered MS signal
        self.vert_cursor: pg.InfiniteLine = pg.InfiniteLine(
            pos=-1,
            
        )
        self.vert_cursor_label: pg.InfLineLabel = pg.InfLineLabel(
            line=self.vert_cursor,
            position=0.90,
            anchor=(0.5, 0),
            text='',
        )
        self.addItem(
            self.vert_cursor
        )

        # State tracking
        self.spectrum: oms.MSSpectrum = oms.MSSpectrum()
        self.spectrum_array: dict = {
            'mz': np.array([]),
            'intsy': np.array([]),
        }



    def setSpectrumArray(
        self,
        spectrum_array: SpectrumArray
    ) -> None:
        self.spectrum_array = spectrum_array
        # Clear all annotations when spectrum changes
        self.label_manager.clear_anchored_labels()
        self.label_manager.clear_ion_annotations()
        self.label_manager.clear_delta_brackets()
        self.plot_widget.sigSpectrumArrayChanged.emit()


    def updateSpectrumPlot(
        self,
    ) -> None:
        self.spectrum_plot.setData(
            *zero_pad_arrays(
                mz_arr=self.spectrum_array['mz'],
                intsy_arr=self.spectrum_array['intsy'],
            )
        )
        self.add_mass_labels()

        # Force the QGraphicsScene to repaint the full viewport. Without
        # this, pyqtgraph occasionally leaves stale primitives (old labels,
        # bracket fragments, scatter points) painted until the user nudges
        # the viewbox. Cheap; runs once per spectrum update.
        scene = self.scene()
        if scene is not None:
            scene.update()
        if self.plot_widget is not None:
            self.plot_widget.viewport().update()


    def scaleViewboxToSpectrumArray(
            self,
    ):
        """
        Scales viewbox to fit the spectrum array
        """
        if self.spectrum_array is None:
            return

        if self.spectrum_array['mz'].size == 0:
            return

        self.vb: 'pg.ViewBox'
        self.vb.setXRange(
            min=0,
            max=(
                    np.nanmax(self.spectrum_array['mz']) * 1.10
            ),
        )

        self.vb.setYRange(
            min=0,
            max=(
                    np.nanmax(self.spectrum_array['intsy']) * 1.60
            ),
        )

    def add_mass_labels(self):
        # Clear previous annotations
        for label in self.label_manager.labels:
            label: pg.TextItem
            self.removeItem(label)

        # Build new annotations
        labels: list[pg.TextItem] = []

        # Get the tallest 100 peaks
        sort_key = np.argsort(
            self.spectrum_array['intsy']
        )

        tallest_mz = self.spectrum_array['mz'][sort_key][-100:]
        tallest_intsy = self.spectrum_array['intsy'][sort_key][-100:]

        # Add m/z labels
        for mz, intsy in zip(tallest_mz, tallest_intsy):
            text = f"{mz:.5f}" # TODO: expose precision to user

            textitem = create_textitem(
                text=text,
                pos=(
                    mz,
                    intsy,
                )
            )

            self.addItem(textitem)
            labels.append(textitem)

        self.label_manager.set_peak_data(
            self.spectrum_array
        )
        self.label_manager.set_labels(
            labels,
            priority_key=lambda x: x.pos().y()  # Sort by intensity
        )


    def snap_vert_cursor_to_nearest(
            self,
            ev
    ):
        if self.spectrum_array is None:
            return

        if self.spectrum_array['mz'].size == 0:
            return

        # Convert mouse coords from scene to plot coords
        mouse_loc = self.getViewBox().mapSceneToView(ev.pos())

        # Get 'pixel size' in plot units
        pixel_size_x, pixel_size_y = self.vb.viewPixelSize()

        # Scale mouse_loc accordingly
        mouse_loc_x = mouse_loc.x() / pixel_size_x
        mouse_loc_y = mouse_loc.y() / pixel_size_y

        # Scale `spectrum_array`
        scaled_spectrum_array_mz = self.spectrum_array['mz'] / pixel_size_x
        scaled_spectrum_array_intsy = self.spectrum_array['intsy'] / pixel_size_y

        # Find idx of nearest point in `spectrum_array`
        idx, dist_squared = find_closest_point(
            tgt_x=mouse_loc_x,
            tgt_y=mouse_loc_y,
            data_x=scaled_spectrum_array_mz,
            data_y=scaled_spectrum_array_intsy,
        )

        if dist_squared < 625:  # i.e. less than 25 pixels
            mz = self.spectrum_array['mz'][idx]
            self.move_vert_cursor(
                mz
            )

            self.plot_widget.MSSignalHovered(
                spectrum_idx=idx,
                mz=mz,
            )

        else:
            self.no_signal_hovered()


    def move_vert_cursor(
            self,
            position: int | float,
    ):
        self.vert_cursor.setValue(
            position
        )


    def hoverEvent(self, ev):
        super().hoverEvent(ev)

        if not ev:
            return

        if ev.exit:
            # User no longer hovering spectrum
            self.no_signal_hovered()

        else:
            self.snap_vert_cursor_to_nearest(ev)


    def no_signal_hovered(self):
        """
        Called when the user is not hovering on/near an MS signal.
        Hides the vertical cursor, and emits appropriate Qt signal
        """
        self.plot_widget.MSSignalHovered(
            spectrum_idx=None,
            mz=None,
        )
        self.move_vert_cursor(-1)


class MSViewBox(
    pg.ViewBox
):
    """
    Custom ViewBox with sensible behaviour for zooming/panning MS data
    """
    def __init__(self, *args, **kwargs):
        super(MSViewBox, self).__init__(*args, **kwargs)
        self.disableAutoRange()
        # Neither m/z nor intensity is ever negative — clamp the view so it
        # can never descend below zero on either axis.
        self.setLimits(xMin=0.0, yMin=0.0)

    def mouseDoubleClickEvent(self, ev):
        # Delegate to the PlotItem's data-driven auto-scale rather than
        # pyqtgraph's generic autoRange(), which can produce a negative
        # Y window when ranging against non-data items (vert cursor,
        # region selector).
        plot_item = getattr(self, '_plot_item', None)
        if plot_item is not None:
            plot_item.scaleViewboxToSpectrumArray()
        else:
            self.autoRange()

    def mouseDragEvent(self, ev, axis=None):
        # if axis is specified, event will only affect that axis.
        ev.accept()  # we accept all buttons

        pos = ev.pos()
        lastPos = ev.lastPos()
        dif = pos - lastPos
        dif = dif * -1

        # Ignore axes if mouse is disabled
        mouseEnabled = np.array(self.state['mouseEnabled'], dtype=np.float64)
        mask = mouseEnabled.copy()
        if axis is not None:
            mask[1 - axis] = 0.0

        # Scale or translate based on mouse button
        if ev.button() in [QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.MiddleButton]:
            if self.state['mouseMode'] == pg.ViewBox.RectMode and axis is None:
                if ev.isFinish():  # This is the final move in the drag; change the view scale now
                    # print "finish"
                    self.rbScaleBox.hide()
                    ax = QtCore.QRectF(pg.Point(ev.buttonDownPos(ev.button())), pg.Point(pos))
                    ax = self.childGroup.mapRectFromParent(ax)
                    self.showAxRect(ax)
                    self.axHistoryPointer += 1
                    self.axHistory = self.axHistory[:self.axHistoryPointer] + [ax]
                else:
                    # update shape of scale box
                    self.updateScaleBox(ev.buttonDownPos(), ev.pos())
            else:
                tr = self.childGroup.transform()
                tr = pg.functions.invertQTransform(tr)
                tr = tr.map(dif * mask) - tr.map(pg.Point(0, 0))

                x = tr.x() if mask[0] == 1 else None
                y = tr.y() if mask[1] == 1 else None

                self._resetTarget()
                if x is not None or y is not None:
                    self.translateBy(x=x, y=0)  # (x=x, y=y)
                self.sigRangeChangedManually.emit(self.state['mouseEnabled'])

        elif ev.button() & QtCore.Qt.MouseButton.RightButton:
            # print "vb.rightDrag"
            if self.state['aspectLocked'] is not False:
                mask[0] = 0

            dif = ev.screenPos() - ev.lastScreenPos()
            dif = np.array([dif.x(), dif.y()])
            dif[0] *= -1
            s = ((mask * 0.02) + 1) ** dif

            tr = self.childGroup.transform()
            tr = pg.functions.invertQTransform(tr)

            # Sets either the x or y scaling factor to 0 depending on whether
            #  the mouse is hovering over the axis
            # s = np.multiply(s, mask)
            # s[s == 0.] = None

            x = s[0] if mouseEnabled[0] == 1 else None
            y = s[1] if mouseEnabled[1] == 1 else None
            if mask[1] != 1:
                y = None   # Disables y-scaling if not dragging on y-axis

            center = pg.Point(tr.map(ev.buttonDownPos(QtCore.Qt.MouseButton.RightButton)))
            center[1] = 0  # Set the center-point's y-value to be 0

            self._resetTarget()
            self.scaleBy(x=x, y=y, center=center)
            self.sigRangeChangedManually.emit(self.state['mouseEnabled'])

    def wheelEvent(self, ev, axis=None):
        """
        Overwrites ViewBox methods to behave more sensibly for MS data
        """
        # Retrieve scaling factor setting
        s = 1.02 ** (ev.delta() * self.state['wheelScaleFactor'])

        center = pg.Point(pg.functions.invertQTransform(self.childGroup.transform()).map(ev.pos()))
        if axis == 0:  # If mouse is hovering on X-axis:
            s = [s, None]  # Don't zoom in the y-direction

        else:  # Otherwise,
            s = [None, s]  # Don't zoom in the x-direction
            center[1] = 0  # Set the center-point's y-value to be 0
            # This is so the plot zooms by 'stretching upwards'

        self._resetTarget()
        self.scaleBy(s, center=center)
        ev.accept()

        # This block might be unneccessary, but included so nothing breaks
        if axis in (0, 1):
            mask = [False, False]
            mask[axis] = self.state['mouseEnabled'][axis]
        else:
            mask = self.state['mouseEnabled'][:]

        self.sigRangeChangedManually.emit(mask)


class MSLabelManager(QtCore.QObject):
    """
    Used to show/hide MS labels to resolve overlaps between labels,
    as well as overlaps between MS data
    """
    # Emitted when an annotation's label is clicked. Carries the
    # annotation's plot-side ID (the same string returned by
    # `add_anchored_label` / `add_ion_annotation` / `add_delta_bracket`)
    # and the Qt mouse button as int.
    sigAnnotationClicked = QtCore.pyqtSignal(str, int)

    def __init__(
            self,
            plotitem: 'MSPlotItem',
            clearance: float = 5.0,  # Pixels (I think?)
            intsy_threshold: float = 0.01, # Relative to max intsy
    ):
        super().__init__()
        self.plotitem: 'MSPlotItem' = plotitem
        self.priority_key = None
        self.labels: list[pg.TextItem] = []

        self.clearance = clearance
        self.intsy_threshold = intsy_threshold
        self.absolute_intsy_threshold = 0
        self.peak_data: Optional[SpectrumArray] = None

        # Anchored labels - user-controlled annotations tied to spectrum indices
        self.anchored_labels: dict[str, dict] = {}

        # Ion annotation graphics (label + isotope envelope)
        self.ion_annotations: dict[str, dict] = {}

        # Delta m/z brackets connecting two signals
        self.delta_brackets: dict[str, dict] = {}

        # Connect to view change signals
        vb = self.plotitem.vb
        vb.sigRangeChanged.connect(self.on_view_changed)
        vb.sigTransformChanged.connect(self.on_view_changed)

    @staticmethod
    def get_label_bounds(
            text_item: pg.TextItem
    ) -> QRectF:
        """
        Retrieves bounding rect for a TextItem in scene coordinates
        """
        bounds: QRectF = text_item.boundingRect()
        scene_bounds: QRectF = text_item.mapToParent(bounds).boundingRect()
        return scene_bounds

    @staticmethod
    def check_overlap(
            rect1: QRectF,
            rect2: QRectF,
    ) -> bool:
        """
        Check if two QRectF objects overlap
        """
        if np.isnan(rect1.left()) or np.isnan(rect2.left()):
            return False
        return rect1.intersects(rect2)

    def set_labels(
            self,
            labels: list[pg.TextItem],
            priority_key=None,
    ):
        self.labels = labels
        self.priority_key = priority_key
        self.update_visibility()

    def set_peak_data(
            self,
            data: dict,
    ):
        """
        Sets the MS spectrum to check for overlaps
        """
        self.peak_data = data
        # Guard against empty/peakless spectra (e.g. a blank scan, or a
        # cleared plot): .max() has no identity on a zero-size array.
        self.absolute_intsy_threshold = (
                data['intsy'].max() * self.intsy_threshold
                if data['intsy'].size
                else 0.0
        )

    def add_anchored_label(
            self,
            spec_idx: int,
            text: str,
            label_id: Optional[str] = None,
    ) -> str:
        """
        Adds an anchored label at the specified spectrum array index.

        :param spec_idx: Index into the current spectrum_array
        :param text: Text content for the label (e.g., chemical formula)
        :param label_id: Optional unique identifier. Auto-generated if not provided.
        :return: The label_id of the created label
        """
        if self.peak_data is None or self.peak_data['mz'].size == 0:
            raise ValueError("No spectrum data available")

        if spec_idx < 0 or spec_idx >= len(self.peak_data['mz']):
            raise IndexError(f"spec_idx {spec_idx} out of range for spectrum array")

        # Generate label_id if not provided
        if label_id is None:
            label_id = str(uuid.uuid4())

        # Get position from spectrum array
        mz: float = self.peak_data['mz'][spec_idx]
        intsy: float = self.peak_data['intsy'][spec_idx]

        # Convert newlines to HTML line breaks
        html_text = text.replace('\n', '<br>')

        # Create the text item (clickable so listeners can wire
        # right-click → delete or similar).
        def _on_click(button, _id=label_id):
            self.sigAnnotationClicked.emit(_id, button)

        textitem = create_textitem(
            text=html_text,
            pos=(mz, intsy),
            level=1.5,
            on_click=_on_click,
        )

        # Add to plot
        self.plotitem.addItem(textitem)

        # Store in anchored labels
        self.anchored_labels[label_id] = {
            'spec_idx': spec_idx,
            'text': text,
            'textitem': textitem,
        }

        return label_id

    def remove_anchored_label(
            self,
            label_id: str,
    ) -> bool:
        """
        Removes an anchored label by its ID.

        :param label_id: The unique identifier of the label to remove
        :return: True if label was found and removed, False otherwise
        """
        if label_id not in self.anchored_labels:
            return False

        # Remove from plot
        label_data = self.anchored_labels[label_id]
        self.plotitem.removeItem(label_data['textitem'])

        # Remove from storage
        del self.anchored_labels[label_id]

        return True

    def update_anchored_label(
            self,
            label_id: str,
            text: Optional[str] = None,
            spec_idx: Optional[int] = None,
    ) -> bool:
        """
        Updates an existing anchored label's text and/or position.

        :param label_id: The unique identifier of the label to update
        :param text: New text content (if None, keeps existing text)
        :param spec_idx: New spectrum index (if None, keeps existing position)
        :return: True if label was found and updated, False otherwise
        """
        if label_id not in self.anchored_labels:
            return False

        if self.peak_data is None or self.peak_data['mz'].size == 0:
            raise ValueError("No spectrum data available")

        label_data = self.anchored_labels[label_id]

        # Update text if provided
        if text is not None:
            label_data['text'] = text
            # Convert newlines to HTML line breaks
            html_text = text.replace('\n', '<br>')
            label_data['textitem'].setHtml(html_text)

        # Update position if provided
        if spec_idx is not None:
            if spec_idx < 0 or spec_idx >= len(self.peak_data['mz']):
                raise IndexError(f"spec_idx {spec_idx} out of range for spectrum array")

            label_data['spec_idx'] = spec_idx
            mz = self.peak_data['mz'][spec_idx]
            intsy = self.peak_data['intsy'][spec_idx]
            label_data['textitem'].setPos(mz, intsy)

        return True

    def clear_anchored_labels(self) -> None:
        """
        Removes all anchored labels from the plot.
        """
        for label_data in self.anchored_labels.values():
            self.plotitem.removeItem(label_data['textitem'])

        self.anchored_labels.clear()

    def add_ion_annotation(
            self,
            spec_idxs: list[int],
            text: str,
            envelope: np.ndarray,
            annot_id: Optional[str] = None,
    ) -> str:
        """
        Adds an ion annotation graphic: a text label at the monoisotopic
        peak plus a theoretical isotope envelope overlay.

        :param spec_idxs: Indices into the current spectrum_array
        :param text: HTML text for the label
        :param envelope: Array of [m/z, rel_intensity] from get_isotope_envelope()
        :param annot_id: Optional unique identifier
        :return: The annot_id of the created annotation
        """
        if self.peak_data is None or self.peak_data['mz'].size == 0:
            raise ValueError("No spectrum data available")

        if annot_id is None:
            annot_id = str(uuid.uuid4())

        def _on_click(button, _id=annot_id):
            self.sigAnnotationClicked.emit(_id, button)

        graphic = IonAnnotationGraphic(
            mz_values=self.peak_data['mz'],
            intsy_values=self.peak_data['intsy'],
            spec_idxs=spec_idxs,
            text=text,
            envelope=envelope,
            on_click=_on_click,
        )
        graphic.add_to_plot(self.plotitem)

        self.ion_annotations[annot_id] = {
            'spec_idxs': spec_idxs,
            'text': text,
            'graphic': graphic,
        }

        return annot_id

    def remove_ion_annotation(self, annot_id: str) -> bool:
        if annot_id not in self.ion_annotations:
            return False

        self.ion_annotations[annot_id]['graphic'].remove_from_plot(self.plotitem)
        del self.ion_annotations[annot_id]
        return True

    def clear_ion_annotations(self) -> None:
        for data in self.ion_annotations.values():
            data['graphic'].remove_from_plot(self.plotitem)
        self.ion_annotations.clear()

    def add_delta_bracket(
            self,
            spec_idx_a: int,
            spec_idx_b: int,
            text: str,
            bracket_id: Optional[str] = None,
    ) -> str:
        if self.peak_data is None or self.peak_data['mz'].size == 0:
            raise ValueError("No spectrum data available")

        if bracket_id is None:
            bracket_id = str(uuid.uuid4())

        mz_a = float(self.peak_data['mz'][spec_idx_a])
        intsy_a = float(self.peak_data['intsy'][spec_idx_a])
        mz_b = float(self.peak_data['mz'][spec_idx_b])
        intsy_b = float(self.peak_data['intsy'][spec_idx_b])

        def _on_click(button, _id=bracket_id):
            self.sigAnnotationClicked.emit(_id, button)

        bracket = DeltaMzBracket(
            mz_a, intsy_a, mz_b, intsy_b, text, on_click=_on_click,
        )
        bracket.add_to_plot(self.plotitem)

        self.delta_brackets[bracket_id] = {
            'spec_idx_a': spec_idx_a,
            'spec_idx_b': spec_idx_b,
            'text': text,
            'bracket': bracket,
        }

        return bracket_id

    def remove_delta_bracket(self, bracket_id: str) -> bool:
        if bracket_id not in self.delta_brackets:
            return False

        self.delta_brackets[bracket_id]['bracket'].remove_from_plot(self.plotitem)
        del self.delta_brackets[bracket_id]
        return True

    def clear_delta_brackets(self) -> None:
        for data in self.delta_brackets.values():
            data['bracket'].remove_from_plot(self.plotitem)
        self.delta_brackets.clear()

    def check_peak_overlap(
            self,
            label_bounds: QRectF
    ) -> bool:
        # if not self.peak_data.any():
        #     return False

        if self.peak_data['mz'].size == 0:
            return False

        if np.isnan(label_bounds.right()):
            return False

        mz_values, intsys = self.peak_data['mz'], self.peak_data['intsy']

        # Find peaks that intersect with mz signals
        mask = (
            (mz_values >= label_bounds.left())
            & (mz_values <= label_bounds.right())
            & (intsys >= label_bounds.top())
        )
        return np.any(mask)

    def on_view_changed(self):
        self.update_visibility()

    def update_visibility(self):
        # Ensure anchored labels are always visible
        for label_data in self.anchored_labels.values():
            label_data['textitem'].setVisible(True)

        # Ensure ion annotation labels are always visible
        for annot_data in self.ion_annotations.values():
            annot_data['graphic'].label.setVisible(True)

        if not self.labels:
            return

        # Reset visibility for auto-generated labels
        for label in self.labels:
            label.setVisible(True)

        # Sort labels by priority if provided
        working_labels = self.labels
        if self.priority_key:
            working_labels = sorted(
                working_labels,
                key=self.priority_key,
                reverse=True,
            )

        visible_bounds = []
        hidden_count = 0

        # Only apply overlap detection to auto-generated labels
        for label in working_labels:
            label: pg.TextItem
            current_bounds = self.get_label_bounds(label)

            # Check if peak is taller than threshold
            if current_bounds.y() < self.absolute_intsy_threshold:
                label.setVisible(False)
                hidden_count += 1
                continue

            # Check for overlap with peaks
            if self.check_peak_overlap(current_bounds):
                label.setVisible(False)
                hidden_count += 1
                continue

            # Check for overlap with other visible labels
            overlaps = any(
                self.check_overlap(current_bounds, bounds)
                for bounds in visible_bounds
            )

            if overlaps:
                label.setVisible(False)
                hidden_count += 1
            else:
                visible_bounds.append(current_bounds)

        return hidden_count


def generate_labeltext(
        mz_value: str,
        adduct_label: str,
) -> str:
    """
    Convenience function for HTML formatting
    """
    # return (f'<div style="text-align: center;"><b>{adduct_label}</b><br>'
    #         f'{mz_value}</div>')
    return (f'<div style="text-align: center;"><b>{adduct_label}</b><br>'
            f'{mz_value}</div>')


def create_textitem(
        text: str,
        pos: tuple[float, float],
        level: float = 1.0,
        on_click=None,
) -> pg.TextItem:
    """
    Convenience function.
    `level` describes where to set the anchor, for multi-line labels
    For example, setting `level` = 2 means this textitem will stack over
    ones with `level` = 1.

    If `on_click` is provided, returns a `ClickableTextItem` that calls
    the callback (with the Qt button as int) on mouse press.
    """
    anchor = (0.5, level + 0.05)
    if on_click is not None:
        textitem = ClickableTextItem(html=text, anchor=anchor, on_click=on_click)
    else:
        textitem = pg.TextItem(html=text, anchor=anchor)
    textitem.setPos(pos[0], pos[1])
    textitem = center_textitem(textitem)
    return textitem


def center_textitem(
        textitem: pg.TextItem,
) -> pg.TextItem:
    """
    Bizarre hoops that must be jumped to center-align HTML labels
    in pyqtgraph.
    From https://stackoverflow.com/a/62602065
    """
    it = textitem.textItem
    option = it.document().defaultTextOption()
    option.setAlignment(QtCore.Qt.AlignCenter)
    it.document().setDefaultTextOption(option)
    it.setTextWidth(it.boundingRect().width())

    return textitem