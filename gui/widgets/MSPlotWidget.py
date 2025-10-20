
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
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QRectF

from typing import Optional


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
        self.setBackground(None)
        self.pi : MSPlotItem = self.getPlotItem()

        self.sigSpectrumArrayChanged.connect(
            self.pi.updateSpectrumPlot
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

        # Set default extraction mode to be NONE
        self.on_xic_mode_changed(XICMode.NONE)

        # Local copy of tool state
        self._tool_type: 'ToolType' = ToolType.NONE
        self._tool_stage: 'ToolStage' = ToolStage.IDLE
        self._xic_mode: 'XICMode' = XICMode.NONE

        # Hovered MS signal
        self.hovered_ms_signal: Optional[tuple[int, float]] = None


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
        print('adding signal marker')
        spectrum_array = self.pi.spectrum_array

        if spectrum_array is None:
            return

        mz = spectrum_array['mz'][spec_idx]
        intsy = spectrum_array['intsy'][spec_idx]

        signal_marker = pg.TargetItem(
            pos=(mz, intsy),
            movable=False,
            # symbol='t',
            # brush=pg.mkBrush('r'),
            # pen=pg.mkPen('r'),
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

        if spectrum_idx:
            self.hovered_ms_signal = (spectrum_idx, mz)
            self.sigMSSignalHovered.emit(self.hovered_ms_signal)

        else:
            self.hovered_ms_signal = None
            self.sigMSpectrumLeaved.emit()
            self.unsetCursor()


    def MSSignalClicked(
        self,
    ):
        self.sigMSSignalClicked.emit(self.hovered_ms_signal)

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


    def scaleViewboxToSpectrumArray(
            self,
    ):
        """
        Scales viewbox to fit the spectrum array
        """
        if not self.spectrum_array:
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
        self.setLimits(xMin=0.0)

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


class MSLabelManager:
    """
    Used to show/hide MS labels to resolve overlaps between labels,
    as well as overlaps between MS data
    """
    def __init__(
            self,
            plotitem: 'MSPlotItem',
            clearance: float = 5.0,  # Pixels (I think?)
            intsy_threshold: float = 0.01, # Relative to max intsy
    ):
        self.plotitem: 'MSPlotItem' = plotitem
        self.priority_key = None
        self.labels: list[pg.TextItem] = []

        self.clearance = clearance
        self.intsy_threshold = intsy_threshold
        self.absolute_intsy_threshold = 0
        self.peak_data: Optional[SpectrumArray] = None

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
        self.absolute_intsy_threshold = (
                data['intsy'].max() * self.intsy_threshold
        )

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
        if not self.labels:
            return

        # Reset visibility
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
) -> pg.TextItem:
    """
    Convenience function.
    `level` describes where to set the anchor, for multi-line labels
    For example, setting `level` = 2 means this textitem will stack over
    ones with `level` = 1.
    """
    textitem = pg.TextItem(
        # text=str(text),
        html=text,
        anchor=(
            0.5,                # X
            level + 0.05,       # Y
        ),
    )
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