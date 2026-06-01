"""
DDA-specific overlay layer for the SampleViewer's spectrum plot.

- On MS1: precursor badges over the m/z of each MS2 scan that was triggered
  from the currently-displayed MS1 scan. Clicking a badge jumps the
  selection to that MS2 scan.
- On MS2: isolation-window shaded region + header text showing precursor
  m/z, charge, RT, and scan number.
"""
import numpy as np
import pyqtgraph as pg

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.data_structs import SampleUUID, ScanArray
    from core.interfaces.data_sources import SampleDataSource
    from gui.widgets.MSPlotWidget import MSPlotWidget
    from gui.views.sample_viewer.spectrum_selection import SelectionManager


# MS-community convention: red diamonds for precursor markers,
# blue for the isolation window.
_BADGE_BRUSH = pg.mkBrush(color=(220, 40, 40, 220))
_BADGE_PEN = pg.mkPen(color=(120, 0, 0), width=1)
_ISOLATION_BRUSH = pg.mkBrush(color=(50, 150, 250, 60))
_ISOLATION_PEN = pg.mkPen(color=(50, 150, 250, 180), width=1)
# Width below which isolation_hi - isolation_lo is treated as "no window
# encoded" (Waters DDA, some Thermo exports). Falls back to a vertical
# marker at the precursor m/z.
_ISOLATION_MIN_WIDTH = 1e-4


class DDAOverlayManager:
    """
    Owns the per-spectrum DDA decorations on a single MSPlotWidget.
    """

    def __init__(
        self,
        plot: 'MSPlotWidget',
        selection_mgr: 'SelectionManager',
        data_source: 'SampleDataSource',
        tool_mgr=None,
    ):
        self.plot = plot
        self.selection_mgr = selection_mgr
        self.data_source = data_source
        # Badge clicks only fire when no tool is active (ToolType.NONE) —
        # otherwise they fight with ensemble-extraction / XIC selection.
        self.tool_mgr = tool_mgr

        self._badges: Optional[pg.ScatterPlotItem] = None
        self._isolation_region: Optional[pg.LinearRegionItem] = None

        # Parallel arrays of (precursor_mz, ms2_scan_array_index) for the
        # currently-rendered badges. Index lookup on click.
        self._badge_targets: list[tuple[float, int]] = []
        self._current_sample_uuid: Optional['SampleUUID'] = None

    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        if self._badges is not None:
            self.plot.pi.removeItem(self._badges)
            self._badges = None
        if self._isolation_region is not None:
            self.plot.pi.removeItem(self._isolation_region)
            self._isolation_region = None
        self._badge_targets = []

    def update(
        self,
        sample_uuid: 'SampleUUID',
        ms_level: int,
        scan_index: int,
    ) -> None:
        """
        Re-render overlays for the currently-displayed scan.

        :param scan_index: index into the active ScanArray's per-scan arrays
            (same semantics as SelectionManager's "scan_num" — NOT the
            original mzML scan number).
        """
        self.clear()
        self._current_sample_uuid = sample_uuid

        injection = self.data_source.get_sample(sample_uuid).injection
        if injection is None or injection.acquisition_mode != 'dda':
            return

        ms2_arr = injection.scan_array_ms2
        if ms2_arr is None or ms2_arr.precursor_mz_arr is None:
            return

        if ms_level == 1:
            ms1_arr = injection.scan_array_ms1
            if ms1_arr is None:
                return
            self._render_ms1_badges(ms1_arr, ms2_arr, scan_index)
        elif ms_level == 2:
            self._render_ms2_overlay(ms2_arr, scan_index)

    # ------------------------------------------------------------------ #
    def _render_ms1_badges(
        self,
        ms1_arr: 'ScanArray',
        ms2_arr: 'ScanArray',
        ms1_scan_index: int,
    ) -> None:
        # Original mzML scan number for the MS1 scan under the cursor.
        ms1_mzml_scan = int(ms1_arr.scan_num_arr[ms1_scan_index])

        # Which MS2 scans were triggered from this MS1 scan?
        ms2_indices = np.where(
            ms2_arr.triggering_ms1_scan_arr == ms1_mzml_scan
        )[0]
        if ms2_indices.size == 0:
            return

        precursor_mzs = ms2_arr.precursor_mz_arr[ms2_indices]

        # Place badges at the intensity of the nearest MS1 peak to each
        # precursor m/z, so they appear sitting on the parent ion.
        spec_array = self.plot.pi.spectrum_array
        if spec_array is None or len(spec_array) == 0:
            return

        spec_mzs = spec_array['mz']
        spec_intsys = spec_array['intsy']

        ys = np.empty(precursor_mzs.size, dtype='f4')
        for i, prec_mz in enumerate(precursor_mzs):
            nearest = int(np.argmin(np.abs(spec_mzs - prec_mz)))
            ys[i] = spec_intsys[nearest]

        scatter = pg.ScatterPlotItem(
            x=precursor_mzs,
            y=ys,
            symbol='d',  # red diamond — MS-community convention
            size=14,
            brush=_BADGE_BRUSH,
            pen=_BADGE_PEN,
            hoverable=True,
        )
        scatter.sigClicked.connect(self._on_badge_clicked)

        self._badge_targets = [
            (float(prec_mz), int(idx))
            for prec_mz, idx in zip(precursor_mzs, ms2_indices)
        ]

        self.plot.pi.addItem(scatter)
        self._badges = scatter

    def _render_ms2_overlay(
        self,
        ms2_arr: 'ScanArray',
        ms2_scan_index: int,
    ) -> None:
        lo = ms2_arr.isolation_lo_arr[ms2_scan_index]
        hi = ms2_arr.isolation_hi_arr[ms2_scan_index]
        prec_mz = float(ms2_arr.precursor_mz_arr[ms2_scan_index])

        if (
            np.isfinite(lo) and np.isfinite(hi)
            and (hi - lo) > _ISOLATION_MIN_WIDTH
        ):
            # Real isolation window encoded — shaded band.
            region = pg.LinearRegionItem(
                values=(float(lo), float(hi)),
                movable=False,
                brush=_ISOLATION_BRUSH,
                pen=_ISOLATION_PEN,
            )
            region.setZValue(-10)
            self.plot.pi.addItem(region)
            self._isolation_region = region
        elif np.isfinite(prec_mz):
            # File has no isolation width (e.g. Waters DDA) — fall back to
            # a single vertical marker at the precursor m/z.
            marker = pg.InfiniteLine(
                pos=prec_mz,
                angle=90,
                movable=False,
                pen=_ISOLATION_PEN,
            )
            marker.setZValue(-10)
            self.plot.pi.addItem(marker)
            self._isolation_region = marker

        charge = int(ms2_arr.precursor_charge_arr[ms2_scan_index])
        rt = float(ms2_arr.rt_arr[ms2_scan_index])
        mzml_scan = int(ms2_arr.scan_num_arr[ms2_scan_index])

        charge_str = f"{charge:+d}" if charge else "?"
        self.plot.update_label(
            f"MS2  precursor m/z {prec_mz:.4f}  z={charge_str}  "
            f"RT={rt:.2f}  scan #{mzml_scan}"
        )

    # ------------------------------------------------------------------ #
    def _on_badge_clicked(
        self,
        scatter: pg.ScatterPlotItem,
        points,
    ) -> None:
        if len(points) == 0 or self._current_sample_uuid is None:
            return

        # Suppress badge clicks unless no tool is active; otherwise they
        # collide with ensemble extraction / XIC selection / spectrum-grab.
        if self.tool_mgr is not None:
            from gui.views.sample_viewer.tools import ToolType
            if self.tool_mgr.active_tool != ToolType.NONE:
                return

        # Same MS1-signal-click suppression as in the EnsembleViewer
        # overlay: a badge sits on a peak top, so the plot's
        # `hovered_ms_signal` is set; clearing it stops the trailing
        # `MSSignalClicked()` in `MSPlotWidget.mousePressEvent`.
        self.plot.hovered_ms_signal = None

        idx = points[0].index()
        if idx < 0 or idx >= len(self._badge_targets):
            return

        _, ms2_scan_array_index = self._badge_targets[idx]
        self.selection_mgr.set_selected_spectrum_by_scan_num(
            uuid=self._current_sample_uuid,
            ms_level=2,
            scan_num=ms2_scan_array_index,
        )
