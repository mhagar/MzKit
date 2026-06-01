"""
DDA decorations for the EnsembleViewer's MS1 and MS2 spectrum plots.

- MS1 plot: a red-diamond precursor badge per MS2 scan in the ensemble.
  Clicking a badge moves the chromatogram selection indicator to that
  MS2 scan's RT, which routes back through the existing populate path.
- MS2 plot: isolation-window region (or vertical marker, if the file
  doesn't encode an isolation width) + a header label showing
  precursor m/z / charge / RT / scan # for the currently-displayed
  MS2 scan.

Symbol and colour conventions match SampleViewer: red diamonds for
precursors, blue for the isolation window.
"""
import numpy as np
import pyqtgraph as pg

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.data_structs import Ensemble, ScanArray
    from gui.widgets.MSPlotWidget import MSPlotWidget
    from gui.widgets.ChromPlotWidget import ChromPlotWidget


_BADGE_BRUSH = pg.mkBrush(color=(220, 40, 40, 220))
_BADGE_PEN = pg.mkPen(color=(120, 0, 0), width=1)
_ISOLATION_BRUSH = pg.mkBrush(color=(50, 150, 250, 60))
_ISOLATION_PEN = pg.mkPen(color=(50, 150, 250, 180), width=1)
_ISOLATION_MIN_WIDTH = 1e-4


class EnsembleDDAOverlayManager:
    """
    Owns the per-ensemble DDA decorations on the EnsembleViewer's two
    spectrum plots. A no-op for ensembles whose injection isn't DDA.
    """

    def __init__(
        self,
        ms1_plot: 'MSPlotWidget',
        ms2_plot: 'MSPlotWidget',
        chrom_plot: 'ChromPlotWidget',
        on_select_rt=None,
    ):
        self.ms1_plot = ms1_plot
        self.ms2_plot = ms2_plot
        self.chrom_plot = chrom_plot
        # Callback(rt: float). The viewer wires this so a badge click
        # both moves the chrom cursor and triggers the same spectrum
        # refresh path that cursor-drag uses, in a single explicit step.
        self.on_select_rt = on_select_rt

        self.ensemble: Optional['Ensemble'] = None

        self._badges: Optional[pg.ScatterPlotItem] = None
        self._isolation_item = None  # LinearRegionItem or InfiniteLine

        # Parallel arrays of (precursor_mz, ms2_scan_array_index) for
        # each rendered badge — used to translate clicks back to scans.
        self._badge_targets: list[tuple[float, int]] = []

    # ------------------------------------------------------------------ #
    def set_ensemble(self, ensemble: 'Ensemble') -> None:
        """
        Bind a new ensemble. Clears any prior overlays.
        """
        self.clear()
        self.ensemble = ensemble

    def update(self, scan_rt: float) -> None:
        """
        Re-render overlays. Call after `SpectrumPlotManager
        .populate_spectrum_plot(scan_rt)`. The badges' Y position
        depends on the currently-displayed MS1 spectrum; the MS2
        overlay depends on which MS2 scan that RT resolves to.
        """
        self.clear()
        if not self._is_dda():
            return

        ms2_arr = self.ensemble.injection.scan_array_ms2
        if ms2_arr is None or ms2_arr.precursor_mz_arr is None:
            return

        self._render_ms1_badges(ms2_arr)

        ms2_scan_idx = int(ms2_arr.rt_to_scan_num(scan_rt))
        self._render_ms2_overlay(ms2_arr, ms2_scan_idx)

    def clear(self) -> None:
        if self._badges is not None:
            self.ms1_plot.pi.removeItem(self._badges)
            self._badges = None
        if self._isolation_item is not None:
            self.ms2_plot.pi.removeItem(self._isolation_item)
            self._isolation_item = None
        self._badge_targets = []

    # ------------------------------------------------------------------ #
    def _is_dda(self) -> bool:
        return (
            self.ensemble is not None
            and self.ensemble.injection is not None
            and self.ensemble.injection.acquisition_mode == 'dda'
        )

    def _render_ms1_badges(self, ms2_arr: 'ScanArray') -> None:
        # All MS2 cofeatures in a DDA ensemble share the same scan_idxs
        # (built that way in `_dda_link_ms2_cofeatures`). Take the first.
        if not self.ensemble.ms2_cofeatures:
            return

        scan_idxs = self.ensemble.ms2_cofeatures[0].scan_idxs
        if scan_idxs.size == 0:
            return

        precursor_mzs = ms2_arr.precursor_mz_arr[scan_idxs]

        # Anchor each badge to the intensity of the nearest peak in the
        # currently-displayed MS1 spectrum. Badges sit on top of the
        # parent ion when it's visible; rest at the baseline otherwise.
        spec_array = self.ms1_plot.pi.spectrum_array
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
            symbol='d',
            size=14,
            brush=_BADGE_BRUSH,
            pen=_BADGE_PEN,
            hoverable=True,
        )
        scatter.sigClicked.connect(self._on_badge_clicked)

        self._badge_targets = [
            (float(prec_mz), int(idx))
            for prec_mz, idx in zip(precursor_mzs, scan_idxs)
        ]

        self.ms1_plot.pi.addItem(scatter)
        self._badges = scatter

    def _render_ms2_overlay(
        self,
        ms2_arr: 'ScanArray',
        ms2_scan_idx: int,
    ) -> None:
        lo = ms2_arr.isolation_lo_arr[ms2_scan_idx]
        hi = ms2_arr.isolation_hi_arr[ms2_scan_idx]
        prec_mz = float(ms2_arr.precursor_mz_arr[ms2_scan_idx])

        if (
            np.isfinite(lo) and np.isfinite(hi)
            and (hi - lo) > _ISOLATION_MIN_WIDTH
        ):
            region = pg.LinearRegionItem(
                values=(float(lo), float(hi)),
                movable=False,
                brush=_ISOLATION_BRUSH,
                pen=_ISOLATION_PEN,
            )
            region.setZValue(-10)
            self.ms2_plot.pi.addItem(region)
            self._isolation_item = region
        elif np.isfinite(prec_mz):
            marker = pg.InfiniteLine(
                pos=prec_mz,
                angle=90,
                movable=False,
                pen=_ISOLATION_PEN,
            )
            marker.setZValue(-10)
            self.ms2_plot.pi.addItem(marker)
            self._isolation_item = marker

        charge = int(ms2_arr.precursor_charge_arr[ms2_scan_idx])
        rt = float(ms2_arr.rt_arr[ms2_scan_idx])
        mzml_scan = int(ms2_arr.scan_num_arr[ms2_scan_idx])
        charge_str = f"{charge:+d}" if charge else "?"
        self.ms2_plot.update_label(
            f"MS2  precursor m/z {prec_mz:.4f}  z={charge_str}  "
            f"RT={rt:.2f}  scan #{mzml_scan}"
        )

    # ------------------------------------------------------------------ #
    def _on_badge_clicked(
        self,
        scatter: pg.ScatterPlotItem,
        points,
    ) -> None:
        # `points` is a numpy array; cast length explicitly.
        if len(points) == 0 or not self._is_dda():
            return
        idx = points[0].index()
        if idx < 0 or idx >= len(self._badge_targets):
            return

        # The MS1 plot's `mousePressEvent` fires `MSSignalClicked()`
        # whenever `hovered_ms_signal` is set — and the cursor is
        # always "over" a peak while hovering a badge (badges sit on
        # peak tops). Clear it so the trailing click no-ops and we
        # don't also trigger the MS1 signal-selection side effects.
        self.ms1_plot.hovered_ms_signal = None

        _, ms2_scan_idx = self._badge_targets[idx]
        ms2_arr = self.ensemble.injection.scan_array_ms2
        target_rt = float(ms2_arr.rt_arr[ms2_scan_idx])

        if self.on_select_rt is not None:
            self.on_select_rt(target_rt)
        else:
            # Fallback: just move the indicator and rely on its
            # sigPositionChanged hookup.
            self.chrom_plot.pi.selection_indicator.setPos(target_rt)
