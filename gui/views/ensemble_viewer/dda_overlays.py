"""
DDA decorations for the EnsembleViewer's MS1 and MS2 spectrum plots.

- MS1 plot: a red-diamond precursor badge per precursor co-feature
  that was sampled for MS2. When a co-feature was fragmented more than
  once, the badge carries a "(N)" count label, and "(N:k)" while the
  k-th of those N spectra is the one on display. Clicking a badge steps
  to the next MS2 scan of that co-feature (wrapping), which routes back
  through the existing populate path.
- MS2 plot: isolation-window region (or vertical marker, if the file
  doesn't encode an isolation width) + a header label showing
  precursor m/z / charge / RT / scan # for the currently-displayed
  MS2 scan.

Symbol and colour conventions match SampleViewer: red diamonds for
precursors, blue for the isolation window.
"""
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.data_structs import Ensemble, ScanArray
    from gui.widgets.MSPlotWidget import MSPlotWidget
    from gui.widgets.ChromPlotWidget import ChromPlotWidget


_BADGE_BRUSH = pg.mkBrush(color=(220, 40, 40, 220))
_BADGE_PEN = pg.mkPen(color=(120, 0, 0), width=1)
_BADGE_LABEL_COLOR = "#FF0000"
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
        self._badge_labels: list[pg.TextItem] = []
        self._isolation_item = None  # LinearRegionItem or InfiniteLine

        # One entry per rendered badge: the sorted array of MS2 scan
        # indices belonging to that precursor co-feature. Used to
        # translate clicks back to scans (and to cycle through them).
        self._badge_targets: list[np.ndarray] = []

        # MS2 scan index currently on display, so badge labels can mark
        # which of a co-feature's repeats is being viewed, and clicks
        # can advance to the next one.
        self._current_ms2_scan_idx: Optional[int] = None

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

        ms2_scan_idx = int(ms2_arr.rt_to_scan_num(scan_rt))
        self._current_ms2_scan_idx = ms2_scan_idx

        self._render_ms1_badges(ms2_arr, ms2_scan_idx)
        self._render_ms2_overlay(ms2_arr, ms2_scan_idx)

    def clear(self) -> None:
        if self._badges is not None:
            self.ms1_plot.pi.removeItem(self._badges)
            self._badges = None
        for label in self._badge_labels:
            self.ms1_plot.pi.removeItem(label)
        self._badge_labels = []
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

    def _render_ms1_badges(
        self,
        ms2_arr: 'ScanArray',
        current_ms2_scan_idx: int,
    ) -> None:
        # All MS2 cofeatures in a DDA ensemble share the same scan_idxs
        # (built that way in `_dda_link_ms2_cofeatures`). Take the first.
        if not self.ensemble.ms2_cofeatures:
            return

        scan_idxs = self.ensemble.ms2_cofeatures[0].scan_idxs
        if scan_idxs.size == 0:
            return

        # Anchor each badge to the intensity of the nearest peak in the
        # currently-displayed MS1 spectrum. Badges sit on top of the
        # parent ion when it's visible; rest at the baseline otherwise.
        spec_array = self.ms1_plot.pi.spectrum_array
        if spec_array is None or len(spec_array) == 0:
            return
        spec_mzs = spec_array['mz']
        spec_intsys = spec_array['intsy']

        # Collapse repeated MS2 events on the same precursor into one
        # badge each, so the diamond count reflects co-features rather
        # than raw scans.
        groups = self._group_scans_by_precursor(scan_idxs, ms2_arr)

        xs = np.empty(len(groups), dtype='f8')
        ys = np.empty(len(groups), dtype='f4')
        for i, (rep_mz, _members) in enumerate(groups):
            nearest = int(np.argmin(np.abs(spec_mzs - rep_mz)))
            xs[i] = rep_mz
            ys[i] = spec_intsys[nearest]

        scatter = pg.ScatterPlotItem(
            x=xs,
            y=ys,
            symbol='d',
            size=14,
            brush=_BADGE_BRUSH,
            pen=_BADGE_PEN,
            hoverable=True,
        )
        scatter.sigClicked.connect(self._on_badge_clicked)
        # Suppress the default pyqtgraph context menu (export/copy etc.)
        # that otherwise opens on right-click.
        scatter.getContextMenus = lambda event=None: None
        # Hand cursor on hover signals clickability.
        scatter.setCursor(QtCore.Qt.PointingHandCursor)

        self._badge_targets = [members for _, members in groups]

        self.ms1_plot.pi.addItem(scatter)
        self._badges = scatter

        # "(N)" / "(N:k)" labels — only where a co-feature was sampled
        # for MS2 more than once.
        for i, (_rep_mz, members) in enumerate(groups):
            n = len(members)
            if n <= 1:
                continue
            where = np.where(members == current_ms2_scan_idx)[0]
            pos_in_group = int(where[0]) + 1 if where.size else None
            label = pg.TextItem(anchor=(-0.15, 0.5))
            label.setHtml(self._badge_label_html(n, pos_in_group))
            label.setPos(float(xs[i]), float(ys[i]))
            label.setZValue(20)
            self.ms1_plot.pi.addItem(label)
            self._badge_labels.append(label)

    def _group_scans_by_precursor(
        self,
        scan_idxs: np.ndarray,
        ms2_arr: 'ScanArray',
    ) -> list[tuple[float, np.ndarray]]:
        """
        Group matched MS2 scan indices by the precursor co-feature they
        belong to, returning (representative_mz, sorted scan_idxs) per
        group, ordered by m/z.

        Scans are assigned to the nearest MS1 cofeature m/z — the same
        relationship `_dda_link_ms2_cofeatures` used to match them — so
        the grouping mirrors the ensemble's own notion of a co-feature.
        """
        scan_idxs = np.sort(np.asarray(scan_idxs))
        prec_mzs = np.asarray(ms2_arr.precursor_mz_arr[scan_idxs], dtype='f8')

        ms1_cofeatures = self.ensemble.ms1_cofeatures
        if ms1_cofeatures:
            ms1_arr = self.ensemble.injection.scan_array_ms1
            cofeature_mzs = np.asarray(
                ms1_arr.mz_lane_label[
                    [cf.mz_lane_idx for cf in ms1_cofeatures]
                ],
                dtype='f8',
            )
            group_keys = np.argmin(
                np.abs(prec_mzs[:, None] - cofeature_mzs[None, :]),
                axis=1,
            )
        else:
            # No MS1 cofeatures to anchor to: group by exact precursor.
            group_keys = prec_mzs

        groups: list[tuple[float, np.ndarray]] = []
        for key in np.unique(group_keys):
            members = scan_idxs[group_keys == key]
            rep_mz = float(np.median(ms2_arr.precursor_mz_arr[members]))
            groups.append((rep_mz, members))

        groups.sort(key=lambda g: g[0])
        return groups

    @staticmethod
    def _badge_label_html(n: int, pos_in_group: Optional[int]) -> str:
        if pos_in_group is None:
            inner = str(n)
        else:
            inner = f"{n}:<b>{pos_in_group}</b>"
        return (
            f'<span style="color:{_BADGE_LABEL_COLOR};">({inner})</span>'
        )

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

        members = self._badge_targets[idx]
        ms2_arr = self.ensemble.injection.scan_array_ms2

        # If a scan in this co-feature is already on display, advance to
        # the next one (wrapping); otherwise start at the first. This is
        # how the user steps through repeated MS2 events now that they
        # share a single badge.
        ms2_scan_idx = int(members[0])
        if self._current_ms2_scan_idx is not None:
            where = np.where(members == self._current_ms2_scan_idx)[0]
            if where.size:
                nxt = (int(where[0]) + 1) % len(members)
                ms2_scan_idx = int(members[nxt])

        target_rt = float(ms2_arr.rt_arr[ms2_scan_idx])

        if self.on_select_rt is not None:
            self.on_select_rt(target_rt)
        else:
            # Fallback: just move the indicator and rely on its
            # sigPositionChanged hookup.
            self.chrom_plot.pi.selection_indicator.setPos(target_rt)
