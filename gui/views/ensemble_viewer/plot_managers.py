"""
Plot manager classes for organizing spectrum and chromatogram plotting logic
"""
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from scipy.stats import pearsonr, spearmanr

from core.utils.array_types import to_spec_arr

from gui.views.ensemble_viewer.utils import (
    match_chrom_arrs,
    normalize_chrom_arr,
    diff_chrom_arr,
    get_pearson_coeff,
)

from numpy.typing import NDArray
from typing import TYPE_CHECKING, Literal, Optional
if TYPE_CHECKING:
    from core.data_structs import Ensemble
    from gui.widgets.MSPlotWidget import MSPlotWidget
    from gui.widgets.ChromPlotWidget import ChromPlotWidget


class ChromatogramPlotManager(QtCore.QObject):
    """
    Manages chromatogram plotting, transforms, and correlation plots
    """
    def __init__(
        self,
        chrom_plot_widget: 'ChromPlotWidget',
        corr_plot_widget: 'pg.PlotWidget',
    ):
        super().__init__()
        self.chrom_plot = chrom_plot_widget
        self.corr_plot = corr_plot_widget

        self.ensemble: Optional['Ensemble'] = None
        self.base_chrom: Optional[np.ndarray] = None
        self.background_chrom: Optional[np.ndarray] = None
        self.selected_rt: float = 0.0

        self.ms1_chroms: list[np.ndarray] = []
        self.ms2_chroms: list[np.ndarray] = []

        # Most-recently-clicked feature (drives correlation plot readout).
        self._last_selected_mz: Optional[float] = None
        self._last_selected_ms_level: Optional[Literal[1, 2]] = None

        # Transform settings
        self._normalize_enabled = False
        self._diff_enabled = False

        # Downward-triangle markers shown on the BPC trace at every RT
        # that carries at least one annotation. Helps users find
        # annotated scans without scrubbing the slider blind. Stored per
        # ms_level so MS1 vs MS2 can be coloured differently.
        self._annotation_markers: dict[int, pg.ScatterPlotItem] = {}
        # Last-set RT lists keyed by ms_level, so we can re-render the
        # markers in-place when the BPC transform changes (normalize /
        # diff toggles) without the viewer needing to re-collect.
        self._annotation_marker_rts: dict[int, list[float]] = {1: [], 2: []}

    def set_ensemble(self, ensemble: 'Ensemble'):
        """
        Set the ensemble and extract ensemble base chromatogram,
        and 'background' chromatogram (i.e. from injection)
        """
        self.ensemble = ensemble

        self.base_chrom = self.ensemble.get_base_chromatogram(
            ms_level=1
        )

        self.background_chrom = self.ensemble.injection.scan_array_ms1.get_bpc()

    def clear(self):
        """
        Reset all chromatogram/correlation state and wipe both plots, so
        no curves or markers from a previous ensemble remain rendered.
        """
        self.ensemble = None
        self.base_chrom = None
        self.background_chrom = None
        self.selected_rt = 0.0
        self.ms1_chroms = []
        self.ms2_chroms = []
        self._last_selected_mz = None
        self._last_selected_ms_level = None
        self._annotation_marker_rts = {1: [], 2: []}

        # Wipe overlay peaks, transient highlights and the BPC trace.
        self.chrom_plot.clearPeaks()
        self.chrom_plot.clearHighlights()
        self.chrom_plot.pi.clear_plots()

        # Remove annotation markers.
        for scatter in self._annotation_markers.values():
            self.chrom_plot.pi.removeItem(scatter)
        self._annotation_markers.clear()

        # Hide the scan selector and clear the title.
        self.chrom_plot.setSelectionIndicatorVisible(False)
        self.chrom_plot.update_label("")

        # Clear the correlation scatter plot.
        self.corr_plot.plotItem.clear()
        self.corr_plot.plotItem.setTitle(None)

    def _apply_selection_bounds(self):
        """
        Constrain the chrom plot's selection indicator to the ensemble's
        RT span (defined by the base cofeature's chromatogram).
        """
        if self.base_chrom is None or len(self.base_chrom) == 0:
            return
        rt_min = float(self.base_chrom['rt'].min())
        rt_max = float(self.base_chrom['rt'].max())
        self.chrom_plot.pi.selection_indicator.setBounds([rt_min, rt_max])

    def set_transform_settings(self, normalize: bool, diff: bool):
        """
        Update transform settings
        """
        self._normalize_enabled = normalize
        self._diff_enabled = diff

    def populate_chromatogram_plot(self, peak_rt: float):
        """
        Populate the chromatogram plot with base chromatogram
        and configure the selection indicator
        """
        if not self.ensemble:
            return

        # Set background chrom array
        self.chrom_plot.setChromArray(
            # self._apply_transforms(self.base_chrom)
            self._apply_transforms(self.background_chrom)
        )

        # Configure slider selector
        self.chrom_plot.setSliderSelectorMovable(True)
        self.chrom_plot.setSelectionIndicatorVisible(True)
        self.chrom_plot.setSelectionIndicator(xpos=peak_rt)
        self.selected_rt = peak_rt

        # Lock dragging to within the ensemble's RT span so the cursor
        # can't wander into scans that don't belong to this ensemble.
        self._apply_selection_bounds()

        # Add title
        self.chrom_plot.update_label(
            text=self.ensemble.format_string
        )

        # The plot rebuild above wiped any prior annotation markers; put
        # them back, now anchored to the (possibly transformed) BPC.
        self._render_annotation_markers()

    def update_chromatogram_plot(self):
        """
        Update the chromatogram plot with MS1/MS2 signal overlays
        """
        self.chrom_plot.clearPeaks()

        colors = {
            1: 'm',
            2: 'g',
        }

        # Add selected MS1 signals if there are any
        for idx, chrom in enumerate(self.ms1_chroms):
            self.chrom_plot.addPeak(
                chrom=self._apply_transforms(chrom),
                uuid=idx+1,
                color=colors[1],
            )

        # Add selected MS2 signals if there are any
        for idx, chrom in enumerate(self.ms2_chroms):
            idx += len(self.ms1_chroms)
            self.chrom_plot.addPeak(
                chrom=self._apply_transforms(chrom),
                uuid=idx+1,
                color=colors[2],
            )

    # Brush/pen per ms_level for annotation markers. Matches the
    # MS1/MS2 convention used by `update_correlation_plot`.
    _MARKER_COLORS: dict[int, tuple[tuple, tuple]] = {
        1: ((220, 60, 220, 230), (110, 30, 110, 230)),   # magenta-ish (MS1)
        2: ((40, 180, 60, 230),  (15, 90, 30, 230)),     # green (MS2)
    }

    def set_annotation_markers(
        self,
        rts_by_level: dict[int, list[float]],
    ):
        """
        Render down-triangle markers on the BPC trace at every RT in
        `rts_by_level[ms_level]`. Y values are interpolated from the
        currently-displayed (transformed) background chromatogram, so
        markers sit on the trace and follow normalize / diff toggles.
        Colored per ms_level (MS1 = magenta, MS2 = green).
        """
        # Cache so transform-toggle handlers can re-render in place.
        self._annotation_marker_rts = {
            1: list(rts_by_level.get(1, [])),
            2: list(rts_by_level.get(2, [])),
        }
        self._render_annotation_markers()

    def _render_annotation_markers(self):
        """
        (Re)draw markers from `self._annotation_marker_rts` using the
        currently-displayed transformed BPC. Called on annotation
        changes and on transform-setting changes.
        """
        # Wipe the prior layer(s).
        for scatter in self._annotation_markers.values():
            self.chrom_plot.pi.removeItem(scatter)
        self._annotation_markers.clear()

        if self.background_chrom is None or len(self.background_chrom) == 0:
            return

        transformed = self._apply_transforms(self.background_chrom)
        bpc_rt = transformed['rt']
        bpc_intsy = transformed['intsy']
        if bpc_rt.size == 0:
            return

        # Ensure rt array is sorted ascending for np.interp.
        if not np.all(np.diff(bpc_rt) >= 0):
            order = np.argsort(bpc_rt)
            bpc_rt = bpc_rt[order]
            bpc_intsy = bpc_intsy[order]

        for ms_level, rts in self._annotation_marker_rts.items():
            if not rts:
                continue
            ys = np.interp(np.asarray(rts, dtype=float), bpc_rt, bpc_intsy)
            brush_rgba, pen_rgba = self._MARKER_COLORS[ms_level]
            scatter = pg.ScatterPlotItem(
                x=list(rts),
                y=list(ys),
                symbol='t1',  # down-triangle
                size=10,
                brush=pg.mkBrush(*brush_rgba),
                pen=pg.mkPen(*pen_rgba),
            )
            # Don't grab right-clicks (would pop pyqtgraph's menu).
            scatter.getContextMenus = lambda event=None: None
            self.chrom_plot.pi.addItem(scatter)
            self._annotation_markers[ms_level] = scatter

    def set_last_selection(
        self,
        mz: float,
        ms_level: Literal[1, 2],
    ):
        """
        Record the most-recently-clicked feature's m/z and MS level so
        the correlation plot can show the seed-vs-target readout.
        """
        self._last_selected_mz = mz
        self._last_selected_ms_level = ms_level

    def _is_dda(self) -> bool:
        return bool(
            self.ensemble
            and self.ensemble.injection
            and self.ensemble.injection.acquisition_mode == 'dda'
        )

    def update_correlation_plot(self):
        """
        Populates a 'correlation plot' where X is the
        intensity of the base cofeature, and Y is the intensity of the
        target cofeature.

        For DDA injections, the MS2 series is suppressed — DDA MS2
        features were never grouped via correlation, so plotting them
        here is misleading.
        """
        self.corr_plot.plotItem.clear()

        # Add a line for perfect correlation
        self.corr_plot.addItem(
            pg.InfiniteLine(
                angle=45,
                movable=False,
            )
        )

        colors = {
            1: (255, 0, 255),  # Magenta
            2: (0, 255, 0),    # Green
        }

        # Get base feature chrom
        base_chrom = self.ensemble.get_base_chromatogram(ms_level=1)

        is_dda = self._is_dda()
        ms_levels: tuple[Literal[1, 2], ...] = (1,) if is_dda else (1, 2)

        # Collected for R² computation (uses the most recent overlap pair)
        last_ref_intsy: Optional[np.ndarray] = None
        last_tgt_intsy: Optional[np.ndarray] = None

        for ms_level in ms_levels:
            chroms = {
                1: self.ms1_chroms,
                2: self.ms2_chroms,
            }[ms_level]

            for chrom in chroms:
                ref_arr, tgt_arr = match_chrom_arrs(
                    reference_chrom=base_chrom, # type: ignore
                    target_chrom=chrom, # type: ignore
                    normalize=True,
                )

                if ref_arr is None or len(ref_arr) == 0:
                    continue

                pen = pg.mkPen(
                    *colors[ms_level],
                )

                scatter = pg.ScatterPlotItem(
                    tgt_arr['intsy'], ref_arr['intsy'],
                    pen=pen
                )

                self.corr_plot.addItem(
                    scatter
                )

                last_ref_intsy = ref_arr['intsy']
                last_tgt_intsy = tgt_arr['intsy']

        self._update_correlation_title(last_ref_intsy, last_tgt_intsy)

    def _update_correlation_title(
        self,
        ref_intsy: Optional[np.ndarray],
        tgt_intsy: Optional[np.ndarray],
    ):
        """
        Build the corr plot's title from the seed m/z, the selected m/z,
        and Pearson + Spearman R² over the matched intensity arrays.
        """
        if not self.ensemble:
            self.corr_plot.plotItem.setTitle(None)
            return

        seed_mz = self.ensemble.base_mz

        if self._last_selected_mz is None:
            self.corr_plot.plotItem.setTitle(
                f"Seed m/z {seed_mz:.4f}: click a signal to populate"
            )
            return

        sel_mz = self._last_selected_mz
        sel_level = self._last_selected_ms_level or 1

        pearson_r2_str = "-"
        spearman_r2_str = "-"
        if (
            ref_intsy is not None
            and tgt_intsy is not None
            and len(ref_intsy) >= 3
            and np.std(ref_intsy) > 0
            and np.std(tgt_intsy) > 0
        ):
            try:
                pr = float(pearsonr(ref_intsy, tgt_intsy).statistic)
                sr = float(spearmanr(ref_intsy, tgt_intsy).statistic)
                pearson_r2_str = f"{pr ** 2:.3f}"
                spearman_r2_str = f"{sr ** 2:.3f}"
            except (ValueError, FloatingPointError):
                pass

        self.corr_plot.plotItem.setTitle(
            f"Seed m/z {seed_mz:.4f}\t"
            f"MS{sel_level} m/z {sel_mz:.4f}\t"
            f"Pearson r2: {pearson_r2_str}\t"
            f"Spearman ρ2: {spearman_r2_str}\t"
        )

    def set_ms1_chroms(self, chroms: list[np.ndarray]):
        """
        Set MS1 chromatograms for overlay
        """
        self.ms1_chroms = chroms

    def set_ms2_chroms(self, chroms: list[np.ndarray]):
        """
        Set MS2 chromatograms for overlay
        """
        self.ms2_chroms = chroms

    def _apply_transforms(
        self,
        chrom: np.ndarray
    ) -> np.ndarray:
        """
        Applies transforms based on settings
        (i.e. normalize, diff)
        """
        _chrom = chrom.copy()
        if self._normalize_enabled:
            _chrom = normalize_chrom_arr(chrom)

        if self._diff_enabled:
            _chrom = diff_chrom_arr(chrom)

        return _chrom


class SpectrumPlotManager(QtCore.QObject):
    """
    Manages MS1 and MS2 spectrum plotting and signal selection graphics
    """
    sigMS1SignalClicked = QtCore.pyqtSignal(tuple)  # (mz, intsy, spec_idx)
    sigMS2SignalClicked = QtCore.pyqtSignal(tuple)  # (mz, intsy, spec_idx)
    sigMSSignalHovered = QtCore.pyqtSignal(tuple)  # (mz, intsy, spec_idx, level)

    def __init__(
        self,
        ms1_plot_widget: 'MSPlotWidget',
        ms2_plot_widget: 'MSPlotWidget',
    ):
        super().__init__()
        self.ms1_plot = ms1_plot_widget
        self.ms2_plot = ms2_plot_widget

        self.ensemble: Optional['Ensemble'] = None
        self.selected_rt: float = 0.0
        self._normalize_spectra: bool = False

        # Connect internal signal forwarding
        self.ms1_plot.sigMSSignalClicked.connect(
            self._on_ms1_clicked
        )
        self.ms2_plot.sigMSSignalClicked.connect(
            self._on_ms2_clicked
        )

        self.ms1_plot.sigMSSignalHovered.connect(
            self._on_ms1_hovered
        )

        self.ms2_plot.sigMSSignalHovered.connect(
            self._on_ms2_hovered
        )

    def set_ensemble(
        self,
        ensemble: 'Ensemble',
    ):
        self.ensemble = ensemble

    def clear(self):
        """
        Reset spectrum state and wipe both MS plots (including any signal
        markers and ion annotations) so nothing from a previous ensemble
        remains rendered.
        """
        self.ensemble = None
        self.selected_rt = 0.0
        self.clear_signal_markers()

        empty = to_spec_arr(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        for plot in (self.ms1_plot, self.ms2_plot):
            # setSpectrumArray also clears anchored labels / ion
            # annotations / delta brackets on the plot.
            plot.setSpectrumArray(empty)
            plot.update_bpc_label("")

    def set_normalize_spectra(self, enabled: bool):
        """
        Toggle per-spectrum intensity normalization (peaks rescaled so
        max intensity = 1.0). BPC readout still shows the un-normalized
        max intensity.
        """
        self._normalize_spectra = enabled

    def populate_spectrum_plot(
        self,
        scan_rt: float,
    ):
        """
        Given a retention time, plots the ensemble spectrum
        for both MS1 and MS2
        """
        self.selected_rt = scan_rt

        for ms_level, plot in zip(
            [1, 2],
            [self.ms1_plot, self.ms2_plot],
        ):
            ms_level: Literal[1, 2]
            spectrum = self.ensemble.get_spectrum(
                ms_level=ms_level,
                scan_rt=scan_rt,
            )

            intsy = spectrum['intsy']
            bpc = float(intsy.max()) if intsy.size else 0.0

            if self._normalize_spectra and bpc > 0:
                spectrum = spectrum.copy()
                spectrum['intsy'] = spectrum['intsy'] / bpc

            plot.setSpectrumArray(spectrum)
            plot.update_bpc_label(
                f"BPC: {bpc:.2e}" if bpc > 0 else ""
            )

    def add_signal_marker(
        self,
        spec_idx: int,
        ms_level: Literal[1, 2],
    ):
        """
        Add a marker to indicate signal selection
        """
        plot_widget: 'MSPlotWidget' = {
            1: self.ms1_plot,
            2: self.ms2_plot,
        }.get(ms_level)

        if plot_widget:
            plot_widget.add_signal_marker(spec_idx=spec_idx)

    def remove_signal_marker(
        self,
        spec_idx: int,
        ms_level: Literal[1, 2],
    ):
        """
        Remove a signal selection marker
        """
        plot_widget: 'MSPlotWidget' = {
            1: self.ms1_plot,
            2: self.ms2_plot,
        }.get(ms_level)

        if plot_widget:
            plot_widget.remove_signal_marker(spec_idx=spec_idx)

    def clear_signal_markers(self):
        """
        Clear all signal selection markers from both plots
        """
        self.ms1_plot.clear_signal_markers()
        self.ms2_plot.clear_signal_markers()

    def _on_ms1_clicked(
        self,
        data: tuple[int, float]  # spec_idx, mz
    ):
        """
        Internal handler for MS1 clicks. Converts the signal
        into from (spec_idx, mz) into (mz, intsy, spec_idx)
        """
        spec_idx, mz = data
        intsy = self._get_intsy(
            spec_idx=spec_idx,
            ms_level=1,
        )

        print(
            f"Clicked: {data}"
        )

        self.sigMS1SignalClicked.emit(
            (mz, intsy, spec_idx)
        )

    def _on_ms2_clicked(
        self,
        data: tuple[int, float]
    ):
        """
        Internal handler for MS2 clicks
        """
        spec_idx, mz = data
        intsy = self._get_intsy(
            spec_idx=spec_idx,
            ms_level=2,
        )

        self.sigMS2SignalClicked.emit(
            ( mz, intsy, spec_idx )
        )

    def _get_intsy(
        self,
        spec_idx: int,
        ms_level: Literal[1, 2],
    ) -> float:
        """
        Patching poor signal design that's baked in to SampleViewer lmao
        """
        spectrum: NDArray[float] = self.ensemble.get_spectrum(
            ms_level=ms_level,
            scan_rt=self.selected_rt
        )
        intsy: float = spectrum['intsy'][spec_idx]  # type: ignore
        return intsy

    def _on_ms1_hovered(
        self,
        data: tuple[int, float]
    ):
        """
        Internal handler for MS1 hovers. Converts the signal
        into from (spec_idx, mz) into (mz, intsy, spec_idx, level)
        """
        spec_idx, mz = data
        intsy = self._get_intsy(
            spec_idx=spec_idx,
            ms_level=1,
        )

        self.sigMSSignalHovered.emit(
            (mz, intsy, spec_idx, 1)
        )

    def _on_ms2_hovered(
        self,
        data: tuple[int, float]
    ):
        """
        Internal handler for MS1 hovers. Converts the signal
        into from (spec_idx, mz) into (mz, intsy, spec_idx, level)
        """
        spec_idx, mz = data
        intsy = self._get_intsy(
            spec_idx=spec_idx,
            ms_level=2,
        )

        self.sigMSSignalHovered.emit(
            (mz, intsy, spec_idx, 2)
        )