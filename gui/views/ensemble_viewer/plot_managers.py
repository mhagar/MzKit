"""
Plot manager classes for organizing spectrum and chromatogram plotting logic
"""
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore

from gui.views.ensemble_viewer.utils import (
    match_chrom_arrs,
    normalize_chrom_arr,
    diff_chrom_arr,
    get_pearson_coeff,
)

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
        self.selected_rt: float = 0.0

        self.ms1_chroms: list[np.ndarray] = []
        self.ms2_chroms: list[np.ndarray] = []

        # Transform settings
        self._normalize_enabled = False
        self._diff_enabled = False

    def set_ensemble(self, ensemble: 'Ensemble'):
        """Set the ensemble and extract base chromatogram"""
        self.ensemble = ensemble
        self.base_chrom = self.ensemble.get_base_chromatogram(ms_level=1)

    def set_transform_settings(self, normalize: bool, diff: bool):
        """Update transform settings"""
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
            self._apply_transforms(self.base_chrom)
        )

        # Configure slider selector
        self.chrom_plot.setSliderSelectorMovable(True)
        self.chrom_plot.setSelectionIndicatorVisible(True)
        self.chrom_plot.setSelectionIndicator(xpos=peak_rt)
        self.selected_rt = peak_rt

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

    def update_correlation_plot(self):
        """
        Populates a 'correlation plot' where X is the
        intensity of the base cofeature, and Y is the intensity of the
        target cofeature
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

        for ms_level in (1, 2):
            ms_level: Literal[1, 2]

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

                if ref_arr is None:
                    # No overlap for some reason??
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

    def set_ms1_chroms(self, chroms: list[np.ndarray]):
        """Set MS1 chromatograms for overlay"""
        self.ms1_chroms = chroms

    def set_ms2_chroms(self, chroms: list[np.ndarray]):
        """Set MS2 chromatograms for overlay"""
        self.ms2_chroms = chroms

    def _apply_transforms(self, chrom: np.ndarray) -> np.ndarray:
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
    sigMS1SignalClicked = QtCore.pyqtSignal(tuple)  # (spec_idx, mz)
    sigMS2SignalClicked = QtCore.pyqtSignal(tuple)  # (spec_idx, mz)

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

        # Connect internal signal forwarding
        self.ms1_plot.sigMSSignalClicked.connect(
            self._on_ms1_clicked
        )
        self.ms2_plot.sigMSSignalClicked.connect(
            self._on_ms2_clicked
        )

    def set_ensemble(self, ensemble: 'Ensemble'):
        """Set the ensemble"""
        self.ensemble = ensemble

    def populate_spectrum_plot(self, scan_rt: float):
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
            plot.setSpectrumArray(
                self.ensemble.get_spectrum(
                    ms_level=ms_level,
                    scan_rt=scan_rt,
                )
            )

    def add_signal_marker(
        self,
        spec_idx: int,
        ms_level: Literal[1, 2],
    ):
        """Add a marker to indicate signal selection"""
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
        """Remove a signal selection marker"""
        plot_widget: 'MSPlotWidget' = {
            1: self.ms1_plot,
            2: self.ms2_plot,
        }.get(ms_level)

        if plot_widget:
            plot_widget.remove_signal_marker(spec_idx=spec_idx)

    def clear_signal_markers(self):
        """Clear all signal selection markers from both plots"""
        self.ms1_plot.clear_signal_markers()
        self.ms2_plot.clear_signal_markers()

    def _on_ms1_clicked(self, data: tuple[int, float]):
        """Internal handler for MS1 clicks"""
        self.sigMS1SignalClicked.emit(data)

    def _on_ms2_clicked(self, data: tuple[int, float]):
        """Internal handler for MS2 clicks"""
        self.sigMS2SignalClicked.emit(data)
