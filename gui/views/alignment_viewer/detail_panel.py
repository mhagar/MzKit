"""
Drives the AlignmentViewer's bottom panel (MS1, MS2 and chromatogram plots)
for the current selection:

 - Ensemble: composite MS1/MS2; its sample's BPC + the ensemble's XIC.
 - Analyte:  consensus MS1/MS2; member samples' BPCs + every member's XIC.
 - Compare:  mirror plots (anchor vs one other item); BPCs + XICs of all
             selected items.

Single spectra are shown normalised to their base peak. The MS1 and MS2
plots share an x-axis, which (like the chromatogram's y-axis) is only
framed once per alignment. The chromatogram x-axis can be linked to the feature
map's (RT link); its y-axis is only fitted once per alignment, so zooming in
on low-abundance signals isn't undone by panning or selecting.
"""
from typing import Optional, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

from core.utils.array_types import to_spec_arr
from core.utils.spectra import normalize_spectrum

if TYPE_CHECKING:
    from core.cli.align_ensembles import PairScore
    from core.data_structs import Ensemble, SampleUUID
    from core.data_structs.alignment import AlignedAnalyte
    from core.utils.array_types import SpectrumArray
    from gui.views.alignment_viewer.context import AlignmentContext, ItemSpectra
    from gui.widgets.AlignmentPlotWidget import AlignmentPlotWidget
    from gui.widgets.ChromPlotWidget import ChromPlotWidget
    from gui.widgets.MSPlotWidget import MSPlotWidget


# Unlinked chromatogram view: seconds of padding around the selected peaks.
CHROM_RT_PAD = 60.0
# BPC overlay opacity: each trace gets 1/n of full opacity, but at least this.
BPC_MIN_OPACITY = 0.2
BPC_GREY = (150, 150, 150)
XIC_COLOR = 'm'


def _empty_spectrum() -> 'SpectrumArray':
    return to_spec_arr(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
    )


def _set_label(plot: 'MSPlotWidget', text: str):
    """Set the MS plot's top-left floating label; empty text hides it."""
    if text:
        plot.update_label(text)
    else:
        plot.floating_label.hide()


def _bpc_label(spectrum: Optional['SpectrumArray']) -> str:
    if spectrum is None or not spectrum.size:
        return ""
    return f"BPC: {float(spectrum['intsy'].max()):.2e}"


class DetailPanel:
    def __init__(
        self,
        ms1_plot: 'MSPlotWidget',
        ms2_plot: 'MSPlotWidget',
        chrom_plot: 'ChromPlotWidget',
        map_plot: 'AlignmentPlotWidget',
    ):
        self.ms1_plot = ms1_plot
        self.ms2_plot = ms2_plot
        self.chrom_plot = chrom_plot
        self.map_plot = map_plot

        self._rt_linked = False
        self._bpc_cache: dict['SampleUUID', np.ndarray] = {}
        # (chromatograms drawn, RTs of the XIC'd ensembles) for re-ranging
        self._chroms: list[np.ndarray] = []
        self._focus_rts: list[float] = []
        # Chromatogram y is fitted once (first draw after a reset), then
        # left to the user
        self._chrom_y_fitted = False
        # Likewise the (shared) MS x-axis: framed once, then kept
        self._ms_x_fitted = False

        self.ms2_plot.setXLink(self.ms1_plot)

    # -- public -------------------------------------------------------------

    def reset(self):
        """Forget cached BPCs (new alignment / project) and clear plots."""
        self._bpc_cache.clear()
        self._chrom_y_fitted = False
        self._ms_x_fitted = False
        self.clear()

    def clear(self):
        for plot in (self.ms1_plot, self.ms2_plot):
            plot.setSpectrumArray(_empty_spectrum())
            _set_label(plot, "")
            plot.update_bpc_label("")
        self._draw_chroms(None, [], [], "")

    def set_rt_linked(self, linked: bool):
        self._rt_linked = linked
        self.chrom_plot.setXLink(self.map_plot if linked else None)
        self._fit_chrom_view()

    def show_ensemble(
        self,
        ctx: 'AlignmentContext',
        sample_uuid: 'SampleUUID',
        ensemble: 'Ensemble',
    ):
        composite = ensemble.composite_spectrum
        name = ctx.sample_name(sample_uuid)
        ms_x = self._ms_x_range
        self._show_single(
            self.ms1_plot, composite.ms1, f"Composite MS1 · {name}",
        )
        self._show_single(
            self.ms2_plot, composite.ms2, f"Composite MS2 · {name}",
        )
        self._frame_ms([composite.ms1, composite.ms2], ms_x)
        self._draw_chroms(ctx, [sample_uuid], [ensemble], f"BPC · {name}")

    def show_analyte(
        self,
        ctx: 'AlignmentContext',
        analyte: 'AlignedAnalyte',
    ):
        members = ctx.members(analyte)
        consensus = ctx.consensus(analyte)
        if consensus is None:
            self.clear()
            return

        source = ctx.sample_name(consensus.sample_uuid)
        ms_x = self._ms_x_range
        self._show_single(
            self.ms1_plot, consensus.composite.ms1,
            f"Consensus MS1 · from {source}",
        )
        self._show_single(
            self.ms2_plot, consensus.composite.ms2,
            f"Consensus MS2 · from {source}",
        )
        self._frame_ms([consensus.composite.ms1, consensus.composite.ms2], ms_x)
        self._draw_chroms(
            ctx, list(members), list(members.values()),
            f"BPC · {len(members)} samples",
        )

    def show_compare(
        self,
        ctx: 'AlignmentContext',
        top: 'ItemSpectra',
        bottom: 'ItemSpectra',
        score: 'PairScore',
        sample_uuids: list['SampleUUID'],
        ensembles: list['Ensemble'],
    ):
        ms_x = self._ms_x_range
        self.ms1_plot.set_mirror_spectra(
            top.ms1, bottom.ms1, top.label, bottom.label,
            matched=score.ms1_matches,
        )
        # Top-left is taken by the mirror labels; cosine goes top-right
        _set_label(self.ms1_plot, "")
        self.ms1_plot.update_bpc_label(f"MS1 cosine: {score.ms1:.3f}")

        if score.ms2 is not None:
            self.ms2_plot.set_mirror_spectra(
                top.ms2, bottom.ms2, top.label, bottom.label,
                matched=score.ms2_matches,
            )
            _set_label(self.ms2_plot, "")
            self.ms2_plot.update_bpc_label(f"MS2 cosine: {score.ms2:.3f}")
        else:
            self.ms2_plot.setSpectrumArray(_empty_spectrum())
            _set_label(self.ms2_plot, "MS2 cosine: n/a (MS2 missing)")
            self.ms2_plot.update_bpc_label("")
        self._frame_ms([top.ms1, bottom.ms1, top.ms2, bottom.ms2], ms_x)

        self._draw_chroms(
            ctx, sample_uuids, ensembles,
            f"BPC · {len(set(sample_uuids))} samples",
        )

    # -- spectra ------------------------------------------------------------

    @staticmethod
    def _show_single(
        plot: 'MSPlotWidget',
        spectrum: Optional['SpectrumArray'],
        label: str,
    ):
        plot.setSpectrumArray(
            normalize_spectrum(spectrum) if spectrum is not None
            else _empty_spectrum()
        )
        _set_label(plot, label if spectrum is not None else f"{label} (none)")
        plot.update_bpc_label(_bpc_label(spectrum))  # raw base peak intensity
        plot.pi.scaleViewboxToSpectrumArray()

    def _frame_ms(
        self,
        spectra: list[Optional['SpectrumArray']],
        prev_x: tuple[float, float],
    ):
        """
        Set the shared (linked) MS x-axis after drawing: on the first draw
        after a reset, frame it so every shown spectrum fits; afterwards,
        restore `prev_x` - the range from before drawing (the plots' own
        scaling resets it on every draw).
        """
        if self._ms_x_fitted:
            self.ms1_plot.getViewBox().setXRange(*prev_x, padding=0)
            return

        mz_max = max(
            (float(np.nanmax(s['mz'])) for s in spectra
             if s is not None and s.size),
            default=None,
        )
        if mz_max is not None:
            self.ms1_plot.getViewBox().setXRange(0, mz_max * 1.10, padding=0)
            self._ms_x_fitted = True

    @property
    def _ms_x_range(self) -> tuple[float, float]:
        return tuple(self.ms1_plot.getViewBox().viewRange()[0])

    # -- chromatograms ------------------------------------------------------

    def _bpc(
        self,
        ctx: 'AlignmentContext',
        sample_uuid: 'SampleUUID',
    ) -> Optional[np.ndarray]:
        if sample_uuid not in self._bpc_cache:
            sample = ctx.data_source.get_sample(sample_uuid)
            if sample is None or sample.injection is None:
                return None
            self._bpc_cache[sample_uuid] = (
                sample.injection.scan_array_ms1.get_bpc()
            )
        return self._bpc_cache[sample_uuid]

    def _draw_chroms(
        self,
        ctx: Optional['AlignmentContext'],
        sample_uuids: list['SampleUUID'],
        ensembles: list['Ensemble'],
        label: str,
    ):
        """
        Overlay the samples' BPCs (opacity max(1/n, BPC_MIN_OPACITY)) and
        the ensembles' base XICs (full opacity).
        """
        pi = self.chrom_plot.pi
        pi.clear_plots()
        self.chrom_plot.clearPeaks()

        bpcs = []
        if ctx is not None:
            for sample_uuid in dict.fromkeys(sample_uuids):  # dedupe, keep order
                bpc = self._bpc(ctx, sample_uuid)
                if bpc is not None:
                    bpcs.append(bpc)

        if bpcs:
            opacity = max(1.0 / len(bpcs), BPC_MIN_OPACITY)
            pi.addChroms(
                bpcs,
                color=pg.mkColor(*BPC_GREY, int(round(255 * opacity))),
                replace=True,
            )

        xics = []
        for ensemble in ensembles:
            xic = ensemble.get_base_chromatogram(ms_level=1)
            self.chrom_plot.addPeak(xic, uuid=ensemble.uuid, color=XIC_COLOR)
            xics.append(xic)

        self._chroms = bpcs + xics
        self._focus_rts = [float(e.peak_rt) for e in ensembles]
        self.chrom_plot.update_label(label)
        self._fit_chrom_view()

    def _fit_chrom_view(self):
        """
        Linked: x follows the feature map.
        Unlinked: frame the XIC'd peaks (+/- CHROM_RT_PAD).
        y is fitted to the visible data only on the first draw after a reset.
        """
        if not self._chroms:
            return
        vb = self.chrom_plot.getViewBox()
        if not self._rt_linked and self._focus_rts:
            vb.setXRange(
                min(self._focus_rts) - CHROM_RT_PAD,
                max(self._focus_rts) + CHROM_RT_PAD,
                padding=0,
            )
        if not self._chrom_y_fitted:
            self._fit_chrom_y()
            self._chrom_y_fitted = True

    def _fit_chrom_y(self):
        x0, x1 = self.chrom_plot.getViewBox().viewRange()[0]
        y_max = 0.0
        for chrom in self._chroms:
            rt = np.asarray(chrom['rt']).ravel()
            intsy = np.asarray(chrom['intsy']).ravel()
            in_view = (rt >= x0) & (rt <= x1)
            if in_view.any():
                y_max = max(y_max, float(intsy[in_view].max()))
        if y_max > 0:
            self.chrom_plot.getViewBox().setYRange(0, y_max * 1.05, padding=0)
