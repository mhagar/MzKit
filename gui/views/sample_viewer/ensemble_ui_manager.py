from PyQt5 import QtCore, QtGui
import pyqtgraph as pg

from core.utils.formula_formatting import format_assignment_label_html

from typing import Optional, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from gui.views.sample_viewer.sample_widget_manager import SampleWidgetManager
    from gui.widgets.SampleWidget import SampleWidget
    from gui.views.sample_viewer.model import SampleViewerItemModel
    from core.data_structs import (
        Ensemble,
        EnsembleUUID,
        SampleUUID,
        FeaturePointer,
        ScanArray,
    )
    from core.data_structs.formula_assignment import FormulaAssignment

class EnsembleUIManager(QtCore.QObject):
    """
    Manages UI elements pertaining to Ensembles
    """
    def __init__(
        self,
        model: Optional['SampleViewerItemModel'],
        widget_manager: 'SampleWidgetManager',
    ):
        super().__init__()
        self._model = model
        self._widget_mgr = widget_manager

    def set_model(
        self,
        model: 'SampleViewerItemModel',
    ):
        self._model = model

    def display_ensembles_for_all_samples(
        self,
        ms_level: Literal[1, 2] = 1,
    ):
        """
        Adds ensemble overlays to all visible widgets
        """
        for uuid, widget in self._widget_mgr.get_all_widgets().items():
            self.clear_ensembles_for_sample(
                uuid=uuid,
            )

            self.display_ensembles_for_sample(
                uuid=uuid,
                ms_level=ms_level,
            )

    def display_ensembles_for_sample(
        self,
        uuid: 'SampleUUID',
        ms_level: Literal[1, 2] = 1,
    ):
        """
        Renders all ensembles for a given sample as colored peak overlays

        :param uuid: Sample UUID
        :param ms_level: MS level to display (1 or 2, default 1)
        """
        injection = self._model.getInjection(uuid)
        if not injection:
            return

        widget = self._widget_mgr.get_widget(uuid)
        if not widget:
            return

        # Clear existing peaks first to prevent stale state
        widget.clearPeaks()

        # Iterate through injection's ensembles and add peak overlays
        for idx, (ensemble_uuid, ensemble) in enumerate(
            injection.ensembles.items()
        ):
            # Get base cofeature chrom.
            chrom_arr = ensemble.get_base_chromatogram(ms_level)

            assignment = self._model.getAssignment(ensemble_uuid)

            widget.addPeak(
                chrom=chrom_arr,
                uuid=ensemble_uuid,
                color=_generate_ensemble_color(ensemble_uuid),
                html_label=_ensemble_label_html(ensemble, assignment),
            )

    def refresh_ensemble_label(
        self,
        ensemble_uuid: 'EnsembleUUID',
    ) -> None:
        """
        Rebuild the on-plot label for a single ensemble overlay — e.g. after a
        FormulaAssignment is added or removed. Resolves which loaded sample
        owns the ensemble, then updates just that peak's label (no full
        redraw). No-op if the ensemble isn't currently drawn.
        """
        for sample_uuid, widget in self._widget_mgr.get_all_widgets().items():
            injection = self._model.getInjection(sample_uuid)
            if not injection or ensemble_uuid not in injection.ensembles:
                continue

            ensemble = injection.ensembles[ensemble_uuid]
            assignment = self._model.getAssignment(ensemble_uuid)
            widget.setPeakLabel(
                uuid=ensemble_uuid,
                html_label=_ensemble_label_html(ensemble, assignment),
            )
            return

    def clear_ensembles_for_sample(
        self,
        uuid: 'SampleUUID',
    ):
        """
        Remove all ensemble overlays for given sample
        """
        widget = self._widget_mgr.get_widget(uuid)
        if widget:
            widget.clearPeaks()

    def show_scan_window_selector(
        self,
        uuid: 'SampleUUID',
        ms_level: Literal[1, 2],
        mass_lane_idx: int,
        initial_scan_window: int,
    ):
        """
        Shows a 'scan window selector' that can be
        used to define the duration with which a user
        would like to extract an Ensemble

        :param uuid: Sample UUID
        :param ms_level:
        :param mass_lane_idx:
        :param initial_scan_window:
        """
        # Clear previous scan window selector
        self.clear_scan_window_selector()

        # Retrieve a feature pointer corresponding to an XIC of entire lane
        scan_array: 'ScanArray' = self._model.getInjection(
            uuid
        ).get_scan_array(
            ms_level=ms_level
        )

        ftr_ptr: 'FeaturePointer'= scan_array. make_feature_pointer(
            mass_lane_idx=mass_lane_idx,
            scan_idxs=None, # returns whole lane
        )

        # Get default rt edges to palce the window initially
        apex_idx: int = ftr_ptr.get_max_intsy_scan_num(scan_array)
        rt_start = scan_array.rt_arr[apex_idx - initial_scan_window]
        rt_end = scan_array.rt_arr[apex_idx + initial_scan_window]

        # Finally, add the graphic to the appropriate sample_widget
        sample_widget: 'SampleWidget' = self._widget_mgr.get_widget(uuid)
        sample_widget.addWindowSelector(
            bounds=(rt_start, rt_end),
            display_arr=ftr_ptr.get_chrom_array(scan_array),
        )

    def clear_scan_window_selector(self):
        """
        Removes all the graphics generated by .show_scan_window_selector()
        """
        for uuid, sample_widget in self._widget_mgr.get_all_widgets().items():
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
        sample_widget = self._widget_mgr.get_widget(uuid)
        return sample_widget.getWindowSelectorBounds()


def _ensemble_label_html(
    ensemble: 'Ensemble',
    assignment: Optional['FormulaAssignment'],
) -> str:
    """
    Overlay label for an ensemble: its identity (compound name) on the first
    line, and its accepted formula (from its FormulaAssignment) on the second.
    Either line may be absent; returns '' if the ensemble has neither.

    The formula line is built by the shared `format_assignment_label_html`,
    so this overlay and the EnsembleViewer's title strip stay in agreement.
    """
    parts: list[str] = []

    if ensemble.identity:
        parts.append(str(ensemble.identity))

    formula_html = format_assignment_label_html(assignment)
    if formula_html:
        parts.append(formula_html)

    return "<br>".join(parts)


def _generate_ensemble_color(
    idx: int,
) -> QtGui.QColor:
    """
    Avoids generating grey, which is used for raw data
    """
    return pg.intColor(
        idx,
        hues=12,
        minValue=150,
        maxValue=255,
        sat=128,
    )

