"""
Modal dialog for ensemble extraction.
Unified settings + launch surface for both automated ensemble generation
 and manual single-ensemble extraction (i.e. "Cmpd" button)

Two modes:
  - AUTO:   sample checklist + all extraction params + optional find-mfs; OK runs
            auto-generation across the checked samples
            (and, if enabled, chains find-mfs over the new ensembles)
  - SINGLE: settings surface for the manual Cmpd tool. The disables settings
            pertaining to auto-extraction, and the sample list.
            Group scoring + find-mfs params stay editable.
            OK just stores the params. the seed/window are still picked on the plot.

All initial values come from the config (`[auto_ensemble]`+ `[findmfs]`).

The dict shapes returned by `get_params` / `get_auto_params` match those
 the controller feeds to `EnsembleExtractionParams` / `AutoEnsembleParams` unchanged
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from gui.resources.EnsembleExtractionDialog import Ui_Dialog
from core.utils.config import save_config, load_default_config
from core.cli.generate_ensemble import (
    AutoEnsembleParams,
    auto_params_from_config,
    auto_params_to_config,
)

if TYPE_CHECKING:
    from configparser import ConfigParser
    from core.formula.params import FindMfsParams
    from core.data_structs import SampleUUID


class ExtractionDialogMode(Enum):
    AUTO = auto()
    SINGLE = auto()


class EnsembleExtractionDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(
        self,
        *,
        config: Optional["ConfigParser"],
        mode: ExtractionDialogMode = ExtractionDialogMode.AUTO,
        loaded_samples: Optional[list[tuple["SampleUUID", str]]] = None,
        selected_uuids: Optional[set["SampleUUID"]] = None,
        prefill_rt: Optional[tuple[float, float]] = None,
        prefill_seed_intsy: Optional[float] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self.setModal(True)

        self.config = config
        self.mode = mode

        # find-mfs sheet manages its own [findmfs] persistence.
        self.findMfsParams.set_config(config)

        # Populate every extraction control from [auto_ensemble].
        if config is not None:
            self._apply_auto_params(
                auto_params_from_config(config)
            )

        self._populate_sample_list(
            loaded_samples or [],
            selected_uuids or set(),
        )

        # Dialog-level config buttons persist [auto_ensemble] only; the find-mfs
        # widget carries its own Save/Reset/RestoreDefaults for [findmfs].
        self.btnConfigBox.clicked.connect(self._on_config_btn_pressed)

        if mode is ExtractionDialogMode.SINGLE:
            self._apply_single_mode()

        # Cross-bar tool prefill (RT window + seed intensity), AUTO mode.
        if prefill_rt is not None:
            self.groupRTWindow.setChecked(True)
            self.spinnerRTStart.setValue(prefill_rt[0])
            self.spinnerRTEnd.setValue(prefill_rt[1])
        if prefill_seed_intsy is not None:
            self.spinnerParentThreshold.setValue(prefill_seed_intsy)

        self.setWindowTitle(
            "Auto Ensemble Extraction"
            if mode is ExtractionDialogMode.AUTO
            else "Ensemble Extraction Settings"
        )

    # -- mode / samples ---------------------------------------------------
    def _apply_single_mode(self) -> None:
        """
        Grey out the auto-only surface (sample list + all auto params); keep the
        shared scoring group and the find-mfs tab editable.
        """
        self.groupSamples.setEnabled(False)
        self.groupAuto.setEnabled(False)

    def _populate_sample_list(
        self,
        loaded_samples: list[tuple["SampleUUID", str]],
        selected_uuids: set["SampleUUID"],
    ) -> None:
        self.listSamples.clear()
        for uuid, name in loaded_samples:
            item = QtWidgets.QListWidgetItem(name)
            item.setData(Qt.UserRole, uuid)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if uuid in selected_uuids else Qt.Unchecked
            )
            self.listSamples.addItem(item)

    def get_selected_sample_uuids(self) -> list["SampleUUID"]:
        uuids: list["SampleUUID"] = []
        for row in range(self.listSamples.count()):
            item = self.listSamples.item(row)
            if item.checkState() == Qt.Checked:
                uuids.append(item.data(Qt.UserRole))
        return uuids

    # -- find-mfs ---------------------------------------------------------

    def run_findmfs(self) -> bool:
        return self.groupRunFindMfs.isChecked()

    def get_findmfs_params(self) -> Optional["FindMfsParams"]:
        """
        The find-mfs parameters when find-mfs is enabled, else None.
        """
        if not self.run_findmfs():
            return None
        return self.findMfsParams.get_params()

    # -- param getters (dict shapes must match the extraction call sites) --

    def _method(self) -> str:
        # combo items are labelled "Cosine"/"Pearson"; the engine wants lowercase.
        return self.comboMethod.currentText().strip().lower()

    def _rt_range(self) -> Optional[tuple[float, float]]:
        """
        (start, end) RT window in SECONDS, or None when the restriction is off or
        the bounds are degenerate.
        Spinners are in minutes; the engine's rt_arr is in seconds, so convert here.
        """
        if not self.groupRTWindow.isChecked():
            return None
        start = float(self.spinnerRTStart.value())
        end = float(self.spinnerRTEnd.value())
        if end <= start:
            return None
        return (start * 60.0, end * 60.0)

    def get_params(self) -> dict:
        """Params for *manual* single-ensemble extraction."""
        return {
            "ms1_corr_threshold": float(self.spinnerMS1Threshold.value()),
            "ms2_corr_threshold": float(self.spinnerMS2Threshold.value()),
            "min_intsy": float(self.spinnerMinIntsy.value()),
            "use_rel_intsy": self.checkRelativeIntensity.isChecked(),
            "method": self._method(),
        }

    def get_auto_params(self) -> dict:
        """
        Params for *automated* generation, shaped to construct an
        AutoEnsembleParams (omitted fields fall back to its defaults, notably
        min_window_halfwidth, min_turn, adduct_ppm_tol, polarity).
        """
        return {
            # shared with manual extraction
            "ms1_corr_threshold": float(self.spinnerMS1Threshold.value()),
            "ms2_corr_threshold": float(self.spinnerMS2Threshold.value()),
            "cofeature_threshold": float(self.spinnerMinIntsy.value()),
            "use_rel_intsy": self.checkRelativeIntensity.isChecked(),
            "method": self._method(),
            # seed selection
            "parent_threshold": float(self.spinnerParentThreshold.value()),
            "window_strategy": self.comboWindow.currentText(),
            "extraction_half_width": int(self.spinnerHalfWidth.value()),
            "adduct_aware": self.groupAdductAware.isChecked(),
            "loose_corr_threshold": float(self.spinnerLooseThreshold.value()),
            # DDA only: gate whether unfragmented coeluting MS1 features are grouped
            "dda_require_ms2": self.checkRequireMs2.isChecked(),
            # peak validation
            "peak_method": self.comboPeakMethod.currentText(),
            "min_peak_width": int(self.spinMinPeakWidth.value()),
            "min_prominence": float(self.spinnerMinProminence.value()),
            "baseline_pct": float(self.spinBaselinePct.value()),
            "edge_fraction": float(self.spinnerEdgeFraction.value()),
            # background suppression
            "require_smoothing_survival": self.groupSmoothing.isChecked(),
            "smoothing_sigma": float(self.spinnerSmoothingSigma.value()),
            "min_smoothing_survival": float(
                self.spinnerMinSmoothingSurvival.value()
            ),
            "max_lane_persistence": (
                float(self.spinnerMaxLanePersistence.value())
                if self.groupPersistence.isChecked()
                else None
            ),
            # RT window (None = whole run)
            "rt_range": self._rt_range(),
        }

    # -- config persistence (auto params) ---------------------------------

    def _on_config_btn_pressed(
        self,
        button: QtWidgets.QAbstractButton,
    ) -> None:
        standard_button = self.btnConfigBox.standardButton(button)
        match standard_button:
            case QtWidgets.QDialogButtonBox.StandardButton.Save:
                self._save_auto_params_to_config()
            case QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults:
                self._apply_auto_params(auto_params_from_config(load_default_config()))
            case QtWidgets.QDialogButtonBox.StandardButton.Reset:
                if self.config is not None:
                    self._apply_auto_params(auto_params_from_config(self.config))

    def _save_auto_params_to_config(self) -> None:
        if self.config is None:
            return
        # Start from the stored params so GUI-hidden fields (min_turn,
        # min_window_halfwidth, adduct_ppm_tol, polarity) are preserved.
        merged = auto_params_from_config(self.config)._replace(
            **self.get_auto_params()
        )
        auto_params_to_config(self.config, merged)
        save_config(self.config)

    def _select_combo(
        self,
        combo: QtWidgets.QComboBox,
        value: str,
        *,
        ci: bool = False,
    ) -> None:
        target = value.lower() if ci else value
        for i in range(combo.count()):
            text = combo.itemText(i)
            if (text.lower() if ci else text) == target:
                combo.setCurrentIndex(i)
                return

    def _apply_auto_params(self, p: AutoEnsembleParams) -> None:
        # shared scoring
        self.spinnerMS1Threshold.setValue(p.ms1_corr_threshold)
        self.spinnerMS2Threshold.setValue(p.ms2_corr_threshold)
        self.spinnerMinIntsy.setValue(p.cofeature_threshold)
        self.checkRelativeIntensity.setChecked(p.use_rel_intsy)
        self._select_combo(self.comboMethod, p.method, ci=True)
        # seed selection
        self.spinnerParentThreshold.setValue(p.parent_threshold)
        self._select_combo(self.comboWindow, p.window_strategy)
        self.spinnerHalfWidth.setValue(p.extraction_half_width)
        self.groupAdductAware.setChecked(p.adduct_aware)
        self.spinnerLooseThreshold.setValue(p.loose_corr_threshold)
        self.checkRequireMs2.setChecked(p.dda_require_ms2)
        # peak validation
        self._select_combo(self.comboPeakMethod, p.peak_method)
        self.spinMinPeakWidth.setValue(p.min_peak_width)
        self.spinnerMinProminence.setValue(p.min_prominence)
        self.spinBaselinePct.setValue(int(p.baseline_pct))
        self.spinnerEdgeFraction.setValue(p.edge_fraction)
        # background suppression
        self.groupSmoothing.setChecked(p.require_smoothing_survival)
        self.spinnerSmoothingSigma.setValue(p.smoothing_sigma)
        self.spinnerMinSmoothingSurvival.setValue(p.min_smoothing_survival)
        self.groupPersistence.setChecked(p.max_lane_persistence is not None)
        if p.max_lane_persistence is not None:
            self.spinnerMaxLanePersistence.setValue(p.max_lane_persistence)
        # RT window (params in seconds; controls in minutes)
        self.groupRTWindow.setChecked(p.rt_range is not None)
        if p.rt_range is not None:
            self.spinnerRTStart.setValue(p.rt_range[0] / 60.0)
            self.spinnerRTEnd.setValue(p.rt_range[1] / 60.0)

    # -- accept validation ------------------------------------------------

    def accept(self) -> None:  # type: ignore[override]
        if (
            self.mode is ExtractionDialogMode.AUTO
            and not self.get_selected_sample_uuids()
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "No samples selected",
                "Select at least one sample to extract from.",
            )
            return
        error = self.findMfsParams.validation_error() if self.run_findmfs() else None
        if error:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid find-mfs parameters",
                f"{error}\n\nFix them on the 'Auto find-mfs' tab, or untick "
                f"'Run find-mfs on new ensembles'.",
            )
            return
        super().accept()
