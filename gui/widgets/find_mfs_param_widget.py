"""
Reusable find-mfs parameter sheet.

Owns the entire find-mfs parameter panel (charge/counts/halogen cap/RDBE/octet +
Mass-Error, Isotope, Chemical-Prior and MS2/MistNet scoring groups) and is the
single source of truth for the ``[findmfs]`` config section. Its state is a
``core.formula.FindMfsParams``.

Embedded (via Qt Designer widget promotion) into both:
  - FormulaFinderDialog  (the "Parameters" tab)
  - EnsembleExtractionDialog  (the "Auto find-mfs" tab)

Every edit is written straight into the shared config (so e.g. the Ensemble
Viewer's auto button, which reads the config, always uses what's on screen) and
saved to disk shortly after. Constructed param-less by pyuic5; the parent
injects a ConfigParser via ``set_config()`` after ``setupUi``.
"""
from typing import Optional, TYPE_CHECKING

from PyQt5 import QtCore, QtWidgets

from gui.resources.FindMfsParamSheet import Ui_Form
from core.formula.params import FindMfsParams
from core.utils.config import save_config, load_default_config

if TYPE_CHECKING:
    from configparser import ConfigParser


# Coalesce bursts of edits (typing, spinning) into one disk write.
_SAVE_DELAY_MS = 500

_INVALID_STYLE = "QLineEdit { border: 1px solid #d9534f; }"


class FindMfsParamWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # Injected later via set_config(); may stay None in config-less contexts.
        self.config: Optional["ConfigParser"] = None

        # Guards against programmatic updates (set_params) echoing back as edits.
        self._loading = False

        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(_SAVE_DELAY_MS)
        self._save_timer.timeout.connect(self._save_to_disk)

        self.lineHalogenCap.setPlaceholderText("e.g. Cl4Br4")
        self._base_tooltips = {
            line: line.toolTip()
            for line in (self.lineMaxCounts, self.lineMinCounts, self.lineHalogenCap)
        }
        self._populate_instrument_combo()
        self._connect_edit_signals()
        self.btnConfigBox.clicked.connect(self.on_config_btn_pressed)

    # -- config wiring ----------------------------------------------------

    def set_config(self, config: Optional["ConfigParser"]) -> None:
        """Inject the config and populate the controls from it."""
        self.config = config
        self.set_params(FindMfsParams.from_config(config))

    def showEvent(self, event) -> None:
        # Several sheets can share one config (e.g. the formula finder and the
        # extraction dialog); pick up edits made in another one since.
        if self.config is not None and not self._save_timer.isActive():
            self.set_params(FindMfsParams.from_config(self.config))
        super().showEvent(event)

    def _populate_instrument_combo(self) -> None:
        """
        Fill the instrument combo with MistNet's canonical types (label -> value);
        unknown names fall back to 'unknown' inside find-mfs anyway.
        """
        if self.comboInstrument.count() > 0:
            return
        for label, value in [
            ("Unknown", "unknown"),
            ("Q-ToF", "qtof"),
            ("Orbitrap", "orbitrap"),
            ("Ion Trap", "iontrap"),
            ("FT-ICR", "fticr"),
        ]:
            self.comboInstrument.addItem(label, value)

    def _connect_edit_signals(self) -> None:
        for spin in (
            self.spinCharge, self.spinRDBEMin, self.spinRDBEMax,
            self.spinMassErrorPpm, self.spinMassErrorDa, self.spinMassErrorWeight,
            self.spinIsotopeWeight, self.spinIsotopeErrorPpm,
            self.spinIsotopeMatchTolDa, self.spinIsotopeMinRelIntsy,
            self.spinChemPriorWeight, self.spinChemPriorStrength,
            self.spinChemPriorSoftness, self.doubleSpinMs2Weight, self.spinTopN,
        ):
            spin.valueChanged.connect(self._on_edited)
        for line in (self.lineMaxCounts, self.lineMinCounts, self.lineHalogenCap):
            line.textChanged.connect(self._on_edited)
        for check in (self.checkOctet, self.checkAutodetectHalogens):
            check.toggled.connect(self._on_edited)
        self.comboInstrument.currentIndexChanged.connect(self._on_edited)

    def on_config_btn_pressed(
        self,
        button: QtWidgets.QAbstractButton,
    ) -> None:
        standard_button = self.btnConfigBox.standardButton(button)
        match standard_button:
            case QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults:
                # The shipped template, ignoring user overrides. Goes through
                # the normal edit path, so it is persisted too.
                self.set_params(
                    FindMfsParams.from_config(load_default_config()),
                    persist=True,
                )
            case QtWidgets.QDialogButtonBox.StandardButton.Save:
                # Edits already persist on their own; just don't wait for it.
                self._save_to_disk()

    # -- persistence ------------------------------------------------------

    def _on_edited(self, *_) -> None:
        self._update_validation()
        if self._loading or self.config is None:
            return
        # In-memory config is updated immediately; only the disk write waits.
        self.get_params().to_config(self.config)
        self._save_timer.start()

    def _save_to_disk(self) -> None:
        self._save_timer.stop()
        if self.config is not None:
            save_config(self.config)

    # -- params -----------------------------------------------------------

    def get_params(self) -> FindMfsParams:
        return FindMfsParams(
            charge=self.spinCharge.value(),
            max_counts=self.lineMaxCounts.text().strip(),
            min_counts=self.lineMinCounts.text().strip(),
            detect_halogens=self.checkAutodetectHalogens.isChecked(),
            halogen_cap=self.lineHalogenCap.text().strip(),
            min_rdbe=self.spinRDBEMin.value(),
            max_rdbe=self.spinRDBEMax.value(),
            check_octet=self.checkOctet.isChecked(),
            error_ppm=self.spinMassErrorPpm.value(),
            error_da=self.spinMassErrorDa.value(),
            mass_weight=self.spinMassErrorWeight.value(),
            iso_weight=self.spinIsotopeWeight.value(),
            iso_ppm=self.spinIsotopeErrorPpm.value(),
            iso_mz_match_da=self.spinIsotopeMatchTolDa.value(),
            iso_min_rel=self.spinIsotopeMinRelIntsy.value(),
            chem_weight=self.spinChemPriorWeight.value(),
            chem_strength=self.spinChemPriorStrength.value(),
            chem_softness=self.spinChemPriorSoftness.value(),
            ms2_weight=self.doubleSpinMs2Weight.value(),
            instrument=self.comboInstrument.currentData() or "unknown",
            top_n=int(self.spinTopN.value()),
        )

    def set_params(self, p: FindMfsParams, persist: bool = False) -> None:
        """Populate the controls. `persist` also writes them to the config."""
        self._loading = True
        try:
            self.spinCharge.setValue(p.charge)
            self.lineMaxCounts.setText(p.max_counts)
            self.lineMinCounts.setText(p.min_counts)
            self.checkAutodetectHalogens.setChecked(p.detect_halogens)
            self.lineHalogenCap.setText(p.halogen_cap)
            self.lineHalogenCap.setEnabled(p.detect_halogens)
            self.spinRDBEMin.setValue(p.min_rdbe)
            self.spinRDBEMax.setValue(p.max_rdbe)
            self.checkOctet.setChecked(p.check_octet)
            self.spinMassErrorPpm.setValue(p.error_ppm)
            self.spinMassErrorDa.setValue(p.error_da)
            self.spinMassErrorWeight.setValue(p.mass_weight)
            self.spinIsotopeWeight.setValue(p.iso_weight)
            self.spinIsotopeErrorPpm.setValue(p.iso_ppm)
            self.spinIsotopeMatchTolDa.setValue(p.iso_mz_match_da)
            self.spinIsotopeMinRelIntsy.setValue(p.iso_min_rel)
            self.spinChemPriorWeight.setValue(p.chem_weight)
            self.spinChemPriorStrength.setValue(p.chem_strength)
            self.spinChemPriorSoftness.setValue(p.chem_softness)
            self.doubleSpinMs2Weight.setValue(p.ms2_weight)
            idx = self.comboInstrument.findData(p.instrument)
            self.comboInstrument.setCurrentIndex(idx if idx >= 0 else 0)
            self.spinTopN.setValue(p.top_n)
        finally:
            self._loading = False
        if persist:
            self._on_edited()
        self._update_validation()

    # -- validation -------------------------------------------------------

    def validation_error(self) -> Optional[str]:
        """Why find-mfs would reject the current constraints, or None if OK."""
        try:
            self.get_params().validate()
        except ValueError as exc:
            return str(exc)
        return None

    def _update_validation(self) -> None:
        """Outline the constraint fields red (with the reason) while invalid."""
        error = self.validation_error()
        for line, base_tip in self._base_tooltips.items():
            line.setStyleSheet(_INVALID_STYLE if error else "")
            line.setToolTip(error or base_tip)
