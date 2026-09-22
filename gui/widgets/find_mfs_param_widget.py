"""
Reusable find-mfs parameter sheet.

Owns the entire find-mfs parameter panel (element/charge/counts/RDBE/octet +
Mass-Error, Isotope, Chemical-Prior and MS2/MistNet scoring groups) plus its own
Save/Reset/RestoreDefaults button box, and is the single source of truth for the
``[findmfs]`` config section.

Embedded (via Qt Designer widget promotion) into both:
  - FormulaFinderDialog  (the "Parameters" tab)
  - EnsembleExtractionDialog  (the "Auto find-mfs" tab)

so the same controls, defaults and persistence back every place find-mfs is
configured. Constructed param-less by pyuic5; the parent injects a ConfigParser
via ``set_config()`` after ``setupUi``.
"""
from typing import Optional, TYPE_CHECKING

from PyQt5 import QtWidgets

from gui.resources.FindMfsParamSheet import Ui_Form
from core.utils.config import save_config, load_default_config

if TYPE_CHECKING:
    from configparser import ConfigParser


_SECTION = "findmfs"


class FindMfsParamWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # Injected later via set_config(); may stay None in config-less contexts.
        self.config: Optional["ConfigParser"] = None

        self._populate_instrument_combo()
        self.btnConfigBox.clicked.connect(self.on_config_btn_pressed)

    # -- config wiring ----------------------------------------------------

    def set_config(self, config: Optional["ConfigParser"]) -> None:
        """Inject the config and populate the controls from it."""
        self.config = config
        self.load_from_config()

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

    def on_config_btn_pressed(
        self,
        button: QtWidgets.QAbstractButton,
    ) -> None:
        standard_button = self.btnConfigBox.standardButton(button)
        match standard_button:
            case QtWidgets.QDialogButtonBox.StandardButton.Save:
                self.save_to_config()
            case QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults:
                # Populate from the shipped template, ignoring user overrides.
                self.load_from_config(load_default_config())
            case QtWidgets.QDialogButtonBox.StandardButton.Reset:
                self.load_from_config()

    def load_from_config(
        self,
        config: Optional["ConfigParser"] = None,
    ) -> None:
        """
        Populate the controls from a ConfigParser. Defaults to the widget's own
        config; pass a different one (e.g. the shipped template) to restore those
        values instead. No-op when no config is available.
        """
        config = config if config is not None else self.config
        if not config:
            return

        g = _SECTION
        self.spinCharge.setValue(config.getint(g, "charge", fallback=0))
        self.spinMassErrorPpm.setValue(config.getfloat(g, "error_ppm", fallback=5.0))
        self.spinMassErrorDa.setValue(config.getfloat(g, "error_da", fallback=0.01))
        self.spinMassErrorWeight.setValue(
            config.getfloat(g, "mass_weight", fallback=1.0)
        )
        self.lineMinCounts.setText(config.get(g, "min_counts", fallback=""))
        self.lineMaxCounts.setText(config.get(g, "max_counts", fallback=""))
        self.spinRDBEMin.setValue(config.getfloat(g, "min_rdbe", fallback=0.0))
        self.spinRDBEMax.setValue(config.getfloat(g, "max_rdbe", fallback=0.0))
        self.checkOctet.setChecked(config.getboolean(g, "check_octet", fallback=True))

        self.spinIsotopeErrorPpm.setValue(config.getfloat(g, "iso_ppm", fallback=5.0))
        self.spinIsotopeMatchTolDa.setValue(
            config.getfloat(g, "iso_mz_match_da", fallback=0.02)
        )
        self.spinIsotopeMinRelIntsy.setValue(
            config.getfloat(g, "iso_min_rel", fallback=0.02)
        )
        self.spinIsotopeWeight.setValue(config.getfloat(g, "iso_weight", fallback=1.0))

        self.spinChemPriorWeight.setValue(
            config.getfloat(g, "chem_weight", fallback=1.0)
        )
        self.spinChemPriorStrength.setValue(
            config.getfloat(g, "chem_strength", fallback=1.0)
        )
        self.spinChemPriorSoftness.setValue(
            config.getfloat(g, "chem_softness", fallback=1.0)
        )

        # === Compound (MS2) / MistNet ===
        self.doubleSpinMs2Weight.setValue(config.getfloat(g, "ms2_weight", fallback=1.0))
        instr = config.get(g, "instrument", fallback="unknown")
        instr_idx = self.comboInstrument.findData(instr)
        self.comboInstrument.setCurrentIndex(instr_idx if instr_idx >= 0 else 0)
        self.checkBoxAcheckAutodetectHalogens.setChecked(
            config.getboolean(g, "autodetect_cl_br", fallback=True)
        )
        self.spinTopN.setValue(config.getint(g, "top_n", fallback=50))

    def save_to_config(self) -> None:
        """Persist the current controls to the ``[findmfs]`` section and disk."""
        if not self.config:
            return
        g = _SECTION

        def setv(key, value):
            self.config.set(section=g, option=key, value=str(value))

        setv("charge", self.spinCharge.value())
        setv("error_ppm", self.spinMassErrorPpm.value())
        setv("error_da", self.spinMassErrorDa.value())
        setv("mass_weight", self.spinMassErrorWeight.value())
        setv("min_counts", self.lineMinCounts.text())
        setv("max_counts", self.lineMaxCounts.text())
        setv("min_rdbe", self.spinRDBEMin.value())
        setv("max_rdbe", self.spinRDBEMax.value())
        setv("check_octet", self.checkOctet.isChecked())

        setv("iso_ppm", self.spinIsotopeErrorPpm.value())
        setv("iso_mz_match_da", self.spinIsotopeMatchTolDa.value())
        setv("iso_min_rel", self.spinIsotopeMinRelIntsy.value())
        setv("iso_weight", self.spinIsotopeWeight.value())

        setv("chem_weight", self.spinChemPriorWeight.value())
        setv("chem_strength", self.spinChemPriorStrength.value())
        setv("chem_softness", self.spinChemPriorSoftness.value())

        setv("ms2_weight", self.doubleSpinMs2Weight.value())
        setv("instrument", self.comboInstrument.currentData() or "unknown")
        setv("autodetect_cl_br", self.checkBoxAcheckAutodetectHalogens.isChecked())
        setv("top_n", int(self.spinTopN.value()))

        save_config(self.config)

    # -- param getters ----------------------------------------------------

    def get_element_set_text(self) -> str:
        return self.comboElementSet.currentText()

    def get_elements_str(self) -> str:
        """Element set as the string find-mfs' compound path expects."""
        return (
            "CHNOPSFClBrI"
            if "halogen" in self.comboElementSet.currentText().lower()
            else "CHNOPS"
        )

    def get_finder_kwargs(self) -> dict:
        return {
            "min_counts": self.lineMinCounts.text(),
            "max_counts": self.lineMaxCounts.text(),
            "filter_rdbe": (self.spinRDBEMin.value(), self.spinRDBEMax.value()),
            "check_octet": self.checkOctet.isChecked(),
        }

    def get_mf_params(self) -> dict:
        """
        find-mfs ion-search kwargs (no ``adduct`` — that lives on the caller,
        e.g. FormulaFinder's own line edit, and is merged in there).
        """
        return {
            "charge": self.spinCharge.value(),
            "error_ppm": self.spinMassErrorPpm.value(),
            "error_da": self.spinMassErrorDa.value(),
            "min_counts": self.lineMinCounts.text(),
            "max_counts": self.lineMaxCounts.text(),
            "filter_rdbe": (self.spinRDBEMin.value(), self.spinRDBEMax.value()),
            "check_octet": self.checkOctet.isChecked(),
        }

    def get_score_params(self) -> dict:
        return {
            "iso_ppm": self.spinIsotopeErrorPpm.value(),
            "iso_mz_match_da": self.spinIsotopeMatchTolDa.value(),
            "iso_min_rel": self.spinIsotopeMinRelIntsy.value(),
            "iso_weight": self.spinIsotopeWeight.value(),
            "mass_weight": self.spinMassErrorWeight.value(),
            "chem_weight": self.spinChemPriorWeight.value(),
            "chem_strength": self.spinChemPriorStrength.value(),
            "chem_softness": self.spinChemPriorSoftness.value(),
        }

    def get_compound_params(self) -> dict:
        """
        Kwargs for FormulaFinder's interactive compound (MS2) search
        (``core.formula.query_from_signals``), minus ``adducts`` which the caller
        merges from its own adduct field. Note this path spells the halogen flag
        ``autodetect_cl_br`` (vs. ``detect_halogens`` for the batch annotator).
        """
        return {
            "elements": self.get_elements_str(),
            "autodetect_cl_br": self.checkBoxAcheckAutodetectHalogens.isChecked(),
            "error_ppm": self.spinMassErrorPpm.value(),
            "instrument": self.comboInstrument.currentData() or "unknown",
            "ms2_weight": self.doubleSpinMs2Weight.value(),
            "top_n": int(self.spinTopN.value()),
            "finder_kwargs": self.get_finder_kwargs(),
        }

    def get_annotation_params(self) -> dict:
        """
        Kwargs for ``core.cli.auto_find_mfs.annotate_ensembles_dia`` (the batch
        MS2 annotator). Same shape as ``annotation_params_from_config``.
        """
        return {
            "elements": self.get_elements_str(),
            "error_ppm": self.spinMassErrorPpm.value(),
            "instrument": self.comboInstrument.currentData() or "unknown",
            "ms2_weight": self.doubleSpinMs2Weight.value(),
            "detect_halogens": self.checkBoxAcheckAutodetectHalogens.isChecked(),
            "top_n": int(self.spinTopN.value()),
            "finder_kwargs": self.get_finder_kwargs(),
        }
