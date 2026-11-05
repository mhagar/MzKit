"""
Parameter sheets for the search tools' config sections ([spectral_search],
[glacier]): one control per field of the section's NamedTuple. Like the
find-mfs sheet, every edit is written straight to the config on disk, which
the searches re-read each time they run.
"""
from typing import NamedTuple, Optional

from PyQt5 import QtCore, QtWidgets

from core.utils.config import load_config, save_config


class Field(NamedTuple):
    name: str                      # the NamedTuple field / config key
    label: str
    tip: str = ""
    lo: float = 0.0
    hi: float = 1e6
    decimals: int = 3              # floats only
    suffix: str = ""
    choices: tuple = ()            # strings: a combo of these


class ParamSheet(QtWidgets.QDialog):
    """A non-modal sheet editing `fields` of config `section`; `defaults` gives each field's type and default."""

    def __init__(self, title: str, section: str, defaults: NamedTuple, fields: list[Field], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.section, self.defaults = section, defaults
        self.controls: dict[str, QtWidgets.QWidget] = {}
        self._loading = True   # building the controls fires their change signals
        form = QtWidgets.QFormLayout(self)
        for f in fields:
            control = self._control(f, getattr(defaults, f.name))
            control.setToolTip(f.tip)
            form.addRow(f.label, control)
            self.controls[f.name] = control
        note = QtWidgets.QLabel("Saved as you edit; used by the next search.")
        note.setEnabled(False)
        form.addRow(note)
        self._load()

    def _control(self, f: Field, default) -> QtWidgets.QWidget:
        if isinstance(default, str):
            combo = QtWidgets.QComboBox()
            combo.addItems(f.choices or (default,))
            combo.currentTextChanged.connect(lambda v, n=f.name: self._save(n, v))
            return combo
        if isinstance(default, int):
            spin = QtWidgets.QSpinBox()
            spin.setRange(int(f.lo), int(f.hi))
        else:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(f.decimals)
            spin.setRange(f.lo, f.hi)
            spin.setSingleStep(10.0 ** (1 - f.decimals))   # e.g. 4 decimals: steps of 0.001
        spin.setSuffix(f.suffix)
        spin.valueChanged.connect(lambda v, n=f.name: self._save(n, v))
        return spin

    def _load(self):
        """The controls from the config (another sheet, or a hand edit, may have changed it)."""
        config = load_config()
        self._loading = True
        for name, control in self.controls.items():
            default = getattr(self.defaults, name)
            value = config.get(self.section, name, fallback=str(default))
            if isinstance(control, QtWidgets.QComboBox):
                if control.findText(value) < 0:
                    control.addItem(value)
                control.setCurrentText(value)
            else:
                control.setValue(type(default)(float(value)) if isinstance(default, int) else float(value))
        self._loading = False

    def _save(self, name: str, value):
        if self._loading:
            return
        config = load_config()
        if not config.has_section(self.section):
            config.add_section(self.section)
        config.set(self.section, name, str(value))
        save_config(config, [self.section])

    def showEvent(self, event):
        self._load()
        super().showEvent(event)

    @property
    def values(self) -> dict:
        """What's on screen, by field."""
        return {
            name: c.currentText() if isinstance(c, QtWidgets.QComboBox) else c.value()
            for name, c in self.controls.items()
        }


def spectral_sheet(parent=None) -> ParamSheet:
    from core.cli.search_spectra import SECTION, SpectralSearchParams
    return ParamSheet("Spectral search parameters", SECTION, SpectralSearchParams(), [
        Field('precursor_tolerance', "Precursor tolerance",
              "Identity: library precursors within this many Da (Da, not ppm: older GNPS entries only give 2 decimals)",
              0.0001, 5.0, 4, " Da"),
        Field('mz_tolerance', "Peak m/z tolerance", "Matching peaks (cosine, modified cosine)", 0.0001, 1.0, 4, " Da"),
        Field('min_matched_peaks', "Min. matched peaks", "Identity hits need at least this many matched peaks", 1, 100),
        Field('max_hits', "Max. hits", "Identity: keep the best this many", 1, 100000),
        Field('top_k', "Analogue hits", "Analogue (embedding) search: this many nearest library spectra", 1, 100000),
    ], parent)


def glacier_sheet(parent=None) -> ParamSheet:
    from core.cli.rank_candidates import SECTION, RankParams
    return ParamSheet("Structure search (GLACIER) parameters", SECTION, RankParams(), [
        Field('collision_energy', "Collision energy", "Predicted at this collision energy", 0.0, 300.0, 1, " eV"),
        Field('instrument', "Instrument", "", choices=('Orbitrap', 'QTOF', 'IT-FT', 'Unknown')),
        Field('mz_tolerance', "Peak m/z tolerance", "Matching experimental to predicted peaks", 0.0001, 1.0, 4, " Da"),
        Field('top_n_peaks', "Explained: top N peaks",
              "Explained k/N counts the N most intense experimental fragments (precursor region excluded)", 1, 200),
        Field('min_exp_rel', "Min. experimental peak", "...of at least this fraction of the base peak", 0.0, 1.0, 3),
        Field('min_pred_rel', "Min. predicted peak",
              "...explained by a predicted peak of at least this fraction of its base peak", 0.0, 1.0, 3),
        Field('mass_ppm', "Mass query tolerance", "Neutral mass queries: +/- this many ppm", 0.1, 1000.0, 1, " ppm"),
    ], parent)
