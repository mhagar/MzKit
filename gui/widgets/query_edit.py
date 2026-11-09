"""
The search tabs' query box: a mode combo (Formula, a mass mode, then any
extra modes) and its line edit. Choosing Formula or the mass mode refills the
box from the target's values (`set_values`); extra modes keep what's typed.
"""
from typing import Optional

from PyQt5 import QtCore, QtWidgets

FORMULA, MASS = 0, 1


class QueryEdit(QtWidgets.QWidget):
    def __init__(self, mass_label: str, placeholders: dict[int, str], extra_modes=(),
                 other_placeholder: str = "", parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.combo, self.edit = QtWidgets.QComboBox(), QtWidgets.QLineEdit()
        self.combo.addItems(["Formula", mass_label, *extra_modes])
        layout.addWidget(self.combo)
        layout.addWidget(self.edit, stretch=1)
        self._placeholders, self._other_placeholder = placeholders, other_placeholder
        self._formula: Optional[str] = None
        self._mass: Optional[float] = None
        self.combo.currentIndexChanged.connect(self._on_mode_changed)
        self._on_mode_changed()

    def set_values(self, formula: Optional[str], mass: Optional[float]):
        """The target's formula and mass: shows the formula if there is one (and Formula is enabled), else the mass, overwriting what's typed."""
        self._formula, self._mass = formula, mass
        formula_ok = self.combo.model().item(FORMULA).isEnabled()
        with QtCore.QSignalBlocker(self.combo):
            self.combo.setCurrentIndex(FORMULA if formula_ok and (formula or mass is None) else MASS)
        self._on_mode_changed()

    def set_formula_enabled(self, enabled: bool):
        self.combo.model().item(FORMULA).setEnabled(enabled)
        if not enabled and self.mode() == FORMULA:
            self.combo.setCurrentIndex(MASS)

    def mode(self) -> int:
        return self.combo.currentIndex()

    def text(self) -> str:
        return self.edit.text().strip()

    def _on_mode_changed(self):
        mode = self.mode()
        self.edit.setPlaceholderText(self._placeholders.get(mode, self._other_placeholder))
        if mode == FORMULA:
            self.edit.setText(self._formula or "")
        elif mode == MASS:
            self.edit.setText("" if self._mass is None else f"{self._mass:.4f}")
