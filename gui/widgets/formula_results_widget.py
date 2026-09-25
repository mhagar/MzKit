"""
Reusable view of one FormulaAssignment: provenance line, sortable ranked
candidate table, and Assign / Clear buttons.

Embedded (via Qt Designer widget promotion) into FormulaFinderDialog for fresh
searches, and wrapped by FormulaAssignmentDialog for reviewing or changing an
assignment that's already registered. The widget only displays and reports
the user's pick; what "assign" means (register an assignment, annotate a
signal, change a registered choice) is up to its host.
"""
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from PyQt5 import QtCore, QtGui, QtWidgets
from molmass import Formula

from gui.resources.FormulaResultsWidget import Ui_Form
from gui.widgets.html_delegate import HTMLDelegate
from core.utils.formula_formatting import format_formula_obj_to_html

if TYPE_CHECKING:
    from core.data_structs.formula_assignment import (
        AssignedCandidate,
        FormulaAssignment,
    )

# Table columns (10):
#   0 Formula  1 Adduct  2 Error(ppm)  3 Error(Da)  4 RDBE
#   5 Mass LL  6 Iso LL  7 Chem Prior  8 MS2 LL     9 Posterior

_LOW = float('-inf')


def _desc(value: Optional[float]) -> float:
    """Sort key for 'higher is better' scores; missing terms sort last."""
    return -value if value is not None else -_LOW


# comboSortBy text (lowercased) -> sort key over AssignedCandidate
_SORT_KEYS = {
    'posterior': lambda c: _desc(c.log_posterior),
    'mass error': lambda c: abs(c.error_ppm) if c.error_ppm is not None else -_LOW,
    'isotope envelope': lambda c: _desc(c.iso_loglik),
    'chemical prior': lambda c: _desc(c.chem_logprior),
}


class FormulaResultsWidget(QtWidgets.QWidget, Ui_Form):
    # Index into assignment.candidates (not the table row: the table is sorted)
    sigAssignRequested = QtCore.pyqtSignal(int)
    sigClearRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self._assignment: Optional['FormulaAssignment'] = None
        self._order: list[int] = []     # table row -> candidate index
        self._allow_clear = False

        # Subscript-styled formulae
        self.tableResults.setItemDelegateForColumn(0, HTMLDelegate(self.tableResults))

        self.comboSortBy.currentIndexChanged.connect(self._populate_table)
        self.tableResults.itemSelectionChanged.connect(self._update_buttons)
        self.tableResults.doubleClicked.connect(self._on_assign)
        self.btnAssignSelected.clicked.connect(self._on_assign)
        self.btnClearAssignment.clicked.connect(self.sigClearRequested)

        self.clear()

    # -- public -----------------------------------------------------------

    @property
    def assignment(self) -> Optional['FormulaAssignment']:
        return self._assignment

    def set_assignment(
        self,
        assignment: Optional['FormulaAssignment'],
        *,
        allow_clear: bool = False,
        note: str = "",
    ) -> None:
        """
        Show `assignment`. `allow_clear` offers 'Clear Assignment' while a
        candidate is chosen (only meaningful for a registered assignment);
        `note` is appended to the provenance line.
        """
        self._assignment = assignment
        self._allow_clear = allow_clear
        self.labelProvenance.setText(
            _provenance_text(assignment, note) if assignment is not None else note
        )
        self._populate_table()

    def clear(self, message: str = "") -> None:
        self.set_assignment(None, note=message)

    # -- table ------------------------------------------------------------

    def _populate_table(self, *_) -> None:
        table = self.tableResults
        table.setRowCount(0)
        self._order = []

        a = self._assignment
        if a is not None:
            key = _SORT_KEYS.get(
                self.comboSortBy.currentText().strip().lower(),
                _SORT_KEYS['posterior'],
            )
            self._order = sorted(
                range(len(a.candidates)), key=lambda i: key(a.candidates[i])
            )
            table.setRowCount(len(self._order))
            for row, idx in enumerate(self._order):
                self._fill_row(row, a.candidates[idx], chosen=idx == a.chosen_idx)

            # Start on the accepted candidate, so reviewing shows the current pick
            if a.chosen_idx is not None and a.chosen_idx in self._order:
                chosen_row = self._order.index(a.chosen_idx)
                table.selectRow(chosen_row)
                table.scrollToItem(table.item(chosen_row, 0))

        self._update_buttons()

    def _fill_row(
        self,
        row: int,
        c: 'AssignedCandidate',
        chosen: bool,
    ) -> None:
        formula_html = format_formula_obj_to_html(Formula(c.formula_str))
        if chosen:
            formula_html = f"<b>{formula_html}</b>"
        if c.manual:
            formula_html += " <i>(manual)</i>"
        values = [
            formula_html,
            c.adduct or "",
            _fmt(c.error_ppm, ".2f"),
            _fmt(c.error_da, ".6f"),
            _fmt(c.rdbe, ".1f"),
            _fmt(c.mass_loglik, ".2f"),
            _fmt(c.iso_loglik, ".2f"),
            _fmt(c.chem_logprior, ".2f"),
            _fmt(c.ms2_loglik, ".2f"),
            _fmt(c.log_posterior, ".2f"),
        ]
        for col, text in enumerate(values):
            item = QtWidgets.QTableWidgetItem(text)
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            if chosen:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            tips = (["Currently assigned"] if chosen else []) + (
                ["Entered manually"] if c.manual else []
            )
            if tips:
                item.setToolTip("; ".join(tips))
            self.tableResults.setItem(row, col, item)

    def _selected_candidate_idx(self) -> Optional[int]:
        rows = self.tableResults.selectionModel().selectedRows()
        if not rows:
            return None
        return self._order[rows[0].row()]

    def _update_buttons(self) -> None:
        a = self._assignment
        idx = self._selected_candidate_idx()
        self.btnAssignSelected.setEnabled(
            idx is not None and (a is None or idx != a.chosen_idx)
        )
        self.btnClearAssignment.setVisible(self._allow_clear)
        self.btnClearAssignment.setEnabled(
            a is not None and a.chosen_idx is not None
        )

    def _on_assign(self, *_) -> None:
        idx = self._selected_candidate_idx()
        if idx is not None:
            self.sigAssignRequested.emit(idx)


def _provenance_text(a: 'FormulaAssignment', note: str = "") -> str:
    """
    One line saying what was searched and how, e.g.
    'm/z 356.0126 · z=1 · adduct H · searched CHNOSClBr · Cl/Br detected ·
     225 candidates'.
    """
    parts = []
    if a.precursor_mz is not None:
        parts.append(f"m/z {a.precursor_mz:.4f}")
    if a.charge is not None:
        parts.append(f"z={a.charge}")
    if a.adducts:
        parts.append("adduct " + ", ".join(str(x) for x in a.adducts))
    if a.elements:
        parts.append(f"searched {a.elements}")
    if a.halogen_detected is True:
        parts.append("Cl/Br detected")
    elif a.halogen_detected is False:
        parts.append("no Cl/Br pattern")
    n = len(a.candidates)
    parts.append(f"{n} candidate{'' if n == 1 else 's'}")
    if a.chosen is not None and a.chosen_by is not None:
        parts.append(
            "picked by you" if a.chosen_by == 'user' else "picked automatically"
        )
    if a.source_kind != 'signal':
        # Registered assignments: when it was made helps when reviewing
        parts.append(
            datetime.fromtimestamp(a.created_at).strftime("%Y-%m-%d %H:%M")
        )
    if note:
        parts.append(note)
    return " · ".join(parts)


def _fmt(value, spec: str) -> str:
    """Format a numeric score term for the table, or '' when it is None."""
    if value is None:
        return ""
    return format(value, spec)
