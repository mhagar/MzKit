"""
Structure search: query the compound databases for the host's AnnotationTarget
(by its formula, neutral mass, or text: name / organism / ID / NP class), or
take a typed SMILES, rank the hits by GLACIER
(core/cli/rank_candidates.py; run in the background by MainController) and
accept one as a Hypothesis (sigAccepted).

The host mirrors the selected candidate's prediction against the target's MS2;
hovering a predicted peak highlights the fragment GLACIER cut out for it
(`highlight_fragment`). Rankings are cached per target key for the session.
"""
from typing import Optional, TYPE_CHECKING

import numpy as np
from PyQt5 import QtCore, QtWidgets

from core.cli.rank_candidates import (
    DbCandidate, fetch_candidates, neutral_mass, rank_params_from_config,
)
from core.data_structs.ensemble import USER, Hypothesis
from core.utils import compound_db, glacier
from core.utils.config import load_config
from gui.dialogues.CompoundDbDialog import NumberItem
from gui.resources.AnnotationPanel import Ui_AnnotationPanel
from gui.widgets.StructureWidget import NAME_ROLE, StructureDelegate

if TYPE_CHECKING:
    from core.cli.rank_candidates import RankResult, ScoredCandidate
    from core.utils.array_types import SpectrumArray
    from gui.widgets.annotation_target import AnnotationTarget
    from gui.widgets.StructureWidget import StructureView

ROW_HEIGHT = 80
# Candidate row's first cell: its index into RankResult.scored
SCORED_ROLE = QtCore.Qt.UserRole + 1
# Query modes: the .ui's Formula and Mass, then a typed SMILES and compound_db's text fields
FORMULA, MASS, SMILES = 0, 1, 2
_PLACEHOLDERS = {FORMULA: "e.g. C20H34O2", MASS: "neutral mass, e.g. 359.2672", SMILES: "SMILES"}


class StructureSearchWidget(
    QtWidgets.QWidget,
    Ui_AnnotationPanel,
):
    # (target key, rank_candidates kwargs): MainController runs it and hands the
    # RankResult back to `set_ranking`
    sigRankRequested = QtCore.pyqtSignal(object, dict)

    # The selected candidate to mirror (ScoredCandidate), or None
    sigCandidateSelected = QtCore.pyqtSignal(object)
    sigAccepted = QtCore.pyqtSignal(object)   # Hypothesis

    def __init__(self, parent=None):
        super().__init__(parent)
        self._target: Optional['AnnotationTarget'] = None
        self.ms2: Optional['SpectrumArray'] = None
        self._precursor_mz: Optional[float] = None
        self._adduct: Optional[str] = None

        # The target's key (None: no target)
        self._key = None
        # focus key -> (RankResult, status text); and those still running
        self._rankings: dict = {}
        self._pending: dict = {}   # focus key -> status text once done
        self._shown: Optional['ScoredCandidate'] = None

        self.setupUi(self)
        # Short names for the .ui's widgets
        self.mode_combo, self.query_edit, self.rank_btn = self.comboMode, self.editQuery, self.btnRank
        self.status_label, self.accept_btn = self.labelStatus, self.btnAccept

        self.rank_btn.clicked.connect(self._rank)
        self.query_edit.returnPressed.connect(self._rank)
        self.mode_combo.addItem("SMILES")
        self.mode_combo.addItems(list(compound_db.TEXT_FIELDS))
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._params_sheet = None
        self.btnParams.clicked.connect(self._show_params)

        self.source_checks: list[QtWidgets.QCheckBox] = []
        self.refresh_sources()

        self.table.setItemDelegateForColumn(0, StructureDelegate(self.table))
        self.table.setColumnWidth(0, ROW_HEIGHT * 3 // 2)
        self.table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.table.itemSelectionChanged.connect(self._on_row_selected)

        self.structure.enlarge_on_hover = False   # it's already the big view
        self.structure.hide()
        # Candidates over the selected one's structure
        self.splitterCandidates.setSizes([300, 200])
        self.accept_btn.clicked.connect(self._accept)

        self.set_target(None)

    def refresh_sources(self):
        """Re-read the compound databases' sources (e.g. after the cache was rebuilt)."""
        for check in self.source_checks:
            check.deleteLater()
        self.source_checks = []
        while self.layoutSources.count():
            self.layoutSources.takeAt(0)
        for source in compound_db.available_sources() if compound_db.cache_ok() else []:
            check = QtWidgets.QCheckBox(source)
            check.setChecked(True)
            self.source_checks.append(check)
            self.layoutSources.addWidget(check)
        self.layoutSources.addStretch(1)

    def _on_mode_changed(self):
        mode = self.mode_combo.currentIndex()
        self.query_edit.setPlaceholderText(_PLACEHOLDERS.get(mode, "contains…"))

    def _show_params(self):
        from gui.widgets.search_param_sheet import glacier_sheet
        if self._params_sheet is None:
            self._params_sheet = glacier_sheet(self)
        self._params_sheet.show()
        self._params_sheet.raise_()

    # -- target -------------------------------------------------------------


    @property
    def precursor_mz(self) -> Optional[float]:
        """The focus's precursor m/z (None when nothing is focused)."""
        return self._precursor_mz

    def set_target(self, target: Optional['AnnotationTarget']):
        """
        Follow the host's focus (None clears). Rank uses the target's MS2 and
        precursor, and the query is prefilled with its formula and adduct.
        Each target key keeps its own ranking; re-showing the same key keeps
        its state (a typed query included).
        """
        key = None if target is None else target.key
        if key is not None and key == self._key:
            self._target = target
            return
        self._target, self._key = target, key
        self.ms2, self._precursor_mz = (None, None) if target is None else (target.ms2, target.precursor_mz)
        self._adduct = None if target is None else target.adduct
        formula = None if target is None else target.formula
        with QtCore.QSignalBlocker(self.mode_combo):
            self.mode_combo.setCurrentIndex(FORMULA if formula or target is None else MASS)
        self._on_mode_changed()
        self.query_edit.setText(
            formula if formula
            else "" if target is None or self._precursor_mz is None
            else f"{neutral_mass(self._precursor_mz, self._adduct):.4f}"
        )

        if target is None:
            problem = "Focus an analyte or open an ensemble to annotate it"
        elif not glacier.model_available():
            problem = glacier.missing_model_message()
        elif self.ms2 is None or not self.ms2.size:
            problem = f"This {target.noun} has no MS2"
        else:
            problem = ""
        for w in (self.mode_combo, self.query_edit, self.rank_btn):
            w.setEnabled(not problem)

        if problem:
            self._show_ranking(None, problem)
        elif key in self._rankings:
            self._show_ranking(*self._rankings[key])
        elif key in self._pending:
            self._show_ranking(None, "Ranking…")
        else:
            self._show_ranking(None, self._adduct_status())

    def _adduct_status(self) -> str:
        return f"Adduct: {self._adduct or 'unknown ([M+H]+ assumed)'}"

    # -- ranking ------------------------------------------------------------

    def _rank(self):
        key = self._key
        if key is None or not self.rank_btn.isEnabled() or key in self._pending:
            return
        params = rank_params_from_config(load_config())   # picks up config edits
        sources = [c.text() for c in self.source_checks if c.isChecked()]
        text = self.query_edit.text().strip()
        mode = self.mode_combo.currentIndex()
        if mode == SMILES:
            from rdkit import Chem
            if not text or Chem.MolFromSmiles(text) is None:
                self.status_label.setText(f"Not a valid SMILES: {text!r}")
                return
            self._pending[key] = "Typed SMILES"
            self._show_ranking(None, "Predicting…")
            self.sigRankRequested.emit(key, self._rank_kwargs([DbCandidate(USER, "", "", text, "")], params))
            return
        if not self.source_checks:
            self.status_label.setText(
                "No compound database cache: build it from Tools > Libraries & databases"
            )
            return
        try:
            if mode == FORMULA:
                candidates, capped = fetch_candidates(sources, formula=text)
            elif mode > SMILES:
                if not text:
                    raise ValueError("Type something to search for")
                candidates, capped = fetch_candidates(
                    sources, text_field=self.mode_combo.currentText(), text=text,
                )
            else:
                try:
                    mass = float(text)
                except ValueError:
                    raise ValueError(f"Not a mass: {text!r}") from None
                candidates, capped = fetch_candidates(sources, mass=mass, ppm=params.mass_ppm)
        except ValueError as e:
            self.status_label.setText(str(e))
            return
        query = f"{text} in {', '.join(sources) or 'no database'}"
        if not candidates:
            self._rankings.pop(key, None)
            self._show_ranking(None, f"No hits for {query}")
            return
        self._pending[key] = (
            f"{len(candidates)} distinct hits for {query}"
            + (" (capped: narrow the search)" if capped else "")
        )
        self._show_ranking(None, f"Ranking {len(candidates)} candidates…")
        self.sigRankRequested.emit(key, self._rank_kwargs(candidates, params))

    def _rank_kwargs(self, candidates: list[DbCandidate], params) -> dict:
        return {
            'ms2': self.ms2,
            'precursor_mz': self._precursor_mz,
            'adduct': self._adduct,
            'candidates': candidates,
            'params': params,
        }

    @property
    def focus_key(self) -> Optional[tuple]:
        return self._key

    @property
    def shown(self) -> Optional['ScoredCandidate']:
        """The selected candidate."""
        return self._shown

    def score_kwargs(self, smiles: str, name: str = "", other: Optional['AnnotationTarget'] = None) -> Optional[dict]:
        """
        rank_candidates kwargs scoring one structure against the target's MS2
        (or `other`'s own MS2, precursor and adduct); None if it can't be.
        """
        target = other if other is not None else self._target
        if target is None or target.ms2 is None or not target.ms2.size or not glacier.model_available():
            return None
        kwargs = self._rank_kwargs([DbCandidate(USER, "", name, smiles, "")], rank_params_from_config(load_config()))
        kwargs.update(ms2=target.ms2, precursor_mz=target.precursor_mz, adduct=target.adduct)
        return kwargs

    def set_ranking(self, key: tuple, result: Optional['RankResult']):
        """A finished (None: failed / cancelled) ranking for focus `key` (see sigRankRequested)."""
        status = self._pending.pop(key, "")
        if result is None:
            status = "Ranking failed or was cancelled: see the process log"
            self._rankings.pop(key, None)
        else:
            status += f"; predicted as {result.adduct}"
            if result.adduct_fallback:
                status += f" (GLACIER doesn't know {self._adduct or 'the adduct'})"
            if result.skipped:
                status += f"; {len(result.skipped)} GLACIER couldn't predict"
            self._rankings[key] = (result, status)
        if key == self._key:
            self._show_ranking(result, status)

    def result(self) -> Optional['RankResult']:
        entry = self._rankings.get(self._key)
        return entry[0] if entry else None

    def _show_ranking(self, result: Optional['RankResult'], status: str):
        self.status_label.setText(status)
        with QtCore.QSignalBlocker(self.table):
            self.table.clearSelection()
            self.table.setSortingEnabled(False)
            self.table.horizontalHeader().setSortIndicator(-1, QtCore.Qt.AscendingOrder)  # ranked order
            scored = result.scored if result else []
            self.table.setRowCount(len(scored))
            for row, s in enumerate(scored):
                c = s.candidate
                structure = QtWidgets.QTableWidgetItem()
                structure.setData(QtCore.Qt.UserRole, c.smiles)
                structure.setData(SCORED_ROLE, row)
                structure.setData(NAME_ROLE, c.label)
                name = QtWidgets.QTableWidgetItem(c.label)
                score = NumberItem(f"{s.explained}/{result.n_considered} ({s.cosine:.2f})")
                score.setData(QtCore.Qt.UserRole, (s.explained, s.cosine))   # the ranking's own order
                organism = QtWidgets.QTableWidgetItem(c.organism)
                organism.setToolTip(c.organisms.replace('|', '; '))
                for col, item in enumerate((structure, name, QtWidgets.QTableWidgetItem(c.formula),
                                            score,
                                            QtWidgets.QTableWidgetItem(f"{c.source}:{c.db_id}"), organism)):
                    self.table.setItem(row, col, item)
            self.table.setSortingEnabled(True)
        self._on_row_selected()

    # -- candidate ----------------------------------------------------------

    def _selected(self) -> Optional['ScoredCandidate']:
        result, rows = self.result(), self.table.selectionModel().selectedRows()
        if result is None or not rows:
            return None
        return result.scored[self.table.item(rows[0].row(), 0).data(SCORED_ROLE)]

    def _on_row_selected(self):
        self._shown = self._selected()
        self.accept_btn.setEnabled(self._shown is not None)
        self.highlight_fragment(None)
        self.sigCandidateSelected.emit(self._shown)

    def highlight_fragment(self, peak_idx: Optional[int]):
        """Draw the shown candidate, with predicted peak `peak_idx`'s fragment lit."""
        if self._shown is None:
            self.structure.hide()
            return
        show_fragment(self.structure, self._shown, peak_idx)
        self.structure.show()

    def reselect(self):
        """Re-announce the selected candidate (its tab came back to the front)."""
        self.sigCandidateSelected.emit(self._shown)

    def _accept(self):
        s, result = self._shown, self.result()
        if s is None or result is None or self._target is None:
            return
        c = s.candidate
        self.sigAccepted.emit(Hypothesis(
            c.smiles, c.label, origin=c.source, db_id=c.db_id,
            scores=scores_of(s, result),
        ))


def scores_of(s: 'ScoredCandidate', result: 'RankResult') -> dict:
    """A ranked structure's Hypothesis.scores."""
    return {
        'explained': f"{s.explained}/{result.n_considered}",
        'cosine': round(s.cosine, 4),
        'adduct': result.adduct,
        'collision_energy': result.params.collision_energy,
        'instrument': result.params.instrument,
    }


def show_fragment(view: 'StructureView', s: 'ScoredCandidate', peak_idx: Optional[int]):
    """Draw `s` in `view`, with predicted peak `peak_idx`'s fragment lit and its ion in the corner."""
    atoms = None if peak_idx is None else np.flatnonzero(s.frag_atoms[peak_idx]).tolist()
    view.set_structure(s.candidate.smiles, s.candidate.label, atoms)
    # The lit peak's ion and GLACIER's H shift on its fragment
    shift = None if peak_idx is None or not s.h_shifts else s.h_shifts[peak_idx]
    view.set_caption("" if shift is None else f"{s.ions[peak_idx]}\n{f'{shift:+d}' if shift else '±0'}H")
