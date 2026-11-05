"""
The Inspector's annotation panel: query the compound databases for the focused
analyte (by its consensus formula, or neutral mass), rank the hits by GLACIER
(core/cli/rank_candidates.py; run in the background by MainController) and
accept one as a Hypothesis of the focused ensemble's EnsembleAnnotation.

The viewer mirrors the selected candidate's prediction against the analyte's
MS2 in the bottom MS2 plot; hovering a predicted peak highlights the fragment
GLACIER cut out for it (`highlight_fragment`). Rankings are cached per analyte
for the session.
"""
from typing import Optional, TYPE_CHECKING

import numpy as np
from PyQt5 import QtCore, QtWidgets

from core.cli.rank_candidates import (
    DbCandidate, fetch_candidates, ion_name, neutral_mass, rank_params_from_config,
)
from core.data_structs.ensemble import USER, Hypothesis
from core.utils import compound_db, glacier
from core.utils.config import load_config
from gui.dialogues.CompoundDbDialog import NumberItem
from gui.resources.AnnotationPanel import Ui_AnnotationPanel
from gui.widgets.StructureWidget import NAME_ROLE, StructureDelegate

if TYPE_CHECKING:
    from core.cli.rank_candidates import RankResult, ScoredCandidate
    from core.data_structs.alignment import AlignedAnalyte
    from core.data_structs.ensemble import Ensemble
    from core.utils.array_types import SpectrumArray
    from gui.views.alignment_viewer.context import AlignmentContext
    from gui.widgets.StructureWidget import StructureView

ROW_HEIGHT = 80
# Candidate row's first cell: its index into RankResult.scored
SCORED_ROLE = QtCore.Qt.UserRole + 1
FORMULA, MASS = 0, 1


class AnnotationPanel(
    QtWidgets.QWidget,
    Ui_AnnotationPanel,
):
    # (focus key, rank_candidates kwargs): MainController runs it and hands the
    # RankResult back to `set_ranking`. The key is (AnalyteUUID, EnsembleUUID or None)
    sigRankRequested = QtCore.pyqtSignal(object, dict)

    # The selected candidate to mirror (ScoredCandidate), or None
    sigCandidateSelected = QtCore.pyqtSignal(object)
    sigAccepted = QtCore.pyqtSignal(object, object)   # AnalyteUUID, Hypothesis

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctx: Optional['AlignmentContext'] = None
        self._analyte: Optional['AlignedAnalyte'] = None
        self.ms2: Optional['SpectrumArray'] = None
        self._precursor_mz: Optional[float] = None
        self._adduct: Optional[str] = None

        # Focused AnalyteUUID, the focused member's EnsembleUUID or None
        self._key: Optional[tuple] = None
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

        self.source_checks: list[QtWidgets.QCheckBox] = []
        if compound_db.cache_ok():
            for source in compound_db.available_sources():
                check = QtWidgets.QCheckBox(source)
                check.setChecked(True)
                self.source_checks.append(check)
                self.layoutSources.addWidget(check)
        self.layoutSources.addStretch(1)

        self.table.setItemDelegateForColumn(0, StructureDelegate(self.table))
        self.table.setColumnWidth(0, ROW_HEIGHT * 3 // 2)
        self.table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.table.itemSelectionChanged.connect(self._on_row_selected)

        self.structure.enlarge_on_hover = False   # it's already the big view
        self.structure.hide()
        # Candidates over the selected one's structure
        self.splitterCandidates.setSizes([300, 200])
        self.accept_btn.clicked.connect(self._accept)

        self.show_analyte(None, None)

    # -- analyte ------------------------------------------------------------


    @property
    def precursor_mz(self) -> Optional[float]:
        """The focus's precursor m/z (None when nothing is focused)."""
        return self._precursor_mz

    def show_analyte(
        self,
        ctx: Optional['AlignmentContext'],
        analyte: Optional['AlignedAnalyte'],
        ensemble: Optional['Ensemble'] = None,
    ):
        """
        Follow the focus (None clears): the analyte, or `ensemble`, one of its
        members. Rank uses the focus's own MS2 and precursor (the analyte's
        representative's / consensus for the analyte), and the query is
        prefilled with its formula and adduct. Each focus keeps its own
        ranking; re-focusing the same one keeps its state.
        """
        key = None if analyte is None else (analyte.uuid, ensemble and ensemble.uuid)
        if key is not None and key == self._key:
            self._ctx = ctx
            return
        self._ctx, self._analyte, self._key = ctx, analyte, key
        if analyte is None:
            self.ms2, self._precursor_mz = None, None
        elif ensemble is not None:
            self.ms2, self._precursor_mz = ensemble.composite_spectrum.ms2, ensemble.resolved_precursor_mz
        else:
            rep = ctx.representative(analyte)
            self.ms2, self._precursor_mz = (None if rep is None else rep.composite.ms2), analyte.consensus_mz
        self._fill_query(ensemble)

        if analyte is None:
            problem = "Focus an analyte to annotate it"
        elif not glacier.model_available():
            problem = glacier.missing_model_message()
        elif not self.source_checks:
            problem = "No compound database cache: build it from Tools > Browse compound databases"
        elif self.ms2 is None or not self.ms2.size:
            problem = f"This {'ensemble' if ensemble is not None else 'analyte'} has no MS2"
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

    def _fill_query(self, ensemble: Optional['Ensemble']):
        """Prefill the query (and adduct) from `ensemble`, else the analyte's members."""
        ctx, analyte = self._ctx, self._analyte
        if analyte is None:
            self._adduct, formula = None, None
        else:
            rep = ctx.representative(analyte)
            members = ctx.members(analyte)
            candidates = (
                [ensemble] if ensemble is not None
                else [members[rep.sample_uuid]] if rep and rep.sample_uuid in members else []
            )
            formula = ctx.focus_formula(analyte, ensemble)
            # The adduct goes with the formula: the focused member's, else any member's
            self._adduct = self._adduct_of(ctx, candidates + list(members.values()))
        with QtCore.QSignalBlocker(self.mode_combo):
            self.mode_combo.setCurrentIndex(FORMULA if formula or analyte is None else MASS)
        self.query_edit.setText(
            formula if formula
            else "" if analyte is None
            else f"{neutral_mass(self._precursor_mz, self._adduct):.4f}"
        )

    @staticmethod
    def _adduct_of(ctx, ensembles) -> Optional[str]:
        """The first of `ensembles` whose formula assignment has an adduct."""
        for ensemble in ensembles:
            assignment = ensemble.formula_assignment
            candidate = assignment and (assignment.chosen or assignment.top)
            if candidate and candidate.adduct:
                return ion_name(candidate.adduct, assignment.charge)
        return None

    # -- ranking ------------------------------------------------------------

    def _rank(self):
        key = self._key
        if key is None or not self.rank_btn.isEnabled() or key in self._pending:
            return
        params = rank_params_from_config(load_config())   # picks up config edits
        sources = [c.text() for c in self.source_checks if c.isChecked()]
        text = self.query_edit.text().strip()
        try:
            if self.mode_combo.currentIndex() == FORMULA:
                candidates, capped = fetch_candidates(sources, formula=text)
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

    def score_kwargs(self, smiles: str, name: str = "", ensemble: Optional['Ensemble'] = None) -> Optional[dict]:
        """
        rank_candidates kwargs scoring one structure against the focus's MS2
        (or `ensemble`'s own MS2, precursor and adduct); None if it can't be.
        """
        if ensemble is not None:
            ms2 = ensemble.composite_spectrum.ms2
        elif self._key is not None:
            ms2 = self.ms2
        else:
            return None
        if ms2 is None or not ms2.size or not glacier.model_available():
            return None
        kwargs = self._rank_kwargs([DbCandidate(USER, "", name, smiles, "")], rank_params_from_config(load_config()))
        if ensemble is not None:
            kwargs.update(ms2=ms2, precursor_mz=ensemble.resolved_precursor_mz,
                          adduct=self._adduct_of(self._ctx, [ensemble]))
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
        s, result, analyte = self._shown, self.result(), self._analyte
        if s is None or result is None or analyte is None:
            return
        c = s.candidate
        self.sigAccepted.emit(analyte.uuid, Hypothesis(
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
