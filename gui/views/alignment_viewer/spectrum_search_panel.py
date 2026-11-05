"""
The Inspector's spectral-library panel: search the spectral libraries
(core/utils/spectrum_db.py) for the focused analyte's MS2, by identity
(cosine within the precursor window) or by analogue (embedding similarity;
core/cli/search_spectra.py, run in the background by MainController), and
accept a hit as a Hypothesis of the focused ensemble's EnsembleAnnotation.

The viewer mirrors the selected hit's library spectrum against the focus's
MS2 in the bottom MS2 plot. Results are cached per (focus, mode) for the
session.
"""
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from PyQt5 import QtCore, QtWidgets

from core.cli.search_spectra import IDENTITY, params_from_config
from core.data_structs.ensemble import Hypothesis
from core.utils import spectrum_db
from core.utils.config import load_config
from gui.dialogues.CompoundDbDialog import NumberItem
from gui.resources.SpectrumSearchPanel import Ui_SpectrumSearchPanel
from gui.views.alignment_viewer.annotation_editor import _parses
from gui.widgets.StructureWidget import NAME_ROLE, StructureDelegate

if TYPE_CHECKING:
    from core.cli.search_spectra import LibraryHit, SearchResult
    from core.data_structs.alignment import AlignedAnalyte
    from core.data_structs.ensemble import Ensemble
    from core.utils.array_types import SpectrumArray
    from gui.views.alignment_viewer.context import AlignmentContext

ROW_HEIGHT = 80
HIT_ROLE = QtCore.Qt.UserRole + 1   # a row's first cell: its index into SearchResult.hits


class SpectrumSearchPanel(QtWidgets.QWidget, Ui_SpectrumSearchPanel):
    # ((focus key, mode), search_spectra kwargs): MainController runs it and hands
    # the SearchResult back to `set_result`. The focus key is (AnalyteUUID, EnsembleUUID or None)
    sigSearchRequested = QtCore.pyqtSignal(object, dict)
    sigRebuildRequested = QtCore.pyqtSignal()   # MainController calls `refresh_index` when done
    sigHitSelected = QtCore.pyqtSignal(object)  # the selected LibraryHit to mirror, or None
    sigAccepted = QtCore.pyqtSignal(object, object)   # AnalyteUUID, Hypothesis

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctx: Optional['AlignmentContext'] = None
        self._analyte: Optional['AlignedAnalyte'] = None
        self.ms2: Optional['SpectrumArray'] = None
        self._precursor_mz: Optional[float] = None
        self._key: Optional[tuple] = None   # the focus: (AnalyteUUID, EnsembleUUID or None)
        self._results: dict = {}   # (focus key, mode) -> (SearchResult, status text)
        self._pending: set = set()
        self._shown: Optional['LibraryHit'] = None
        # Imported embedder -> {analyte uuid str: vector} (session only)
        self._query_embeddings: dict[str, dict] = {}

        self.setupUi(self)
        self.btnSearch.clicked.connect(self._search)
        self.btnRebuild.clicked.connect(self._rebuild)
        self.btnImportEmbeddings.clicked.connect(self._import_embeddings)
        self.btnAccept.clicked.connect(self._accept)
        self.comboMode.currentIndexChanged.connect(self._show_current)

        self.library_checks: list[QtWidgets.QCheckBox] = []
        self.table.setItemDelegateForColumn(0, StructureDelegate(self.table))
        self.table.setColumnWidth(0, ROW_HEIGHT * 3 // 2)
        self.table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        self.structure.enlarge_on_hover = False
        self.structure.hide()
        self.splitterCandidates.setSizes([300, 200])

        self.refresh_index()

    # -- index --------------------------------------------------------------


    @property
    def precursor_mz(self) -> Optional[float]:
        """The focus's precursor m/z (None when nothing is focused)."""
        return self._precursor_mz

    def refresh_index(self):
        """Re-read the libraries and embedders (after a rebuild); cached results are dropped."""
        self._results.clear()
        for check in self.library_checks:
            check.deleteLater()
        self.library_checks = []
        while self.layoutLibraries.count():
            self.layoutLibraries.takeAt(0)
        ok = spectrum_db.cache_ok()
        for library in spectrum_db.available_libraries() if ok else []:
            check = QtWidgets.QCheckBox(library)
            check.setChecked(True)
            self.library_checks.append(check)
            self.layoutLibraries.addWidget(check)
        self.layoutLibraries.addStretch(1)
        self.btnRebuild.setEnabled(True)
        self._fill_modes()
        self._refresh_state()

    def _fill_modes(self):
        """Identity, then each embedder this focus can query (imported ones need query embeddings)."""
        current = self.comboMode.currentData()
        with QtCore.QSignalBlocker(self.comboMode):
            self.comboMode.clear()
            self.comboMode.addItem("Identity (cosine)", IDENTITY)
            for embedder in spectrum_db.embedders() if spectrum_db.cache_ok() else []:
                if embedder == spectrum_db.MS2DEEPSCORE:
                    self.comboMode.addItem("Analogue (MS2DeepScore)", embedder)
                elif self._query_embedding(embedder) is not None:
                    self.comboMode.addItem(f"Analogue ({embedder})", embedder)
            self.comboMode.setCurrentIndex(max(self.comboMode.findData(current), 0))

    def _rebuild(self):
        self.btnRebuild.setEnabled(False)
        self.labelStatus.setText(f"Rebuilding the index from {spectrum_db.DB_DIR} in the background…")
        self.sigRebuildRequested.emit()

    def _import_embeddings(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import query embeddings (<name>.<embedder>.npz)", "", "Embeddings (*.npz)",
        )
        if not path:
            return
        embedder = Path(path).stem.rpartition('.')[2]
        try:
            ids, embeddings = spectrum_db.read_embeddings_npz(Path(path))
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Import query embeddings", str(e))
            return
        if embedder not in spectrum_db.embedders():
            QtWidgets.QMessageBox.warning(
                self, "Import query embeddings",
                f"No library embeddings for {embedder!r}: add <library>.{embedder}.npz to "
                f"{spectrum_db.DB_DIR} and rebuild the index",
            )
            return
        self._query_embeddings[embedder] = dict(zip(ids, embeddings))
        self._fill_modes()
        self.comboMode.setCurrentIndex(self.comboMode.findData(embedder))
        self._refresh_state()

    # -- focus --------------------------------------------------------------

    def show_analyte(
        self,
        ctx: Optional['AlignmentContext'],
        analyte: Optional['AlignedAnalyte'],
        ensemble: Optional['Ensemble'] = None,
    ):
        """Follow the focus (None clears): the analyte's representative MS2, or `ensemble`'s own."""
        key = None if analyte is None else (analyte.uuid, ensemble and ensemble.uuid)
        self._ctx = ctx
        if key is not None and key == self._key:
            return
        self._analyte, self._key = analyte, key
        if analyte is None:
            self.ms2, self._precursor_mz = None, None
        elif ensemble is not None:
            self.ms2, self._precursor_mz = ensemble.composite_spectrum.ms2, ensemble.resolved_precursor_mz
        else:
            rep = ctx.representative(analyte)
            self.ms2, self._precursor_mz = (None if rep is None else rep.composite.ms2), analyte.consensus_mz
        self._fill_modes()
        self._refresh_state()

    def _query_embedding(self, embedder: str):
        """The focus's imported embedding (analyte focus only: the consensus .mgf is per analyte)."""
        if self._key is None or self._key[1] is not None:
            return None
        return self._query_embeddings.get(embedder, {}).get(str(self._key[0]))

    def _problem(self) -> str:
        if self._analyte is None:
            return "Focus an analyte to search the spectral libraries"
        if not self.library_checks:
            return f"No spectral library index: add .mgf files to {spectrum_db.DB_DIR} and Rebuild index"
        if self.ms2 is None or not self.ms2.size:
            return f"This {'ensemble' if self._key[1] is not None else 'analyte'} has no MS2"
        return ""

    def _refresh_state(self):
        problem = self._problem()
        for w in (self.comboMode, self.btnSearch):
            w.setEnabled(not problem)
        self.btnImportEmbeddings.setEnabled(bool(self.library_checks))
        self._show_current(problem=problem)

    def _show_current(self, *_, problem: Optional[str] = None):
        problem = self._problem() if problem is None else problem
        slot = (self._key, self.comboMode.currentData())
        if problem:
            self._show_result(None, problem)
        elif slot in self._results:
            self._show_result(*self._results[slot])
        elif slot in self._pending:
            self._show_result(None, "Searching…")
        else:
            self._show_result(None, f"Precursor m/z {self._precursor_mz:.4f}")

    # -- search -------------------------------------------------------------

    def _search(self):
        mode = self.comboMode.currentData()
        slot = (self._key, mode)
        if self._problem() or slot in self._pending:
            return
        libraries = [c.text() for c in self.library_checks if c.isChecked()]
        if not libraries:
            self.labelStatus.setText("Tick at least one library")
            return
        kwargs = {
            'ms2': self.ms2,
            'precursor_mz': float(self._precursor_mz),
            'libraries': libraries,
            'mode': mode,
            'params': params_from_config(load_config()),   # picks up config edits
        }
        if mode not in (IDENTITY, spectrum_db.MS2DEEPSCORE):
            kwargs['query_embedding'] = self._query_embedding(mode)
        self._pending.add(slot)
        self._show_result(None, "Searching…")
        self.sigSearchRequested.emit(slot, kwargs)

    @property
    def shown(self) -> Optional['LibraryHit']:
        return self._shown

    def set_result(self, slot: tuple, result: Optional['SearchResult']):
        """A finished (None: failed / cancelled) search for `slot` (see sigSearchRequested)."""
        self._pending.discard(slot)
        if result is None:
            self._results.pop(slot, None)
            status = "Search failed or was cancelled: see the process log"
        else:
            status = f"{len(result.hits)} hits of {result.n_searched} library spectra searched"
            self._results[slot] = (result, status)
        if slot == (self._key, self.comboMode.currentData()):
            self._show_result(result, status)

    def result(self) -> Optional['SearchResult']:
        entry = self._results.get((self._key, self.comboMode.currentData()))
        return entry[0] if entry else None

    def _show_result(self, result: Optional['SearchResult'], status: str):
        self.labelStatus.setText(status)
        with QtCore.QSignalBlocker(self.table):
            self.table.clearSelection()
            self.table.setSortingEnabled(False)
            self.table.horizontalHeader().setSortIndicator(-1, QtCore.Qt.AscendingOrder)   # ranked order
            hits = result.hits if result else []
            self.table.setRowCount(len(hits))
            for row, h in enumerate(hits):
                structure = QtWidgets.QTableWidgetItem()
                structure.setData(QtCore.Qt.UserRole, h.smiles)
                structure.setData(HIT_ROLE, row)
                structure.setData(NAME_ROLE, h.name)
                for col, item in enumerate((
                    structure, QtWidgets.QTableWidgetItem(h.name),
                    _number(f"{h.score:.3f}", h.score),
                    _number(f"{h.mod_cosine:.2f} ({h.n_matched})", (h.mod_cosine, h.n_matched)),
                    _number(f"{h.delta_mz:+.4f}", h.delta_mz), QtWidgets.QTableWidgetItem(h.adduct),
                    QtWidgets.QTableWidgetItem(f"{h.library}:{h.spectrum_id}"),
                )):
                    self.table.setItem(row, col, item)
            self.table.setSortingEnabled(True)
        self._on_row_selected()

    # -- hit ----------------------------------------------------------------

    def _selected(self) -> Optional['LibraryHit']:
        result, rows = self.result(), self.table.selectionModel().selectedRows()
        if result is None or not rows:
            return None
        return result.hits[self.table.item(rows[0].row(), 0).data(HIT_ROLE)]

    def _on_row_selected(self):
        self._shown = self._selected()
        # Hypotheses need a structure: library entries without a valid SMILES can't be accepted
        self.btnAccept.setEnabled(self._shown is not None and _parses(self._shown.smiles))
        if self._shown is None or not self._shown.smiles:
            self.structure.hide()
        else:
            self.structure.set_structure(self._shown.smiles, self._shown.name)
            self.structure.show()
        self.sigHitSelected.emit(self._shown)

    def reselect(self):
        """Re-announce the selected hit (its tab came back to the front)."""
        self.sigHitSelected.emit(self._shown)

    def _accept(self):
        h, result, analyte = self._shown, self.result(), self._analyte
        if h is None or result is None or analyte is None:
            return
        self.sigAccepted.emit(analyte.uuid, Hypothesis(
            h.smiles, h.name, origin=h.library,
            claim='structure' if result.mode == IDENTITY else 'analogue',
            db_id=h.spectrum_id,
            scores={
                'cosine' if result.mode == IDENTITY else result.mode: round(h.score, 4),
                'mod_cosine': round(h.mod_cosine, 4),
                'matched': h.n_matched,
                'delta_mz': round(h.delta_mz, 4),
            },
        ))


def _number(text: str, value) -> NumberItem:
    item = NumberItem(text)
    item.setData(QtCore.Qt.UserRole, value)
    return item
