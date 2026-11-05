"""
Spectral search: search the spectral libraries (core/utils/spectrum_db.py) for
the host's AnnotationTarget's MS2, by identity (cosine within the precursor
window) or by analogue (embedding similarity; core/cli/search_spectra.py, run
in the background by MainController), and accept a hit as a Hypothesis
(sigAccepted).

The host mirrors the selected hit's library spectrum against the target's MS2.
Results are cached per (target key, mode) for the session. The index is
rebuilt, and query embeddings imported, from Tools > Libraries & databases.
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
from gui.widgets.annotation_editor import _parses
from gui.widgets.StructureWidget import NAME_ROLE, StructureDelegate

if TYPE_CHECKING:
    from core.cli.search_spectra import LibraryHit, SearchResult
    from core.utils.array_types import SpectrumArray
    from gui.widgets.annotation_target import AnnotationTarget

ROW_HEIGHT = 80
HIT_ROLE = QtCore.Qt.UserRole + 1   # a row's first cell: its index into SearchResult.hits

# Imported embedder -> {embedding id (analyte uuid str): vector}: session only,
# shared by every host's spectral search
QUERY_EMBEDDINGS: dict[str, dict] = {}


def import_query_embeddings(path: Path) -> str:
    """
    Load <name>.<embedder>.npz (ids, embeddings) into QUERY_EMBEDDINGS.

    :return: the embedder.
    :raises ValueError: unreadable, or no library embeddings for that embedder.
    """
    embedder = Path(path).stem.rpartition('.')[2]
    ids, embeddings = spectrum_db.read_embeddings_npz(Path(path))
    if embedder not in spectrum_db.embedders():
        raise ValueError(
            f"No library embeddings for {embedder!r}: add <library>.{embedder}.npz to "
            f"{spectrum_db.DB_DIR} and rebuild the index"
        )
    QUERY_EMBEDDINGS[embedder] = dict(zip(ids, embeddings))
    return embedder


class SpectralSearchWidget(QtWidgets.QWidget, Ui_SpectrumSearchPanel):
    # ((target key, mode), search_spectra kwargs): MainController runs it and hands
    # the SearchResult back to `set_result`
    sigSearchRequested = QtCore.pyqtSignal(object, dict)
    sigHitSelected = QtCore.pyqtSignal(object)  # the selected LibraryHit to mirror, or None
    sigAccepted = QtCore.pyqtSignal(object)     # Hypothesis

    def __init__(self, parent=None):
        super().__init__(parent)
        self._target: Optional['AnnotationTarget'] = None
        self.ms2: Optional['SpectrumArray'] = None
        self._precursor_mz: Optional[float] = None
        self._key = None   # the target's key
        self._results: dict = {}   # (focus key, mode) -> (SearchResult, status text)
        self._pending: set = set()
        self._shown: Optional['LibraryHit'] = None
        self._params_sheet = None

        self.setupUi(self)
        self.btnSearch.clicked.connect(self._search)
        self.btnParams.clicked.connect(self._show_params)
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
        self.refresh_modes()

    def refresh_modes(self):
        """Re-offer the analogue modes (e.g. after query embeddings were imported)."""
        self._fill_modes()
        self._refresh_state()

    def _show_params(self):
        from gui.widgets.search_param_sheet import spectral_sheet
        if self._params_sheet is None:
            self._params_sheet = spectral_sheet(self)
        self._params_sheet.show()
        self._params_sheet.raise_()

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

    # -- focus --------------------------------------------------------------

    def set_target(self, target: Optional['AnnotationTarget']):
        """Follow the host's focus (None clears): search the target's MS2 at its precursor."""
        key = None if target is None else target.key
        self._target = target
        if key is not None and key == self._key:
            return
        self._key = key
        self.ms2, self._precursor_mz = (None, None) if target is None else (target.ms2, target.precursor_mz)
        self._fill_modes()
        self._refresh_state()

    def _query_embedding(self, embedder: str):
        """The target's imported embedding (analytes only: the consensus .mgf is per analyte)."""
        embedding_id = None if self._target is None else self._target.embedding_id
        if embedding_id is None:
            return None
        return QUERY_EMBEDDINGS.get(embedder, {}).get(embedding_id)

    def _problem(self) -> str:
        if self._target is None:
            return "Focus an analyte or open an ensemble to search the spectral libraries"
        if not self.library_checks:
            return (f"No spectral library index: add .mgf files to {spectrum_db.DB_DIR} and build it "
                    "from Tools > Libraries & databases")
        if self.ms2 is None or not self.ms2.size:
            return f"This {self._target.noun} has no MS2"
        return ""

    def _refresh_state(self):
        problem = self._problem()
        for w in (self.comboMode, self.btnSearch):
            w.setEnabled(not problem)
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
        h, result = self._shown, self.result()
        if h is None or result is None or self._target is None:
            return
        self.sigAccepted.emit(Hypothesis(
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
