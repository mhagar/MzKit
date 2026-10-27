"""
Experimental tool: browse the cached structure databases (core/utils/compound_db.py)
by formula, neutral mass +/- ppm, or text (name, organism, ...).
"""
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtSvg import QSvgRenderer

from core.utils import compound_db
from core.utils.render_molecule import smiles_to_svg

HEADERS = ['Structure', 'Name', 'ID', 'SMILES', 'Formula', 'Exact mass', 'Δppm',
           'Organism type', 'Genus', 'Species', 'NP class']
STRUCTURE_ROW_HEIGHT = 160


class StructureDelegate(QtWidgets.QStyledItemDelegate):
    """Draws the SMILES in the cell's UserRole; only painted (visible) rows get rendered."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cache: dict[str, QSvgRenderer | None] = {}

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        smiles = index.data(QtCore.Qt.UserRole)
        if smiles not in self._cache:
            try:
                svg, _ = smiles_to_svg(smiles)
                renderer = QSvgRenderer(QtCore.QByteArray(svg.encode()))
                renderer.setAspectRatioMode(QtCore.Qt.KeepAspectRatio)
                self._cache[smiles] = renderer
            except ValueError:
                self._cache[smiles] = None
        if renderer := self._cache[smiles]:
            renderer.render(painter, QtCore.QRectF(option.rect.adjusted(2, 2, -2, -2)))


class NumberItem(QtWidgets.QTableWidgetItem):
    """Shows formatted text, sorts by the float in UserRole."""

    def __lt__(self, other):
        return self.data(QtCore.Qt.UserRole) < other.data(QtCore.Qt.UserRole)


class CompoundDbDialog(QtWidgets.QDialog):
    def __init__(self, process_controller, parent=None):
        super().__init__(parent)
        self.process_controller = process_controller
        self.setWindowTitle("Compound databases")
        self.resize(1100, 650)

        self.source_combo = QtWidgets.QComboBox()
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(['Formula', 'Neutral mass', *compound_db.TEXT_FIELDS])
        self.query_edit = QtWidgets.QLineEdit()
        self.ppm_spin = QtWidgets.QDoubleSpinBox()
        self.ppm_spin.setRange(0.1, 1000)
        self.ppm_spin.setValue(5)
        self.ppm_spin.setSuffix(" ppm")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.search_btn = QtWidgets.QPushButton("Search")
        self.search_btn.clicked.connect(self.search)
        self.search_btn.setDefault(True)  # Enter searches
        self.render_check = QtWidgets.QCheckBox("Render structures")
        self.render_check.toggled.connect(self._on_render_toggled)
        self.build_btn = QtWidgets.QPushButton()
        self.build_btn.clicked.connect(self.build_cache)
        self.build_btn.setAutoDefault(False)
        self.status_label = QtWidgets.QLabel()

        bar = QtWidgets.QHBoxLayout()
        for w in (self.source_combo, self.mode_combo, self.query_edit, self.ppm_spin,
                  self.search_btn, self.render_check, self.build_btn):
            bar.addWidget(w)
        bar.setStretchFactor(self.query_edit, 1)

        self.table = QtWidgets.QTableWidget(0, len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setItemDelegateForColumn(0, StructureDelegate(self.table))
        self.table.setColumnWidth(0, 220)
        self.table.setColumnHidden(0, True)
        self.table.horizontalHeader().setStretchLastSection(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(self.table)
        layout.addWidget(self.status_label)

        self._on_mode_changed()
        self._refresh_cache_state()

    def _refresh_cache_state(self):
        exists = compound_db.DEFAULT_DB.exists()
        ok = compound_db.cache_ok()
        self.build_btn.setText("Rebuild cache" if exists else "Build database cache")
        self.search_btn.setEnabled(ok)
        self.source_combo.clear()
        if ok:
            self.source_combo.addItems(compound_db.available_sources())
        elif exists:
            self.status_label.setText("Cache is from an older version: rebuild it")
        else:
            self.status_label.setText(
                f"No cache yet: build it from the dumps in {compound_db.DB_DIR}"
            )

    def _on_mode_changed(self):
        by_mass = self.mode_combo.currentIndex() == 1
        self.ppm_spin.setEnabled(by_mass)
        self.query_edit.setPlaceholderText(
            {0: "e.g. C20H34O2", 1: "e.g. 359.2672"}.get(self.mode_combo.currentIndex(), "contains…")
        )

    def _on_render_toggled(self, on: bool):
        self.table.setColumnHidden(0, not on)
        self.table.verticalHeader().setDefaultSectionSize(
            STRUCTURE_ROW_HEIGHT if on else self.fontMetrics().height() + 8
        )

    def build_cache(self):
        self.build_btn.setEnabled(False)
        self.search_btn.setEnabled(False)
        self.status_label.setText("Building cache in the background…")
        self.process_controller.start_process(
            module_path="core.utils.compound_db",
            function_name="build_db",
            on_completion_func=self._on_built,
        )

    def _on_built(self, result):
        self.build_btn.setEnabled(True)
        self._refresh_cache_state()
        self.status_label.setText("Cache built." if result else "Cache build failed: see the process log.")

    def search(self):
        if not self.search_btn.isEnabled():
            return
        source, text = self.source_combo.currentText(), self.query_edit.text().strip()
        try:
            mode = self.mode_combo.currentIndex()
            if mode == 0:
                rows = compound_db.search_formula(source, text)
            elif mode >= 2:
                if not text:
                    raise ValueError("Type something to search for")
                rows = compound_db.search_text(source, self.mode_combo.currentText(), text)
            else:
                try:
                    mass = float(text)
                except ValueError:
                    raise ValueError(f"Not a mass: {text!r}") from None
                rows = compound_db.search_mass(source, mass, self.ppm_spin.value())
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return

        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setSortIndicator(-1, QtCore.Qt.AscendingOrder)  # keep query order
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            db_id, name, smiles, formula, mass, _inchikey, org_type, genus, organisms, np_class, *dppm = row
            structure = QtWidgets.QTableWidgetItem()
            structure.setData(QtCore.Qt.UserRole, smiles)
            self.table.setItem(i, 0, structure)
            values = [name, db_id, smiles, formula, mass, dppm[0] if dppm else None,
                      org_type, genus.replace('|', '; '), organisms.replace('|', '; '), np_class]
            for col, value in enumerate(values, 1):
                if isinstance(value, float):
                    item = NumberItem(f"{value:.{4 if col == 5 else 2}f}")
                    item.setData(QtCore.Qt.UserRole, value)
                else:
                    item = QtWidgets.QTableWidgetItem(value or '')
                    if col in (8, 9):  # genus / species lists can be long
                        item.setToolTip(value)
                self.table.setItem(i, col, item)
        self.table.setSortingEnabled(True)
        capped = " (capped; narrow the search)" if len(rows) >= compound_db.MAX_RESULTS else ""
        self.status_label.setText(f"{len(rows)} hits in {source}{capped}")

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.matches(QtGui.QKeySequence.Copy):
            cells = sorted(self.table.selectedIndexes(), key=lambda ix: (ix.row(), ix.column()))
            lines: dict[int, list[str]] = {}
            for ix in cells:
                lines.setdefault(ix.row(), []).append(str(ix.data() or ix.data(QtCore.Qt.UserRole) or ''))
            QtWidgets.QApplication.clipboard().setText('\n'.join('\t'.join(v) for v in lines.values()))
        else:
            super().keyPressEvent(event)
