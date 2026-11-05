"""
Tools > Libraries & databases: the state of the spectral library index
(spectra/*.mgf, core/utils/spectrum_db.py) and the compound database cache
(core/utils/compound_db.py), rebuilding either (run by MainController), and
importing query embeddings for an external embedder (session only, shared by
every spectral search tab).
"""
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from core.utils import compound_db, spectrum_db
from gui.widgets.spectral_search_widget import QUERY_EMBEDDINGS, import_query_embeddings


class LibrariesDialog(QtWidgets.QDialog):
    sigRebuildSpectral = QtCore.pyqtSignal()
    sigBuildCompounds = QtCore.pyqtSignal()
    sigEmbeddingsImported = QtCore.pyqtSignal(str)   # embedder

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Libraries & databases")
        self.resize(560, 0)

        self.spectral_label, self.compound_label = QtWidgets.QLabel(), QtWidgets.QLabel()
        for label in (self.spectral_label, self.compound_label):
            label.setWordWrap(True)
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.spectral_btn = QtWidgets.QPushButton()
        self.spectral_btn.setToolTip(
            "Rebuild the index from the .mgf files (and <library>.<embedder>.npz embeddings) in the spectra/ folder"
        )
        self.spectral_btn.clicked.connect(self._rebuild_spectral)
        self.embeddings_btn = QtWidgets.QPushButton("Import query embeddings…")
        self.embeddings_btn.setToolTip(
            "Load <name>.<embedder>.npz (ids, embeddings) for the analytes of the consensus .mgf from "
            "Export table, to search libraries embedded by that tool"
        )
        self.embeddings_btn.clicked.connect(self._import_embeddings)
        self.compound_btn = QtWidgets.QPushButton()
        self.compound_btn.setToolTip(f"Build the compound cache from the database dumps in {compound_db.DB_DIR}")
        self.compound_btn.clicked.connect(self._build_compounds)

        spectral = QtWidgets.QGroupBox("Spectral libraries")
        lay = QtWidgets.QVBoxLayout(spectral)
        lay.addWidget(self.spectral_label)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.spectral_btn)
        row.addWidget(self.embeddings_btn)
        row.addStretch(1)
        lay.addLayout(row)

        compounds = QtWidgets.QGroupBox("Compound databases")
        lay = QtWidgets.QVBoxLayout(compounds)
        lay.addWidget(self.compound_label)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.compound_btn)
        row.addStretch(1)
        lay.addLayout(row)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(spectral)
        layout.addWidget(compounds)
        self.refresh()

    def refresh(self):
        """Re-read both indexes' state."""
        if spectrum_db.cache_ok():
            libraries = ", ".join(spectrum_db.available_libraries()) or "none"
            embedders = ", ".join(spectrum_db.embedders()) or "none"
            imported = ", ".join(QUERY_EMBEDDINGS) or "none"
            text = (f"Libraries: {libraries}<br>Embeddings: {embedders}<br>"
                    f"Query embeddings imported this session: {imported}")
        elif spectrum_db.DEFAULT_DB.exists():
            text = "The index is from an older version: rebuild it"
        else:
            text = f"No index yet: add .mgf files to {spectrum_db.DB_DIR} and build it"
        self.spectral_label.setText(f"{text}<br><small>{spectrum_db.DB_DIR}</small>")
        self.spectral_btn.setText("Rebuild index" if spectrum_db.DEFAULT_DB.exists() else "Build index")
        self.embeddings_btn.setEnabled(spectrum_db.cache_ok())

        if compound_db.cache_ok():
            text = f"Sources: {', '.join(compound_db.available_sources()) or 'none'}"
        elif compound_db.DEFAULT_DB.exists():
            text = "The cache is from an older version: rebuild it"
        else:
            text = f"No cache yet: build it from the dumps in {compound_db.DB_DIR}"
        self.compound_label.setText(f"{text}<br><small>{compound_db.DB_DIR}</small>")
        self.compound_btn.setText("Rebuild cache" if compound_db.DEFAULT_DB.exists() else "Build cache")

    def _rebuild_spectral(self):
        self.spectral_btn.setEnabled(False)
        self.spectral_label.setText("Rebuilding the spectral index in the background…")
        self.sigRebuildSpectral.emit()

    def _build_compounds(self):
        self.compound_btn.setEnabled(False)
        self.compound_label.setText("Building the compound cache in the background…")
        self.sigBuildCompounds.emit()

    def build_done(self, ok: bool):
        """A build MainController ran has finished."""
        for btn in (self.spectral_btn, self.compound_btn):
            btn.setEnabled(True)
        self.refresh()
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Libraries & databases", "The build failed: see the process log")

    def _import_embeddings(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import query embeddings (<name>.<embedder>.npz)", "", "Embeddings (*.npz)",
        )
        if not path:
            return
        try:
            embedder = import_query_embeddings(Path(path))
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Import query embeddings", str(e))
            return
        self.refresh()
        self.sigEmbeddingsImported.emit(embedder)
