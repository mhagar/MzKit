"""
The annotation tabs, for any host with an MS2 plot (Alignment Viewer, Ensemble
Viewer): Annotation (AnnotationEditor), Search (spectra) (SpectralSearchWidget)
and Search (structures) (StructureSearchWidget), added to the host's
QTabWidget.

The host hands in an AnnotationTarget for its focus (`set_target`) and draws
what `sigMirror` asks for on its MS2 plot, feeding hovers back through
`mirror_hovered`. Background work goes out through sigRankRequested /
sigSpectralSearchRequested (MainController runs it) and comes back through
`set_ranking` / `set_spectral_result`. Edits go to the ensemble through
`data_source.update_ensemble`.
"""
import dataclasses
from typing import NamedTuple, Optional, TYPE_CHECKING

from PyQt5 import QtCore, QtWidgets

from core.cli.rank_candidates import ScoredCandidate
from core.data_structs.ensemble import EnsembleAnnotation, Hypothesis
from gui.widgets.annotation_editor import AnnotationEditor
from gui.widgets.annotation_target import AnnotationTarget, ensemble_adduct
from gui.widgets.spectral_search_widget import SpectralSearchWidget
from gui.widgets.structure_search_widget import StructureSearchWidget, scores_of
from gui.widgets.StructureWidget import StructureView

if TYPE_CHECKING:
    from core.interfaces.data_sources import EnsembleDataSource
    from core.utils.array_types import SpectrumArray


class MirrorSpec(NamedTuple):
    """An MS2 mirror for the host to draw: observed (top) against a prediction / library spectrum."""
    top: 'SpectrumArray'
    bottom: 'SpectrumArray'
    top_label: str
    bottom_label: str
    matches: list
    top_precursor: Optional[float]
    bottom_precursor: Optional[float]


def _editor_page() -> tuple[QtWidgets.QWidget, dict]:
    """The Annotation page: target label, name, notes, hypotheses over the structure."""
    page = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(page)
    layout.setSpacing(2)
    layout.setContentsMargins(0, 2, 0, 0)
    label = QtWidgets.QLabel()
    label.setWordWrap(True)
    name, notes = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    name.setPlaceholderText("Name...")
    notes.setPlaceholderText("Notes...")
    splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
    splitter.setChildrenCollapsible(False)
    tree = QtWidgets.QTreeWidget()
    tree.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed)
    tree.setRootIsDecorated(False)
    tree.setUniformRowHeights(True)
    tree.setWordWrap(False)
    tree.setHeaderLabels(["Structure", "Name", "Origin"])
    structure = StructureView()
    splitter.addWidget(tree)
    splitter.addWidget(structure)
    for w in (label, name, notes):
        layout.addWidget(w)
    layout.addWidget(splitter, stretch=1)
    return page, dict(edit_name=name, label_target=label, edit_notes=notes, tree=tree, structure=structure)


class AnnotationTabs(QtCore.QObject):
    sigRankRequested = QtCore.pyqtSignal(object, dict)            # key, rank_candidates kwargs
    sigSpectralSearchRequested = QtCore.pyqtSignal(object, dict)  # slot, search_spectra kwargs
    sigMirror = QtCore.pyqtSignal(object)                         # MirrorSpec, or None: back to normal
    sigStatus = QtCore.pyqtSignal(str)                            # for the host's status bar

    def __init__(self, tab_widget: QtWidgets.QTabWidget, data_source: 'EnsembleDataSource'):
        super().__init__(tab_widget)
        self.tabs, self.data_source = tab_widget, data_source
        self._target: Optional[AnnotationTarget] = None
        # The MS2 mirror on show: a structure's ScoredCandidate, or a LibraryHit (None: not mirroring)
        self._mirrored = None
        # (target key, SMILES) -> a hypothesis's ScoredCandidate, for the session
        self._hypothesis_scores: dict = {}
        # (target key, library, spectrum id) -> a library hypothesis's LibraryHit, for the session
        self._library_hits: dict = {}

        self.editor_page, widgets = _editor_page()
        self.editor = AnnotationEditor(**widgets)
        self.spectral = SpectralSearchWidget()
        self.structure_search = StructureSearchWidget()
        for page, title in ((self.editor_page, "Annotation"), (self.spectral, "Search (spectra)"),
                            (self.structure_search, "Search (structures)")):
            tab_widget.addTab(page, title)

        self.editor.sigAnnotationChanged.connect(
            lambda uuid, ann: self.data_source.update_ensemble(uuid, annotation=ann)
        )
        self.editor.sigHypothesisSelected.connect(
            lambda h: self._front() is self.editor_page and self._on_hypothesis_selected(h)
        )
        self.editor.sigApplyToAllRequested.connect(self._apply_to_all)
        self.structure_search.sigRankRequested.connect(self.sigRankRequested)
        self.structure_search.sigCandidateSelected.connect(
            lambda s: self._front() is self.structure_search and self._show_candidate(s)
        )
        self.structure_search.sigAccepted.connect(self._on_accepted)
        self.spectral.sigSearchRequested.connect(self.sigSpectralSearchRequested)
        self.spectral.sigHitSelected.connect(
            lambda h: self._front() is self.spectral and self._show_hit(h)
        )
        self.spectral.sigAccepted.connect(self._on_accepted)
        tab_widget.currentChanged.connect(self._on_tab_changed)

    def _front(self):
        return self.tabs.currentWidget()

    def _on_tab_changed(self):
        page = self._front()
        if page is self.editor_page:
            self.editor.reselect()
        elif page in (self.spectral, self.structure_search):
            page.reselect()
        else:   # a host page came to the front
            self._show_candidate(None)

    # -- in -----------------------------------------------------------------

    def set_target(self, target: Optional[AnnotationTarget]):
        """Follow the host's focus (None clears)."""
        self._target = target
        self.structure_search.set_target(target)
        self.spectral.set_target(target)
        self.editor.set_target(target)

    def set_ranking(self, key, result):
        """A finished candidate ranking (None: failed / cancelled), from MainController."""
        if not (isinstance(key, tuple) and key and key[0] == 'hypothesis'):
            self.structure_search.set_ranking(key, result)
            return
        _, target_key, smiles, ensemble_uuid = key
        if result is None or not result.scored:
            return
        scored = result.scored[0]
        if target_key is not None:
            self._hypothesis_scores[(target_key, smiles)] = scored
        # Straight onto the ensemble scored, wherever the focus has moved since
        ensemble = self.data_source.get_ensemble(ensemble_uuid)
        if ensemble is not None and ensemble.annotation is not None:
            self.data_source.update_ensemble(ensemble_uuid, annotation=ensemble.annotation.with_scores(
                smiles, scores_of(scored, result),
            ))
        selected = self.editor.selected()
        if (self._front() is self.editor_page and selected is not None and selected.smiles == smiles
                and self._target is not None and self._target.key == target_key):
            self._show_candidate(scored)

    def set_spectral_result(self, slot, result):
        """A finished spectral-library search (None: failed / cancelled), from MainController."""
        self.spectral.set_result(slot, result)

    def refresh_index(self):
        """After a spectral index rebuild."""
        self.spectral.refresh_index()

    def refresh_sources(self):
        """After a compound database cache rebuild."""
        self.structure_search.refresh_sources()

    def refresh_modes(self):
        """After query embeddings were imported."""
        self.spectral.refresh_modes()

    def mirror_hovered(self, peak_idx: Optional[int]):
        """A predicted peak of the mirror hovered (None: left): light up its fragment."""
        if not isinstance(self._mirrored, ScoredCandidate):
            return
        if self._front() is self.structure_search:
            self.structure_search.highlight_fragment(peak_idx)
        else:
            self.editor.highlight_fragment(self._mirrored, peak_idx)

    # -- accept / hypotheses ------------------------------------------------

    def _on_accepted(self, hypothesis: Hypothesis):
        """A candidate / library hit becomes the target's chosen hypothesis, shown in its tab."""
        if self._front() is self.structure_search and (shown := self.structure_search.shown) is not None:
            self._hypothesis_scores[(self._target.key, hypothesis.smiles)] = shown
        if self._front() is self.spectral and (hit := self.spectral.shown) is not None:
            self._library_hits[(self._target.key, hit.library, hit.spectrum_id)] = hit
        self.editor.add_hypothesis(hypothesis)
        self.tabs.setCurrentWidget(self.editor_page)

    def _on_hypothesis_selected(self, hypothesis: Optional[Hypothesis]):
        """
        Mirror a hypothesis: a library hit's library spectrum, else its GLACIER
        prediction for the target's MS2 (scored once per session).
        """
        if hypothesis is None or self._target is None:
            self._show_candidate(None)
            return
        key = self._target.key
        if hypothesis.origin in {c.text() for c in self.spectral.library_checks}:
            self._show_library_hypothesis(hypothesis)
            return
        if (scored := self._hypothesis_scores.get((key, hypothesis.smiles))) is not None:
            self._show_candidate(scored)
            return
        self._show_candidate(None)
        kwargs = self.structure_search.score_kwargs(hypothesis.smiles, hypothesis.name)
        if kwargs is not None:
            self.sigRankRequested.emit(('hypothesis', key, hypothesis.smiles, self.editor.target_uuid), kwargs)

    def _apply_to_all(self, hypothesis: Hypothesis):
        """
        Every member of the target gets `hypothesis` as its chosen structure,
        scored against its own MS2 (never another member's scores).
        """
        members = [] if self._target is None else self._target.members
        if not members:
            return
        n_other = sum(
            1 for e in members
            if e.annotation is not None and e.annotation.chosen is not None
            and e.annotation.smiles != hypothesis.smiles
        )
        if QtWidgets.QMessageBox.question(
            self.tabs, "Apply structure to all aligned ensembles",
            f"Choose {hypothesis.name or hypothesis.smiles} as the structure of "
            f"{len(members)} member{'' if len(members) == 1 else 's'}; "
            f"{n_other} of them have a different structure chosen.",
        ) != QtWidgets.QMessageBox.Yes:
            return

        copy = dataclasses.replace(hypothesis, scores={})
        n_unscored = 0
        for ensemble in members:
            ann = (ensemble.annotation or EnsembleAnnotation()).with_hypothesis(copy)
            self.data_source.update_ensemble(ensemble.uuid, annotation=ann)
            if ann.chosen.scores:   # already scored there
                continue
            composite = ensemble.composite_spectrum
            own = AnnotationTarget(key=None, ensemble=ensemble, ms2=composite.ms2,
                                   precursor_mz=composite.precursor_mz, adduct=ensemble_adduct(ensemble))
            kwargs = self.structure_search.score_kwargs(hypothesis.smiles, hypothesis.name, own)
            if kwargs is None:
                n_unscored += 1
                continue
            self.sigRankRequested.emit(('hypothesis', None, hypothesis.smiles, ensemble.uuid), kwargs)
        self.sigStatus.emit(
            f"Structure applied to {len(members)} member(s)"
            + (f"; {n_unscored} can't be scored (no MS2 or GLACIER model)" if n_unscored else "")
        )

    def _show_library_hypothesis(self, hypothesis: Hypothesis):
        """Mirror the library spectrum a hypothesis was accepted from (re-read from the index if need be)."""
        from core.cli.search_spectra import library_hit, params_from_config
        from core.utils.config import load_config

        target, slot = self._target, (self._target.key, hypothesis.origin, hypothesis.db_id)
        hit = self._library_hits.get(slot)
        if hit is None and target.ms2 is not None and target.ms2.size and target.precursor_mz is not None:
            hit = library_hit(target.ms2, target.precursor_mz, hypothesis.origin, hypothesis.db_id,
                              params_from_config(load_config()))
            if hit is not None:
                self._library_hits[slot] = hit
        if hit is None:
            self._show_candidate(None)
            self.sigStatus.emit(f"{hypothesis.origin}:{hypothesis.db_id} isn't in the spectral index")
            return
        self._show_hit(hit)

    # -- mirror -------------------------------------------------------------

    def _show_candidate(self, scored: Optional[ScoredCandidate]):
        """Mirror the target's MS2 against `scored`'s prediction (None: back to normal)."""
        if scored is None:
            if self._mirrored is not None:
                self._mirrored = None
                self.sigMirror.emit(None)
            return
        search = self.structure_search
        self._mirrored = scored
        self.sigMirror.emit(MirrorSpec(
            search.ms2, scored.prediction, "Observed", "Predicted (GLACIER)", scored.matches,
            search.precursor_mz, search.precursor_mz,
        ))

    def _show_hit(self, hit):
        """Mirror the target's MS2 against library hit `hit`'s spectrum (None: back to normal)."""
        if hit is None:
            self._show_candidate(None)
            return
        self._mirrored = hit
        self.sigMirror.emit(MirrorSpec(
            self.spectral.ms2, hit.peaks, "Observed",
            f"Library ({hit.library}: {hit.name or hit.spectrum_id})", hit.matches,
            self.spectral.precursor_mz, hit.precursor_mz,
        ))

    def forget_mirror(self):
        """The host redrew its MS2 plot itself: nothing is mirrored any more."""
        self._mirrored = None

    @property
    def mirrored(self):
        """The ScoredCandidate / LibraryHit being mirrored (None: not mirroring)."""
        return self._mirrored
