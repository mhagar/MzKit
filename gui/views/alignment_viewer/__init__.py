"""
Alignment Viewer — interactive feature-map for an ``EnsembleAlignment``.

Each Ensemble is a strip laid out by retention time within a per-sample
lane; strips of the same AlignedAnalyte share a colour and a connecting
line (see gui/widgets/AlignmentPlotWidget.py).

Cluster mode instead lays analytes out in the leaf order of a hierarchical
clustering by a registered SimilarityMatrix - computed (MS2 modified cosine /
MS2DeepScore, core/cli/cluster_analytes.py) or imported
(core/cli/import_similarity.py) - with the dendrogram drawn above the map
(gui/widgets/AlignmentDendrogramWidget.py). Matrix computation and clustering
run in the background via MainController (sigClusterRequested).

User AnalyteGroups (core/data_structs/analyte_group.py) are drawn on the
group track above the map in either layout (gui/widgets/AlignmentGroupTrackWidget.py)
and edited from the Inspector's groups panel (groups_panel.py, group_dialog.py).

The selection on the map drives:
 - the Inspector side panel (properties / members / comparison table),
 - the bottom panel (MS1, MS2, chromatogram; see detail_panel.py),
 - the status bar (hover readout).

Actions into the Ensemble / Sample viewers go out via signals the
MainController wires up.
"""
from typing import Optional, TYPE_CHECKING

from PyQt5 import QtWidgets, QtCore, QtGui

from core.cli.cluster_analytes import (
    ClusterKey, ClusterParams, cluster_params_from_config, linkage_from_config,
    scoring_name, similarity_provenance,
)
from core.data_structs.similarity_matrix import ImportParams, SimilarityMatrix
from core.utils.config import load_config
from gui.resources.AlignmentViewerWindow import Ui_Form
from gui.views.alignment_viewer.cluster_params_dialog import ClusterParamsDialog
from gui.views.alignment_viewer.context import AlignmentContext, ItemSpectra
from gui.views.alignment_viewer.detail_panel import DetailPanel, XIC_COLOR
from gui.views.alignment_viewer.group_dialog import AnalyteGroupDialog, jsonable
from gui.views.alignment_viewer.groups_panel import (
    GroupsPanel, selection_analyte_uuids,
)
from gui.views.alignment_viewer.inspector import Inspector
from gui.views.alignment_viewer.render_params_dialog import RenderParamsDialog
from gui.widgets.alignment_plot.layout import (
    ANCHOR_COLOR, CURRENT_COLOR, HoverTarget,
)
from gui.widgets.alignment_plot.params import RenderParams

if TYPE_CHECKING:
    from core.cli.align_ensembles import PairScore
    from core.cli.cluster_analytes import AnalyteClustering
    from core.data_structs import Ensemble, SampleUUID
    from core.data_structs.alignment import EnsembleAlignment
    from gui.views.alignment_viewer.data_source import AlignmentViewerDataSource


class AlignmentViewer(
    QtWidgets.QWidget,
    Ui_Form,
):
    """
    Feature-map viewer for a single EnsembleAlignment.
    """

    # Actions requested from the plot, wired up by MainController.
    sigViewEnsembleRequested = QtCore.pyqtSignal(object)  # Ensemble
    sigAddSamplesRequested = QtCore.pyqtSignal(object)     # list[SampleUUID]
    sigAutoFindMfsRequested = QtCore.pyqtSignal(object)    # list[Ensemble]
    # EnsembleAlignment, ClusterParams (compute) | SimilarityMatrix, linkage
    sigClusterRequested = QtCore.pyqtSignal(object, object, str)

    def __init__(
        self,
        data_source: 'AlignmentViewerDataSource',
        parent=None,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self.data_source = data_source

        self._params = RenderParams()
        self._alignment: Optional['EnsembleAlignment'] = None
        self._ctx: Optional[AlignmentContext] = None
        self._selection: list[HoverTarget] = []

        # Compare mode: the selected items, their scores vs the anchor, and
        # which one is on the mirror plots
        self._compare_items: list[ItemSpectra] = []
        self._compare_targets: list[HoverTarget] = []   # parallel to items
        self._compare_scores: list[Optional['PairScore']] = []
        self._compare_idx: int = 1

        self._params_dialog: Optional[RenderParamsDialog] = None

        # Cluster mode: what to cluster by - ClusterParams (a registered
        # matrix computed that way, else compute one) or a matrix uuid - and
        # the results for the current alignment, keyed by ClusterKey
        config = load_config()
        self._cluster_source: 'ClusterParams | int' = (
            cluster_params_from_config(config)
        )
        self._cluster_linkage: str = linkage_from_config(config)
        self._clusterings: dict[ClusterKey, 'AnalyteClustering'] = {}
        self._shown_clustering: Optional['AnalyteClustering'] = None
        self._cluster_dialog: Optional[ClusterParamsDialog] = None

        self.plot_item = self.plotAlignment.pi
        self.detail_panel = DetailPanel(
            ms1_plot=self.plotMS1,
            ms2_plot=self.plotMS2,
            chrom_plot=self.plotChrom,
            map_plot=self.plotAlignment,
        )
        self.inspector = Inspector(
            title_label=self.labelInspectorTitle,
            stack=self.stackedWidget,
            summary_label=self.labelAlignmentSummary,
            table_ensemble=self.tableEnsembleProps,
            table_analyte_props=self.tableAnalyteProps,
            table_analyte_members=self.tableAnalyteMembers,
            table_compare=self.tableCompare,
        )
        self.groups_panel = GroupsPanel(
            self.data_source,
            table=self.tableGroups,
            svg=self.svgGroupStructure,
            btn_edit=self.btnGroupEdit,
            btn_delete=self.btnGroupDelete,
            btn_add=self.btnGroupAddSelected,
            btn_remove=self.btnGroupRemoveSelected,
        )
        # Analyte -> x of the current layout, shared with the group track
        self._x_of_analyte: dict = {}

        self._setup_layout()
        self._setup_toolbar()
        self._add_status_bar()
        self._connect_signals()

        self.detail_panel.set_rt_linked(self.btnLinkRT.isChecked())
        self.inspector.show_summary(None)

    # -- setup --------------------------------------------------------------

    def _setup_layout(self):
        self.verticalLayout_4.setContentsMargins(4, 4, 4, 0)
        self.splitter_2.setStretchFactor(1, 1)
        self.splitter_2.setSizes([260, 740])
        self.splitter.setSizes([550, 450])
        self.plotDendrogram.link_to(self.plot_item)
        self.plotDendrogram.hide()
        self.plotGroupTrack.link_to(self.plot_item)
        self.plotGroupTrack.hide()
        self.plotMS1.pi.hideAxis('bottom')  # m/z-linked to the MS2 plot below

    def _setup_toolbar(self):
        self.btnLinkRT.setText("Link RT")
        self.btnLinkRT.setIcon(
            QtGui.QIcon("gui/resources/icons/phosphor_icons/link-simple-fill.svg")
        )
        self.btnLinkRT.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btnLinkRT.setToolTip(
            "Link the chromatogram's RT axis to the alignment map's"
        )
        self.btnRenderParams.setText("Display…")
        self.btnRenderParams.setToolTip("Edit strip / line display settings")
        self.btnClusterMode.setCheckable(True)
        self.btnClusterMode.setText("Cluster (MS2)")
        self.btnClusterMode.setToolTip(
            "Order analytes by hierarchical clustering of their MS2 "
            "(modified cosine) instead of by retention time"
        )
        self.btnClusterParams.setText("Clustering…")
        self.btnClusterParams.setToolTip("Edit clustering parameters")
        self.btnGroupSelected.setText("Group selected…")
        self.btnGroupSelected.setToolTip(
            "Name / annotate the selected analytes as a group"
        )
        self.btnGroupSelected.setEnabled(False)

    def _add_status_bar(self):
        """
        Hover readout. Its QSizeGrip also gives this MDI subwindow a
        bottom-right resize handle (mirrors EnsembleViewer).
        """
        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.setMaximumHeight(18)  # pixels
        self.verticalLayout_4.addWidget(self.status_bar)

    def _connect_signals(self):
        self.plot_item.sigSelectionChanged.connect(self._on_selection_changed)
        self.plot_item.sigHovered.connect(self._on_hovered)
        self.plot_item.sigEnsembleActivated.connect(
            self.sigViewEnsembleRequested
        )
        self.plot_item.sigContextRequested.connect(self._show_context_menu)
        self.plotChrom.sigEnsemblePeakClicked.connect(self._on_xic_clicked)
        self.plotChrom.sigEnsemblePeakDoubleClicked.connect(
            self._on_xic_double_clicked
        )

        self.inspector.sigCompareRowSelected.connect(self._on_compare_row)
        self.inspector.sigAnchorRequested.connect(self._on_anchor_requested)
        self.inspector.sigMemberActivated.connect(
            self.sigViewEnsembleRequested
        )

        self.btnLinkRT.toggled.connect(self._update_rt_link)
        self.btnRenderParams.clicked.connect(self._show_params_dialog)
        self.btnClusterMode.toggled.connect(self._on_cluster_mode_toggled)
        self.btnClusterParams.clicked.connect(self._show_cluster_dialog)
        self.plotDendrogram.sigSubtreeClicked.connect(self._select_analytes)
        self.plotDendrogram.sigNodeHovered.connect(self._on_node_hovered)
        self.btnGroupSelected.clicked.connect(lambda: self._group_analytes())
        self.plotGroupTrack.sigGroupHovered.connect(
            self.plot_item.set_hover_analytes
        )
        self.plotGroupTrack.sigGroupClicked.connect(self._select_analytes)

        # Formula assignments landing (from EnsembleViewer, auto find-mfs,
        # ...) change what the Inspector shows
        self.data_source.subscribe_to_changes(
            addition_callback=self._on_assignment_changed,
            removal_callback=self._on_assignment_changed,
            update_callback=self._on_assignment_changed,
            change_type='Assignment',
        )
        self.data_source.subscribe_to_changes(
            addition_callback=lambda _: None,
            removal_callback=self._on_alignment_removed,
            change_type='Alignment',
        )
        self.data_source.subscribe_to_changes(
            addition_callback=self._on_groups_changed,
            removal_callback=self._on_groups_changed,
            update_callback=self._on_groups_changed,
            change_type='Group',
        )
        self.data_source.subscribe_to_changes(
            addition_callback=lambda _: None,
            removal_callback=self._on_matrix_removed,
            change_type='Matrix',
        )

    # -- public API (kept stable for MainController / SubWindowManager) ------

    def set_alignment(
        self,
        alignment: 'EnsembleAlignment',
    ):
        self._alignment = alignment
        self._ctx = AlignmentContext(alignment, self.data_source)
        self._selection = []
        self._clear_cluster_mode()
        if (
            not isinstance(self._cluster_source, ClusterParams)
            and self._cluster_matrix() is None
        ):
            # A matrix sharing no analytes with this alignment: back to config
            self._cluster_source = cluster_params_from_config(load_config())
        if self._cluster_dialog is not None:
            self._cluster_dialog.set_alignment(alignment)
        self.detail_panel.reset()
        self._render()
        self.inspector.show_summary(alignment)
        self.status_bar.showMessage(self._summary_text())

    def select_ensemble(
        self,
        sample_uuid: 'SampleUUID',
        ensemble_uuid: 'EnsembleUUID',
        as_analyte: bool = False,
    ) -> bool:
        """
        Select an ensemble (or the analyte it belongs to) on the map and pan
        to it. Prefers the shown alignment; otherwise switches to the most
        recent alignment containing the ensemble. False if none does.
        """
        candidates = [self._alignment] if self._alignment else []
        candidates += [
            self.data_source.get_alignment(u)
            for u in reversed(self.data_source.get_all_alignment_uuids())
        ]
        for alignment in candidates:
            if alignment is None:
                continue
            found = _locate_ensemble(alignment, ensemble_uuid)
            if found is None or found[1] != sample_uuid:
                continue
            aid = found[0]
            if alignment is not self._alignment:
                self.set_alignment(alignment)
            if as_analyte:
                return self.plot_item.reveal('analyte', aid)
            return self.plot_item.reveal('ensemble', aid, sample_uuid)
        return False

    def imported_similarities(self) -> list[tuple[SimilarityMatrix, ImportParams]]:
        """
        The registered imported matrices sharing analytes with the shown
        alignment, each with the transform it was last clustered with.
        """
        if self._alignment is None:
            return []
        return [
            (m, m.default_transform)
            for m in self.data_source.matrices_for_analytes(
                a.uuid for a in self._alignment.analytes
            )
            if m.is_imported
        ]

    def reset_for_new_project(self):
        self._alignment = None
        self._ctx = None
        self._selection = []
        self._clear_cluster_mode()
        if self._cluster_dialog is not None:
            self._cluster_dialog.set_alignment(None)
        self.plot_item.clear_alignment()
        self._x_of_analyte = {}
        self._refresh_groups()
        self.detail_panel.reset()
        self.inspector.show_summary(None)
        self.status_bar.clearMessage()

    # -- rendering ----------------------------------------------------------

    def _render(
        self,
        preserve_range: bool = False,
        preserve_selection: bool = False,
    ):
        """
        (Re)draw the current alignment with the current RenderParams, laid
        out by RT or - in cluster mode, once a clustering is available - by
        leaf order.
        """
        if self._alignment is None:
            return

        clustering = self._active_clustering()
        if clustering is not self._shown_clustering:
            if clustering is None:
                self.plotDendrogram.clear_clustering()
            else:
                self.plotDendrogram.set_clustering(clustering)
            self._shown_clustering = clustering
        self.plotDendrogram.setVisible(clustering is not None)

        if clustering is None:
            self._x_of_analyte = {
                a.uuid: a.consensus_rt for a in self._alignment.analytes
            }
            self.plot_item.set_alignment(
                self._alignment, self.data_source, self._params,
                preserve_range=preserve_range,
                preserve_selection=preserve_selection,
            )
        else:
            slot_of = {u: i for i, u in enumerate(clustering.analyte_uuids)}
            self._x_of_analyte = slot_of
            self.plot_item.set_alignment(
                self._alignment, self.data_source, self._params,
                preserve_range=preserve_range,
                preserve_selection=preserve_selection,
                x_of=lambda aid, analyte, ens: slot_of.get(analyte.uuid),
                width_scale=self._params.cluster_width_scale,
                x_axis_visible=False,
                stagger=False,   # one analyte per column: nothing to overlap
            )
        self._refresh_groups()
        self._update_rt_link()
        self.status_bar.showMessage(self._summary_text())

    def _show_params_dialog(self):
        if self._params_dialog is None:
            self._params_dialog = RenderParamsDialog(self._params, parent=self)
            self._params_dialog.sigParamsChanged.connect(self._on_params_changed)
        self._params_dialog.show()
        self._params_dialog.raise_()

    def _on_params_changed(self, params: RenderParams):
        self._params = params
        self._render(preserve_range=True, preserve_selection=True)

    def _update_rt_link(self):
        """
        Link the chromatogram's RT axis to the map's - only meaningful when
        the map's x axis *is* RT, so cluster mode always unlinks it.
        """
        rt_layout = self._active_clustering() is None
        self.btnLinkRT.setEnabled(rt_layout)
        linked = rt_layout and self.btnLinkRT.isChecked()
        if linked != self.detail_panel.rt_linked:
            self.detail_panel.set_rt_linked(linked)

    # -- cluster mode -------------------------------------------------------

    def _cluster_matrix(self) -> Optional[SimilarityMatrix]:
        """
        The registered matrix the current source resolves to for this
        alignment (None: not computed yet, or not covering it).
        """
        if self._alignment is None:
            return None
        source = self._cluster_source
        if isinstance(source, ClusterParams):
            return self.data_source.find_matrix(
                similarity_provenance(source), self._alignment,
            )
        matrix = self.data_source.get_matrix(source)
        if matrix is None or matrix not in self.data_source.matrices_for_analytes(
            a.uuid for a in self._alignment.analytes
        ):
            return None
        return matrix

    def _cluster_key(self) -> Optional[ClusterKey]:
        matrix = self._cluster_matrix()
        if matrix is None:
            return None
        return ClusterKey(
            matrix.uuid, matrix.default_transform, self._cluster_linkage,
        )

    def _current_clustering(self) -> Optional['AnalyteClustering']:
        """The cached clustering for the current settings, if any."""
        key = self._cluster_key()
        return None if key is None else self._clusterings.get(key)

    def _active_clustering(self) -> Optional['AnalyteClustering']:
        """The clustering to lay out by, or None for RT layout."""
        if not self.btnClusterMode.isChecked():
            return None
        return self._current_clustering()

    def _clear_cluster_mode(self):
        """Drop cached clusterings and fall back to RT mode (no re-render)."""
        self._clusterings.clear()
        with QtCore.QSignalBlocker(self.btnClusterMode):
            self.btnClusterMode.setChecked(False)
        self._update_rt_link()

    def _on_cluster_mode_toggled(self, checked: bool):
        if self._alignment is None:
            with QtCore.QSignalBlocker(self.btnClusterMode):
                self.btnClusterMode.setChecked(False)
            return
        if checked and self._current_clustering() is None:
            self._request_clustering()
            return
        # Units change between modes: keep the selection, reset the range
        self._render(preserve_selection=True)

    def _request_clustering(self):
        """Ask MainController to cluster; the map stays as is meanwhile."""
        matrix = self._cluster_matrix()
        if matrix is None and not isinstance(self._cluster_source, ClusterParams):
            # The chosen matrix is gone (or doesn't cover this alignment)
            self._cluster_source = cluster_params_from_config(load_config())
        self.status_bar.showMessage(
            f"Clustering analytes by {self._source_name(matrix)}…"
        )
        self.sigClusterRequested.emit(
            self._alignment, matrix or self._cluster_source,
            self._cluster_linkage,
        )

    def _source_name(self, matrix: Optional[SimilarityMatrix]) -> str:
        if matrix is not None:
            return matrix.name
        if isinstance(self._cluster_source, ClusterParams):
            return scoring_name(self._cluster_source.scoring)
        return "similarity"

    def set_clustering(
        self,
        alignment_uuid: int,
        clustering: Optional['AnalyteClustering'],
    ):
        """
        Clustering results landing (from MainController). Ignored if the
        alignment has changed since; None (cancelled) leaves cluster mode.
        """
        if self._alignment is None or alignment_uuid != self._alignment.uuid:
            return
        if clustering is None:
            with QtCore.QSignalBlocker(self.btnClusterMode):
                self.btnClusterMode.setChecked(False)
            self.status_bar.showMessage(self._summary_text())
            return
        self._clusterings[clustering.params] = clustering
        if self._active_clustering() is clustering:
            self._render(preserve_selection=True)
        self._refresh_compare()

    def _show_cluster_dialog(self):
        if self._cluster_dialog is None:
            config = load_config()
            self._cluster_dialog = ClusterParamsDialog(
                self.data_source,
                cluster_params_from_config(config),
                linkage_from_config(config),
                parent=self,
            )
            self._cluster_dialog.sigApply.connect(self._on_cluster_params)
            self._cluster_dialog.set_alignment(self._alignment)
        self._cluster_dialog.show()
        self._cluster_dialog.raise_()

    def _on_cluster_params(
        self,
        source: 'ClusterParams | SimilarityMatrix',
        linkage_method: str,
    ):
        self._cluster_source = (
            source.uuid if isinstance(source, SimilarityMatrix) else source
        )
        self._cluster_linkage = linkage_method
        if self._alignment is None:
            return
        if not self.btnClusterMode.isChecked():
            self.btnClusterMode.setChecked(True)   # requests / renders
        elif self._current_clustering() is not None:
            self._render(preserve_selection=True)
            self._refresh_compare()
        else:
            self._request_clustering()

    def _select_analytes(self, analyte_uuids: list, additive: bool):
        """Select these analytes (a dendrogram subtree / group-track click)."""
        targets = self.plot_item.analyte_targets(analyte_uuids)
        if additive:
            keys = {t.analyte_id for t in targets}
            targets = [
                t for t in self.plot_item.selection
                if not (t.kind == 'analyte' and t.analyte_id in keys)
            ] + targets
        self.plot_item.set_selection(targets)

    def _on_node_hovered(self, info: Optional[tuple[float, int]]):
        if info is None:
            self.status_bar.showMessage(self._summary_text())
            return
        similarity, n = info
        self.status_bar.showMessage(
            f"Merge at similarity {similarity:.2f}  ·  {n} analytes  "
            f"(click to select, Ctrl+click to add)"
        )

    # -- selection ----------------------------------------------------------

    def _on_selection_changed(self, targets: list[HoverTarget]):
        self._selection = targets
        self._compare_idx = 1
        self._show_selection()
        self._refresh_groups_panel()

    def _show_selection(self):
        ctx, targets = self._ctx, self._selection
        # Compare mode only while a comparison is shown (see below)
        self._compare_items, self._compare_targets = [], []
        self.plot_item.set_compare_highlight(None, None)
        if ctx is None or not targets:
            self.inspector.show_summary(self._alignment)
            self.detail_panel.clear()
            return

        if len(targets) == 1:
            self._show_single(ctx, targets[0])
            return

        items: list[ItemSpectra] = []
        item_targets: list[HoverTarget] = []
        for target in targets:
            item = ctx.item_spectra(target)
            if item is not None:
                items.append(item)
                item_targets.append(target)
        if len(items) < 2:
            self._show_single(ctx, targets[0])
            return

        self._compare_items = items
        self._compare_targets = item_targets
        self._compare_scores = [None] + [
            ctx.score(items[0], item) for item in items[1:]
        ]
        self._compare_idx = min(self._compare_idx, len(items) - 1)
        self.inspector.show_compare(
            items, self._anchor_similarities(items), self._compare_idx,
        )
        self._show_compare_plots()

    def _refresh_compare(self):
        """Re-show a multi-selection (e.g. so Similarity picks up a new clustering)."""
        if len(self._selection) > 1:
            self._show_selection()

    def _show_single(self, ctx: AlignmentContext, target: HoverTarget):
        if target.kind == 'ensemble':
            self.inspector.show_ensemble(
                ctx, target.sample_uuid, target.ensemble, target.analyte,
            )
            self.detail_panel.show_ensemble(
                ctx, target.sample_uuid, target.ensemble,
            )
        else:
            self.inspector.show_analyte(ctx, target.analyte)
            self.detail_panel.show_analyte(ctx, target.analyte)

    def _anchor_similarities(
        self,
        items: list[ItemSpectra],
    ) -> list[Optional[float]]:
        """
        Each item's similarity to the anchor, from the cached clustering for
        the current cluster settings (modified cosine or imported matrix).
        None where unavailable: no clustering yet, or an item that isn't an
        analyte in it.
        """
        clustering = self._current_clustering()
        if clustering is None:
            return [None] * len(items)
        slot = {u: i for i, u in enumerate(clustering.analyte_uuids)}
        anchor = slot.get(items[0].analyte_uuid)
        if anchor is None:
            return [None] * len(items)
        return [
            None if (i := slot.get(item.analyte_uuid)) is None
            else float(clustering.similarity[anchor, i])
            for item in items
        ]

    def _on_anchor_requested(self, row: int):
        if row <= 0 or row >= len(self._compare_targets):
            return
        target = self._compare_targets[row]
        self._selection = [target] + [t for t in self._selection if t is not target]
        self._compare_idx = 1
        self._show_selection()

    def _on_compare_row(self, row: int):
        if row == self._compare_idx or row >= len(self._compare_items):
            return
        self._compare_idx = row
        self._show_compare_plots()

    def _show_compare_plots(self):
        idx = self._compare_idx
        anchor, current = self._compare_targets[0], self._compare_targets[idx]
        self.plot_item.set_compare_highlight(anchor, current)

        sample_uuids, ensembles, colors = [], [], []
        for target in self._selection:
            color = (
                ANCHOR_COLOR if target is anchor
                else CURRENT_COLOR if target is current
                else XIC_COLOR
            )
            for sample_uuid, ens in self._target_members(target):
                sample_uuids.append(sample_uuid)
                ensembles.append(ens)
                colors.append(color)

        self.detail_panel.show_compare(
            self._ctx,
            self._compare_items[0],
            self._compare_items[idx],
            self._compare_scores[idx],
            sample_uuids,
            ensembles,
            colors,
        )

    def _target_members(
        self,
        target: HoverTarget,
    ) -> list[tuple['SampleUUID', 'Ensemble']]:
        """The ensemble(s) behind a target (an analyte expands to its members)."""
        if target.kind == 'ensemble' and target.ensemble is not None:
            return [(target.sample_uuid, target.ensemble)]
        if target.kind == 'analyte' and target.analyte is not None:
            return list(self._ctx.members(target.analyte).items())
        return []

    # -- chromatogram clicks ------------------------------------------------

    def _compare_row_of(self, ensemble_uuid) -> Optional[int]:
        """The compare row whose item contains this ensemble."""
        for row, target in enumerate(self._compare_targets):
            if any(e.uuid == ensemble_uuid for _, e in self._target_members(target)):
                return row
        return None

    def _xic_target(self, ensemble_uuid) -> Optional[HoverTarget]:
        if self._alignment is None:
            return None
        found = _locate_ensemble(self._alignment, ensemble_uuid)
        return None if found is None else self.plot_item.ensemble_target(*found)

    def _on_xic_clicked(self, ensemble_uuid, button: int):
        """Compare mode: like clicking its compare row. Otherwise: like the map."""
        if button != QtCore.Qt.LeftButton:
            return
        if self._compare_targets:
            row = self._compare_row_of(ensemble_uuid)
            if row:  # not the anchor
                self.inspector.set_current_compare_row(row)
                self._on_compare_row(row)
            return
        target = self._xic_target(ensemble_uuid)
        if target is not None:
            additive = bool(
                QtWidgets.QApplication.keyboardModifiers()
                & QtCore.Qt.ControlModifier
            )
            self.plot_item.select(target, additive)

    def _on_xic_double_clicked(self, ensemble_uuid):
        """Compare mode: make its item the anchor. Otherwise: open it."""
        if self._compare_targets:
            row = self._compare_row_of(ensemble_uuid)
            if row:
                self._on_anchor_requested(row)
            return
        target = self._xic_target(ensemble_uuid)
        if target is not None and target.ensemble is not None:
            self.sigViewEnsembleRequested.emit(target.ensemble)

    # -- hover --------------------------------------------------------------

    def _on_hovered(self, target: Optional[HoverTarget]):
        ctx = self._ctx
        if ctx is None or target is None:
            self.status_bar.showMessage(self._summary_text())
            return

        if target.kind == 'ensemble' and target.ensemble is not None:
            ens = target.ensemble
            parts = [
                ctx.sample_name(target.sample_uuid),
                f"base m/z {ens.base_mz:.4f}",
                f"intensity {ens.base_intsy:.2e}",
                f"RT {ens.peak_rt:.2f} s",
            ]
            formula = ctx.formula(ens)
            if formula:
                parts.append(formula)

        elif target.kind == 'analyte' and target.analyte is not None:
            analyte = target.analyte
            rep = ctx.representative(analyte)
            parts = [
                "Analyte",
                f"consensus RT {analyte.consensus_rt:.2f} s",
                "representative base m/z "
                + (f"{rep.base_mz:.4f}" if rep else "n/a"),
                f"{len(analyte.ensemble_map)}/{ctx.alignment.sample_count} samples",
            ]
            if label := ctx.formula_label(analyte):
                parts.append(label)

        elif target.kind == 'sample':
            parts = [ctx.sample_name(target.sample_uuid)]
        else:
            parts = []

        self.status_bar.showMessage("  ·  ".join(parts))

    def _summary_text(self) -> str:
        alignment = self._alignment
        if alignment is None:
            return ""
        multi = sum(1 for a in alignment.analytes if len(a.ensemble_map) > 1)
        text = (
            f"{alignment.analyte_count} analytes across "
            f"{alignment.sample_count} samples "
            f"({multi} matched, "
            f"{alignment.analyte_count - multi} singletons)"
        )
        clustering = self._active_clustering()
        if clustering is not None:
            text += f"  ·  clustered by {self._source_name(self._cluster_matrix())}"
            if clustering.n_dropped:
                n = clustering.n_dropped
                text += (
                    f"  ·  {n} analyte{'' if n == 1 else 's'} "
                    f"{clustering.dropped_reason} not clustered"
                )
        return text

    # -- registry changes ---------------------------------------------------

    def _on_assignment_changed(self, _assignment):
        # Only the Inspector shows formulae; leave the plots (and their
        # zoom) alone.
        if self._ctx is None or len(self._selection) != 1:
            return
        target = self._selection[0]
        if target.kind == 'ensemble':
            self.inspector.show_ensemble(
                self._ctx, target.sample_uuid, target.ensemble, target.analyte,
            )
        else:
            self.inspector.show_analyte(self._ctx, target.analyte)

    def _on_alignment_removed(self, alignment: 'EnsembleAlignment'):
        if self._alignment is not None and alignment.uuid == self._alignment.uuid:
            self.reset_for_new_project()

    def _on_matrix_removed(self, matrix: SimilarityMatrix):
        """Drop its clusterings; if it was laying out the map, back to RT."""
        for key in [k for k in self._clusterings if k.matrix_uuid == matrix.uuid]:
            del self._clusterings[key]
        if self._cluster_source == matrix.uuid:
            self._cluster_source = cluster_params_from_config(load_config())
        shown = self._shown_clustering
        if shown is not None and shown.params.matrix_uuid == matrix.uuid:
            with QtCore.QSignalBlocker(self.btnClusterMode):
                self.btnClusterMode.setChecked(False)
            self._render(preserve_selection=True)
            self._refresh_compare()

    # -- analyte groups -----------------------------------------------------

    def _on_groups_changed(self, _group):
        self._refresh_groups()

    def _refresh_groups(self):
        """Redraw the group track and the Inspector's groups panel."""
        if self._alignment is None:
            self.plotGroupTrack.clear_groups()
        else:
            self.plotGroupTrack.set_groups(
                self.data_source.groups_for_analytes(self._x_of_analyte),
                self._x_of_analyte,
            )
        self.plotGroupTrack.setVisible(
            self._alignment is not None and self.plotGroupTrack.has_visible_groups()
        )
        self._refresh_groups_panel()

    def _refresh_groups_panel(self):
        selected = selection_analyte_uuids(self._selection)
        self.btnGroupSelected.setEnabled(bool(selected))
        in_view = (
            {a.uuid for a in self._alignment.analytes} if self._alignment else set()
        )
        self.groups_panel.refresh(in_view, selected)

    def _group_analytes(self, targets: Optional[list[HoverTarget]] = None):
        """Open the group dialog for `targets` (default: the selection)."""
        analyte_uuids = selection_analyte_uuids(
            self._selection if targets is None else targets
        )
        if self._alignment is None or not analyte_uuids:
            return
        clustering = self._active_clustering()
        provenance = {
            'alignment_uuid': self._alignment.uuid,
            'clustering': None if clustering is None else jsonable(clustering.params),
        }
        AnalyteGroupDialog(
            self.data_source, analyte_uuids, provenance, parent=self,
        ).exec_()

    # -- context menu -------------------------------------------------------

    def _show_context_menu(
            self,
            target: Optional[HoverTarget],
            global_pos,
    ):
        if target is None:
            return

        menu = QtWidgets.QMenu(self)

        if target.kind == 'ensemble':
            act_open = menu.addAction("Open in Ensemble Viewer")
            act_open.triggered.connect(
                lambda: self.sigViewEnsembleRequested.emit(target.ensemble)
            )
            act_sample = menu.addAction("Add sample to Sample Viewer")
            act_sample.triggered.connect(
                lambda: self.sigAddSamplesRequested.emit([target.sample_uuid])
            )
            if target.analyte and len(target.analyte.ensemble_map) > 1:
                act_all = menu.addAction("Add all aligned samples to Sample Viewer")
                act_all.triggered.connect(
                    lambda: self.sigAddSamplesRequested.emit(
                        list(target.analyte.ensemble_map.keys())
                    )
                )

        elif target.kind == 'analyte':
            act_all = menu.addAction("Add all aligned samples to Sample Viewer")
            act_all.triggered.connect(
                lambda: self.sigAddSamplesRequested.emit(
                    list(target.analyte.ensemble_map.keys())
                )
            )

        elif target.kind == 'sample':
            act_sample = menu.addAction("Add sample to Sample Viewer")
            act_sample.triggered.connect(
                lambda: self.sigAddSamplesRequested.emit([target.sample_uuid])
            )

        self._add_find_mfs_action(menu, target)

        targets = (
            self._selection
            if len(self._selection) > 1 and target in self._selection
            else [target]
        )
        if selection_analyte_uuids(targets):
            if not menu.isEmpty():
                menu.addSeparator()
            menu.addAction("Group selected analytes…").triggered.connect(
                lambda: self._group_analytes(targets)
            )

        if not menu.isEmpty():
            menu.exec_(global_pos)

    def _add_find_mfs_action(
            self,
            menu: QtWidgets.QMenu,
            target: HoverTarget,
    ):
        """
        'Auto find-mfs' on the right-clicked item: an ensemble, or all of an
        analyte's member ensembles. Right-clicking inside a multi-selection
        applies it to the whole selection.
        """
        targets = (
            self._selection
            if len(self._selection) > 1 and target in self._selection
            else [target]
        )
        targets = [t for t in targets if t.kind in ('ensemble', 'analyte')]
        ensembles = self._ensembles_for(targets)
        if not ensembles:
            return

        n = len(ensembles)
        noun = f"{n} ensemble{'' if n == 1 else 's'}"
        if len(targets) > 1:
            label = f"Auto find-mfs on {len(targets)} selected items ({noun})"
        elif targets[0].kind == 'analyte':
            label = f"Auto find-mfs on this analyte ({noun})"
        else:
            label = "Auto find-mfs on this ensemble"

        if not menu.isEmpty():
            menu.addSeparator()
        action = menu.addAction(label)
        action.setToolTip(
            "Formulae you picked yourself are kept; everything else is "
            "(re-)annotated with the saved find-mfs parameters"
        )
        action.triggered.connect(
            lambda: self.sigAutoFindMfsRequested.emit(ensembles)
        )

    def _ensembles_for(
            self,
            targets: list[HoverTarget],
    ) -> list['Ensemble']:
        """The ensembles behind `targets` (analytes -> their members), deduplicated."""
        if self._ctx is None:
            return []
        found: dict = {}
        for t in targets:
            if t.kind == 'ensemble' and t.ensemble is not None:
                found.setdefault(t.ensemble.uuid, t.ensemble)
            elif t.kind == 'analyte' and t.analyte is not None:
                for ensemble in self._ctx.members(t.analyte).values():
                    found.setdefault(ensemble.uuid, ensemble)
        return list(found.values())


def _locate_ensemble(
    alignment: 'EnsembleAlignment',
    ensemble_uuid,
) -> Optional[tuple[int, 'SampleUUID']]:
    """(analyte row, sample) of an ensemble in `alignment`, or None."""
    for aid, analyte in enumerate(alignment.analytes):
        for sample_uuid, ens_uuid in analyte.ensemble_map.items():
            if ens_uuid == ensemble_uuid:
                return aid, sample_uuid
    return None
