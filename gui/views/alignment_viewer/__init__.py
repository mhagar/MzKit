"""
Alignment Viewer — interactive feature-map for an ``EnsembleAlignment``.

Each Ensemble is a strip laid out by retention time within a per-sample
lane; strips of the same AlignedAnalyte share a colour and a connecting
line (see gui/widgets/AlignmentPlotWidget.py).

Cluster mode instead lays analytes out in the leaf order of a hierarchical
clustering by MS2 modified cosine (core/cli/cluster_analytes.py), with the
dendrogram drawn above the map (gui/widgets/AlignmentDendrogramWidget.py).
Clustering runs in the background via MainController (sigClusterRequested).

The selection on the map drives:
 - the Inspector side panel (properties / members / comparison table),
 - the bottom panel (MS1, MS2, chromatogram; see detail_panel.py),
 - the status bar (hover readout).

Actions into the Ensemble / Sample viewers go out via signals the
MainController wires up.
"""
from typing import Optional, TYPE_CHECKING

from PyQt5 import QtWidgets, QtCore, QtGui

from core.cli.cluster_analytes import ClusterParams, cluster_params_from_config
from core.utils.config import load_config
from gui.resources.AlignmentViewerWindow import Ui_Form
from gui.views.alignment_viewer.cluster_params_dialog import ClusterParamsDialog
from gui.views.alignment_viewer.context import AlignmentContext, ItemSpectra
from gui.views.alignment_viewer.detail_panel import DetailPanel
from gui.views.alignment_viewer.inspector import Inspector
from gui.views.alignment_viewer.render_params_dialog import RenderParamsDialog
from gui.widgets.alignment_plot.layout import HoverTarget
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
    sigClusterRequested = QtCore.pyqtSignal(object, object)  # EnsembleAlignment, ClusterParams

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
        self._compare_scores: list[Optional['PairScore']] = []
        self._compare_idx: int = 1

        self._params_dialog: Optional[RenderParamsDialog] = None

        # Cluster mode: results for the current alignment, keyed by params
        self._cluster_params: ClusterParams = cluster_params_from_config(
            load_config()
        )
        self._clusterings: dict[ClusterParams, 'AnalyteClustering'] = {}
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

        self.inspector.sigCompareRowSelected.connect(self._on_compare_row)
        self.inspector.sigMemberActivated.connect(
            self.sigViewEnsembleRequested
        )

        self.btnLinkRT.toggled.connect(self._update_rt_link)
        self.btnRenderParams.clicked.connect(self._show_params_dialog)
        self.btnClusterMode.toggled.connect(self._on_cluster_mode_toggled)
        self.btnClusterParams.clicked.connect(self._show_cluster_dialog)
        self.plotDendrogram.sigSubtreeClicked.connect(self._on_subtree_clicked)
        self.plotDendrogram.sigNodeHovered.connect(self._on_node_hovered)

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

    # -- public API (kept stable for MainController / SubWindowManager) ------

    def set_alignment(
        self,
        alignment: 'EnsembleAlignment',
    ):
        self._alignment = alignment
        self._ctx = AlignmentContext(alignment, self.data_source)
        self._selection = []
        self._clear_cluster_mode()
        self.detail_panel.reset()
        self._render()
        self.inspector.show_summary(alignment)
        self.status_bar.showMessage(self._summary_text())

    def reset_for_new_project(self):
        self._alignment = None
        self._ctx = None
        self._selection = []
        self._clear_cluster_mode()
        self.plot_item.clear_alignment()
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
            self.plot_item.set_alignment(
                self._alignment, self.data_source, self._params,
                preserve_range=preserve_range,
                preserve_selection=preserve_selection,
            )
        else:
            slot_of = {u: i for i, u in enumerate(clustering.analyte_uuids)}
            self.plot_item.set_alignment(
                self._alignment, self.data_source, self._params,
                preserve_range=preserve_range,
                preserve_selection=preserve_selection,
                x_of=lambda aid, analyte, ens: slot_of.get(analyte.uuid),
                width_scale=self._params.cluster_width_scale,
                x_axis_visible=False,
                stagger=False,   # one analyte per column: nothing to overlap
            )
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

    def _active_clustering(self) -> Optional['AnalyteClustering']:
        """The clustering to lay out by, or None for RT layout."""
        if not self.btnClusterMode.isChecked():
            return None
        return self._clusterings.get(self._cluster_params)

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
        if checked and self._cluster_params not in self._clusterings:
            self._request_clustering()
            return
        # Units change between modes: keep the selection, reset the range
        self._render(preserve_selection=True)

    def _request_clustering(self):
        """Ask MainController to cluster; the map stays as is meanwhile."""
        self.status_bar.showMessage("Clustering analytes by MS2…")
        self.sigClusterRequested.emit(self._alignment, self._cluster_params)

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

    def _show_cluster_dialog(self):
        if self._cluster_dialog is None:
            self._cluster_dialog = ClusterParamsDialog(
                self._cluster_params, parent=self,
            )
            self._cluster_dialog.sigApply.connect(self._on_cluster_params)
        self._cluster_dialog.show()
        self._cluster_dialog.raise_()

    def _on_cluster_params(self, params: ClusterParams):
        self._cluster_params = params
        if self._alignment is None:
            return
        if not self.btnClusterMode.isChecked():
            self.btnClusterMode.setChecked(True)   # requests / renders
        elif params in self._clusterings:
            self._render(preserve_selection=True)
        else:
            self._request_clustering()

    def _on_subtree_clicked(self, analyte_uuids: list, additive: bool):
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

    def _show_selection(self):
        ctx, targets = self._ctx, self._selection
        if ctx is None or not targets:
            self.inspector.show_summary(self._alignment)
            self.detail_panel.clear()
            return

        if len(targets) == 1:
            self._show_single(ctx, targets[0])
            return

        items: list[ItemSpectra] = []
        for target in targets:
            item = ctx.item_spectra(target)
            if item is not None:
                items.append(item)
        if len(items) < 2:
            self._show_single(ctx, targets[0])
            return

        self._compare_items = items
        self._compare_scores = [None] + [
            ctx.score(items[0], item) for item in items[1:]
        ]
        self._compare_idx = min(self._compare_idx, len(items) - 1)
        self.inspector.show_compare(
            items, self._compare_scores, self._compare_idx,
        )
        self._show_compare_plots()

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

    def _on_compare_row(self, row: int):
        if row == self._compare_idx or row >= len(self._compare_items):
            return
        self._compare_idx = row
        self._show_compare_plots()

    def _show_compare_plots(self):
        sample_uuids, ensembles = self._selected_member_ensembles()
        idx = self._compare_idx
        self.detail_panel.show_compare(
            self._ctx,
            self._compare_items[0],
            self._compare_items[idx],
            self._compare_scores[idx],
            sample_uuids,
            ensembles,
        )

    def _selected_member_ensembles(
        self,
    ) -> tuple[list['SampleUUID'], list['Ensemble']]:
        """Every ensemble in the selection (analytes expand to members)."""
        sample_uuids, ensembles = [], []
        for target in self._selection:
            if target.kind == 'ensemble' and target.ensemble is not None:
                sample_uuids.append(target.sample_uuid)
                ensembles.append(target.ensemble)
            elif target.kind == 'analyte' and target.analyte is not None:
                for sample_uuid, ens in self._ctx.members(target.analyte).items():
                    sample_uuids.append(sample_uuid)
                    ensembles.append(ens)
        return sample_uuids, ensembles

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
            agreement = ctx.formula_agreement(analyte)
            if agreement:
                parts.append(f"{agreement[0]} ({agreement[1]}/{agreement[2]})")

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
        if clustering is not None and clustering.n_dropped_no_ms2:
            n = clustering.n_dropped_no_ms2
            text += (
                f"  ·  {n} analyte{'' if n == 1 else 's'} without MS2 "
                f"not clustered"
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
