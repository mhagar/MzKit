import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from core.data_structs import Ensemble
from gui.widgets.ChromPlotWidget import ChromGraphicItem
from gui.resources.EnsembleViewerWindow import Ui_Form
from gui.views.ensemble_viewer.tools import (
    ToolType, ToolStage, Mode,
    ToolManager,
)
from gui.views.ensemble_viewer.find_formula import Controller as FmlaCtrlr

from typing import TYPE_CHECKING, Optional, Literal

if TYPE_CHECKING:
    from core.interfaces.data_sources import SampleDataSource
    from gui.widgets.MSPlotWidget import MSPlotWidget


class EnsembleViewer(
    QtWidgets.QWidget,
    Ui_Form,
):
    sigSelectionMade = QtCore.pyqtSignal()
    sigConfigurationMade = QtCore.pyqtSignal()

    def __init__(
        self,
        data_source: 'SampleDataSource'
    ):
        super().__init__()
        self.setupUi(self)
        self.data_source = data_source
        self.tool_manager = ToolManager()
        self.find_formula_ctlr = FmlaCtrlr(self)

        self.ensemble: Optional['Ensemble'] = None

        self.base_chrom: Optional[np.ndarray] = None
        self.ms1_chroms: list[np.ndarray] = []
        self.ms2_chroms: list[np.ndarray] = []

        self._setup_plots()
        self._setup_actions()
        self._setup_tool_listeners()
        self._connect_signals()
        self._hide_misc_plots()

    def _connect_signals(self):
        self.ms1_plot.sigMSSignalClicked.connect(
            self.on_ms1_signal_clicked
        )

        self.ms2_plot.sigMSSignalClicked.connect(
            self.on_ms2_signal_clicked
        )

        self.checkNormalize.clicked.connect(
            self.update_chromatogram_plot
        )

        self.checkNormalize.clicked.connect(
            self.populate_chromatogram_plot
        )

        # *** TOOLS ***
        self.find_formula_ctlr.sigSignalSelected.connect(
            self._update_signal_selection_graphics
        )

        self.find_formula_ctlr.sigSelectionCleared.connect(
            self._clear_signal_selection_graphics
        )

        # FOR TESTING:
        self.tool_manager.sigToolChanged.connect(
            lambda tool: print(
                f"Tool changed to: {tool}"
            )
        )

        self.tool_manager.sigStageChanged.connect(
            lambda stage: print(
                f"Stage changed to: {stage}"
            )
        )

        self.tool_manager.sigModeChanged.connect(
            lambda mode: print(
                f"Mode changed to: {mode}"
            )
        )

    def _setup_plots(self):
        """
        """
        # Configure MS plots
        ms2_vb = self.ms2_plot.pi.vb
        ms1_vb = self.ms1_plot.pi.vb
        ms2_vb.setXLink(ms1_vb)

        # Configure corr plot
        self.corrPlotWidget.addItem(
            pg.InfiniteLine(
                angle=45,
            )
        )

    def _setup_actions(self):
        """
        Configure action group for mutually exclusive tools
        """
        # This is for switching between 'composite' and 'scan' mode
        self.mode_action_group = QtWidgets.QActionGroup(self)
        self.mode_action_group.triggered.connect(
            self.on_mode_action_triggered
        )

        # This is for switching between 'find formula' and 'measure loss'
        self.tool_action_group = QtWidgets.QActionGroup(self)
        self.tool_action_group.triggered.connect(
            self.on_tool_action_triggered
        )

        for action, action_type, btn in [
            (self.actionScan,       'mode', self.toolScan),
            (self.actionComposite,  'mode', self.toolComposite),
            (self.actionFindFormula,'tool', self.toolFindFormula),
            (self.actionMeasureLoss,'tool', self.toolMeasureLoss),
        ]:
            action: QtWidgets.QAction
            action_type: Literal['mode', 'tool']
            btn: QtWidgets.QToolButton

            btn.setDefaultAction(action)

            match action_type:
                case 'mode':
                    self.mode_action_group.addAction(
                        action
                    )

                case 'tool':
                    self.tool_action_group.addAction(
                        action
                    )

        # Initialize in Composite mode
        self.actionComposite.trigger()

    def _setup_tool_listeners(self):
        """
        Connect the tool manager to whatever objects need to respond to
        tool activations
        """
        for listener in [
            self,
            self.ms1_plot,
            self.ms2_plot,
        ]:
            self.tool_manager.register_listener(listener)

    def on_mode_action_triggered(
        self,
        action: QtWidgets.QAction,
    ):
        """
        If user switches to scan/composite mode,
        tells ToolManager to either reset or switch to scan tool
        """
        mode = {
            self.actionScan: Mode.SCAN,
            self.actionComposite: Mode.COMPOSITE,
        }.get(action)

        if not mode:
            raise ValueError(
                f"QAction has no corresponding mode: {action}"
            )

        self.tool_manager.request_mode(mode)

    def on_tool_action_triggered(
        self,
        action: QtWidgets.QAction,
    ):
        """
        Tells ToolManager that user wants to switch to a tool
        """
        tool_map = {
            self.actionFindFormula: ToolType.FINDFORMULA,
            self.actionMeasureLoss: ToolType.MEASURELOSS,
        }

        tool_type = tool_map.get(
            action,
            ToolType.NONE,  # Default if invalid tool
        )

        self.tool_manager.request_tool(tool_type)

    def on_tool_type_changed(
        self,
        tool: ToolType,
    ):
        """
        Controls this widget's behaviour when tool is changed
        """
        if tool == ToolType.NONE:
            return

        match tool:
            case ToolType.FINDFORMULA:
                self.tool_manager.request_next_stage()

            case ToolType.MEASURELOSS:
                pass

        print(
            f"EnsembleViewer: changed to {tool}"
        )
        self._update_tool_buttons()
        self._clear_signal_selection_graphics()

    def on_tool_stage_changed(
        self,
        stage: ToolStage,
    ):
        print(
            f"EnsembleViewer: changed to {stage}"
        )
        self._update_tool_buttons()

    def on_tool_mode_changed(
        self,
        mode: Mode
    ):
        print(
            f"EnsembleViewer: changed to {mode}"
        )
        self._update_tool_buttons()

    def on_tool_reset(self):
        self._update_tool_buttons()
        self._clear_signal_selection_graphics()

    def _update_tool_buttons(
        self,
    ):
        """
        Updates the tool buttons to match the state of ToolManager,
        ***without emitting signals***!!
        """
        _ = (
            ( self.toolScan,         'mode',      Mode.SCAN),
            ( self.toolComposite,    'mode',      Mode.COMPOSITE),
            ( self.toolFindFormula,  'tool',  ToolType.FINDFORMULA),
            ( self.toolMeasureLoss,  'tool',  ToolType.MEASURELOSS),
        )

        for btn, activation_type, activation_condition in _:
            activation_type: Literal['mode', 'tool']
            btn: QtWidgets.QToolButton

            with QtCore.QSignalBlocker(btn):
                match activation_type:
                    case 'mode':
                        btn.setChecked(
                            activation_condition == self.tool_manager.active_mode
                        )

                    case 'tool':
                        btn.setChecked(
                            activation_condition == self.tool_manager.active_tool
                        )

    def set_ensemble(
        self,
        ensemble: 'Ensemble'
    ):
        self.ensemble = ensemble

        self.base_chrom = self.ensemble.get_base_chromatogram(ms_level=1)
        self.ms1_chroms = self.ensemble.get_chromatograms(ms_level=1)
        self.ms1_chroms = self.ensemble.get_chromatograms(ms_level=1)

        self.populate_plots()

    def _hide_misc_plots(self):
        self.checkShowMiscPlots.setChecked(False)
        self.tabWidget.setVisible(False)

    def _show_misc_plots(self):
        self.checkShowMiscPlots.setChecked(True)
        self.tabWidget.setVisible(True)

    def populate_plots(
        self,
    ):
        if not self.ensemble:
            return

        self.populate_spectrum_plot()
        self.populate_chromatogram_plot()
        # self.populate_correlation_plot()

    def populate_spectrum_plot(self):
        # Add spectrum arrays
        for i, plot in enumerate(
            [self.ms1_plot, self.ms2_plot, ]
        ):
            plot.setSpectrumArray(
                self.ensemble.get_spectrum(ms_level=i + 1)  # type: ignore
            )

    def populate_chromatogram_plot(self):
        self.chromPlotWidget.setChromArray(
            self._apply_chrom_transforms(self.base_chrom)
        )

    def update_chromatogram_plot(
        self,
    ):
        # Repopulate chromatogram plot
        self.chromPlotWidget.clearPeaks()
        colors = {
            1: 'm',
            2: 'g',
        }

        for idx, chrom in enumerate(self.ms1_chroms):
            self.chromPlotWidget.addPeak(
                        chrom=self._apply_chrom_transforms(chrom),
                        uuid=idx,
                        color=colors[1],
                    )

        for idx, chrom in enumerate(self.ms2_chroms):
            idx += len(self.ms1_chroms)
            self.chromPlotWidget.addPeak(
                chrom=self._apply_chrom_transforms(chrom),
                uuid=idx,
                color=colors[2],
            )

    def _apply_chrom_transforms(
        self,
        chrom: np.ndarray,
    ) -> np.ndarray:
        """
        Applies transforms based on which checkboxes are activated
        (i.e. normalize, diff,)
        """
        _chrom = chrom.copy()
        if self._normalize_is_checked():
            _chrom = _normalize_chrom_arr(chrom)

        if self._diff_is_checked():
            _chrom = _diff_chrom_arr(chrom)

        return _chrom

    def _normalize_is_checked(self) -> bool:
        return self.checkNormalize.isChecked()

    def _diff_is_checked(self) -> bool:
        return self.checkDiff.isChecked()

    def populate_correlation_plot(self):
        """
        Populates a 'correlation plot' where X is the
        intensity of the base cofeature, and Y is the intensity of the
        target co feature
        :return:
        """
        self.corrPlotWidget.plotItem.clear()

        colors = {
            1: (255, 0, 255),  # Magenta
            2: (0, 255, 0),    # Green
        }

        # Get base feature chrom
        base_chrom = self.ensemble.get_base_chromatogram(ms_level=1)
        base_intsy = self.ensemble.base_intsy

        for ms_level in (1, 2):
            ms_level: Literal[1, 2]

            chroms = self.ensemble.get_chromatograms(
                ms_level=ms_level,
            )

            for chrom in chroms:

                ref_arr, tgt_arr = _match_chrom_arrys(
                    reference_chrom=base_chrom, # type: ignore
                    target_chrom=chrom, # type: ignore
                    normalize=True,
                )

                opacity: int = max(
                    int(
                        255 * max(chrom['intsy']) / base_intsy
                    ),

                    20,
                )

                pen = pg.mkPen(
                    *colors[ms_level], opacity
                )

                chrom_graphic = ChromGraphicItem(
                    intsy_arr=tgt_arr['intsy'],
                    rt_arr=ref_arr['intsy'],  # Note how I'm filling rt arg with intsy
                    pen=pen,
                )

                self.corrPlotWidget.addItem(
                    chrom_graphic
                )

    def on_ms1_signal_clicked(
        self,
        data: tuple[int, float], # [spec_idx, mz_float]
    ):
        """
        Called when user clicks on a signal in the MS1 spectrum
        """
        spec_idx, mz = data
        print(
            f"Clicked: {data}"
        )

        if not spec_idx:
            return

        self.find_formula_ctlr.handle_ms_signal_clicked(
            data=data,
            ms_level=1,
        )

        self.ms1_chroms = self.ensemble.get_chromatograms(
            ms_level=1,
            idxs=slice(spec_idx, spec_idx + 1),
        )

        self.update_chromatogram_plot()

    def on_ms2_signal_clicked(
        self,
        data: tuple[int, float], # [spec_idx, mz_float]
    ):
        """
        Called when user clicks on a signal in the MS1 spectrum
        """
        spec_idx, mz = data
        print(
            f"Clicked: {data}"
        )

        if not spec_idx:
            return

        self.find_formula_ctlr.handle_ms_signal_clicked(
            data=data,
            ms_level=2,
        )

        self.ms2_chroms = self.ensemble.get_chromatograms(
            ms_level=2,
            idxs=slice(spec_idx, spec_idx + 1),
        )

        self.update_chromatogram_plot()

    def _update_signal_selection_graphics(
        self,
        data: tuple[int, float],
        level: Literal[1, 2, None],
        is_selected: bool,
    ):
        """
        Called whenever user selects/deselects a signal.
        Called with `level = None` if user switches from MS1 <-> MS2 spectrum
        """
        # Get the plot widget the selection is in
        plot_widget: 'MSPlotWidget' = {
            1: self.ms1_plot,
            2: self.ms2_plot,
            None: None,
        }.get(level)

        if not plot_widget:
            return

        # Tell MSPlotWidget to add/remove marker
        match is_selected:
            case True:
                plot_widget.add_signal_marker(
                    spec_idx=data[0]
                )

            case False:
                plot_widget.remove_signal_marker(
                    spec_idx=data[0]
                )


    def _clear_signal_selection_graphics(self):
        for plot_widget in (self.ms1_plot, self.ms2_plot):
            plot_widget: 'MSPlotWidget'
            plot_widget.clear_signal_markers()


def _match_chrom_arrys(
    reference_chrom: np.ndarray[float],
    target_chrom: np.ndarray[float],
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a reference chrom_array and a target chrom_array,
    returns a slice where the two chroms overlap in time.

    :param reference_chrom:
    :param target_chrom:
    :param normalize: If true, normalizes intensities of both chroms
    such that their maximum intsy = 1
    :return:
    """
    # Slice two arrays into just the overlapping regions
    (ref_start, ref_end), (tgt_start, tgt_end) = _find_overlap_region(
        reference_chrom['rt'],
        target_chrom['rt'],
    )

    ref_arr = reference_chrom[ref_start: ref_end].copy()
    tgt_arr = target_chrom[tgt_start: tgt_end].copy()

    # Normalize both of them to 1
    if normalize:
        ref_arr['intsy'] = ref_arr['intsy'] / max(ref_arr['intsy'])
        tgt_arr['intsy'] = tgt_arr['intsy'] / max(tgt_arr['intsy'])

    return ref_arr, tgt_arr


def _find_overlap_region(
    arr_a: np.ndarray[float],
    arr_b: np.ndarray[float],
) -> Optional[tuple[tuple, tuple]]:
    """
    Returns indices where arr_a and arr_b overlap in values,
    assuming that both arrays contain monotonically increasing elements

    (i.e. represent successive retention time values)
    """
    start = max(arr_a[0], arr_b[0])
    end = min(arr_a[-1], arr_b[-1])

    if start > end:
        return None  # No overlap

    # Find indices for overlapping region
    a_start_idx = np.searchsorted(
        arr_a, start, side='left',
    )

    a_end_idx = np.searchsorted(
        arr_a, end, side='right',
    )

    b_start_idx = np.searchsorted(
        arr_b, start, side='left',
    )

    b_end_idx = np.searchsorted(
        arr_b, end, side='right',
    )

    return (a_start_idx, a_end_idx), (b_start_idx, b_end_idx)


def _normalize_chrom_arr(
    chrom: np.ndarray
) -> np.ndarray:
    """
    Returns a chrom array that's been normalized such that
    maximum intensity = 1.0
    """
    arr = chrom.copy()
    arr['intsy'] = arr['intsy']/max(arr['intsy'])

    return arr


def _diff_chrom_arr(
    chrom: np.ndarray
) -> np.ndarray:
    """
    Returns a chrom array that's been 'differentiated'
    by subtracting each intsy value from the next
    """
    arr = chrom.copy()
    arr['intsy'] = np.diff(arr['intsy'], append=0.0)

    return arr

