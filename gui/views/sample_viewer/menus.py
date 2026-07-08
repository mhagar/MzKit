import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt

from core.utils.config import save_config, load_default_config
from core.cli.generate_ensemble import (
    AutoEnsembleParams,
    auto_params_from_config,
    auto_params_to_config,
)


class FingerprintDisplayMenu(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlags(  # For pop-up style behaviour
            Qt.Popup | Qt.FramelessWindowHint
        )

        self.setMaximumSize(
            QtCore.QSize(400, 200)
        )

        layout = QtWidgets.QVBoxLayout(self)
        # self.plot_widget = pg.PlotWidget()
        # self.plot_widget.setFixedSize(200, 300)

        self.graphics_layout_widget = pg.GraphicsLayoutWidget()

        self.colorbar = pg.ColorBarItem(
            values=(-1, 1),
            colorMap='CET-D1A',
            orientation='horizontal',
            limits=(-1.1, 1.1),
            rounding=0.1,
            width=100,
        )
        self.graphics_layout_widget.addItem(
            self.colorbar
        )

        layout.addWidget(
            self.graphics_layout_widget, # type: ignore
        )


class EnsembleExtractionSettingsMenu(QtWidgets.QWidget):
    """
    A drop-down menu containing settings for ensemble extraction

    # TODO: Should probably turn this into a .ui file; it's getting complex
    """
    sigSettingsChanged = QtCore.pyqtSignal(
        dict,  # { ms1_corr_threshold: float,
    )          #   ms2_corr_threshold: float,
               #   min_intsy: float,
               #   use_rel_intsy: bool,
               #   method: str ('cosine' | 'pearson'), }

    def __init__(self, parent=None, config=None):
        super().__init__(parent)

        # Persisted auto-generation defaults live in the [auto_ensemble] config
        # section; may be None (e.g. manual-extraction-only contexts / tests).
        self.config = config

        self.setWindowFlags(  # For pop-up style behaviour
            Qt.Popup | Qt.FramelessWindowHint
        )

        self.setMaximumSize(
            QtCore.QSize(820, 760)
        )

        # Layout widgets
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        columns_layout = QtWidgets.QHBoxLayout()
        left_layout = QtWidgets.QVBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()
        for _col in (left_layout, right_layout):
            _col.setAlignment(Qt.AlignTop)
            _col.setSpacing(2)  # labels stack above controls; keep rows tight
        columns_layout.addLayout(left_layout)
        columns_layout.addLayout(right_layout)
        main_layout.addLayout(columns_layout)
        layout = left_layout

        # *** Scoring method ***
        # Cosine similarity of XICs empirically discriminates peak shapes
        # better than Pearson correlation, so it's the default. The combo's
        # userData carries the metric string consumed downstream.
        method_label = QtWidgets.QLabel("Scoring method:")
        method_tip = (
            "Cosine similarity of XICs empirically discriminates peak shapes\n"
            "better than Pearson correlation, so it's the default."
        )
        method_label.setToolTip(method_tip)
        layout.addWidget(method_label)
        self.comboMethod = QtWidgets.QComboBox()
        self.comboMethod.addItem("Cosine", "cosine")
        self.comboMethod.addItem("Pearson", "pearson")
        self.comboMethod.setCurrentIndex(0)  # cosine
        self.comboMethod.setToolTip(method_tip)
        self.comboMethod.currentIndexChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.comboMethod
        )

        # *** Correlation threshold spinners ***
        # Defaults are tuned for cosine (runs higher than Pearson):
        # ~0.95 MS1 / ~0.90 MS2. Step in 0.01 for fine control near the top.
        ms1_thresh_label = QtWidgets.QLabel("MS1 score threshold:")
        ms1_thresh_tip = (
            "Minimum MS1 XIC similarity score for a candidate to join the\n"
            "ensemble. Defaults are tuned for cosine (which is higher than\n"
            "Pearson): ~0.95 MS1 / ~0.90 MS2."
        )
        ms1_thresh_label.setToolTip(ms1_thresh_tip)
        layout.addWidget(ms1_thresh_label)
        self.spinnerMS1Threshold = QtWidgets.QDoubleSpinBox()
        self.spinnerMS1Threshold.setSingleStep(0.01)
        self.spinnerMS1Threshold.setMinimum(0.01)
        self.spinnerMS1Threshold.setMaximum(0.999)
        self.spinnerMS1Threshold.setValue(0.95)
        self.spinnerMS1Threshold.setToolTip(ms1_thresh_tip)
        self.spinnerMS1Threshold.valueChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.spinnerMS1Threshold
        )


        ms2_thresh_label = QtWidgets.QLabel("MS2 score threshold:")
        ms2_thresh_tip = (
            "[DIA] Minimum MS2 feature similarity score for a candidate to join\n"
            "the ensemble. Only relevant when DIA MS2 scans are available."
        )
        ms2_thresh_label.setToolTip(ms2_thresh_tip)
        layout.addWidget(ms2_thresh_label)
        self.spinnerMS2Threshold = QtWidgets.QDoubleSpinBox()
        self.spinnerMS2Threshold.setSingleStep(0.01)
        self.spinnerMS2Threshold.setMinimum(0.01)
        self.spinnerMS2Threshold.setMaximum(0.999)
        self.spinnerMS2Threshold.setValue(0.90)
        self.spinnerMS2Threshold.setToolTip(ms2_thresh_tip)
        self.spinnerMS2Threshold.valueChanged.connect(
            self.param_changed

        )
        layout.addWidget(
            self.spinnerMS2Threshold
        )

        # *** Minimum Intensity Spinner
        min_intsy_label = QtWidgets.QLabel("Minimum intensity:")
        min_intsy_tip = (
            "Signals below this intensity are ignored when looking for\n"
            "cofeatures to add to the ensemble."
        )
        min_intsy_label.setToolTip(min_intsy_tip)
        layout.addWidget(min_intsy_label)
        self.spinnerMinIntsy = QtWidgets.QDoubleSpinBox()
        self.spinnerMinIntsy.setMinimum(1)
        self.spinnerMinIntsy.setMaximum(1e9)
        self.spinnerMinIntsy.setStepType(
                QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType
        )
        self.spinnerMinIntsy.setValue(2000)
        self.spinnerMinIntsy.setToolTip(min_intsy_tip)
        self.spinnerMinIntsy.valueChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.spinnerMinIntsy
        )

        # *** Relative intensities checkbox ***
        self.checkRelativeIntensity = QtWidgets.QCheckBox()
        self.checkRelativeIntensity.setText(
            "Use relative signal intensity"
        )
        self.checkRelativeIntensity.setChecked(True)
        self.checkRelativeIntensity.setToolTip(
            "Score cofeatures on intensity relative to the seed peak,\n"
            "rather than on absolute intensity. \n"
            "(Makes no difference when using cosine)"
        )
        self.checkRelativeIntensity.toggled.connect(
            self.param_changed
        )

        layout.addWidget(
            self.checkRelativeIntensity
        )

        # =====================================================================
        # Auto-generation-only settings. These are ignored by manual extraction
        # (the user picks the seed by hand there) but consumed by the auto path
        # via get_auto_params(). The shared settings above carry over to both.
        # =====================================================================
        auto_sep = QtWidgets.QFrame()
        auto_sep.setFrameShape(QtWidgets.QFrame.HLine)
        auto_sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(auto_sep)

        auto_header = QtWidgets.QLabel("Auto-generation:")
        auto_header.setStyleSheet("font-weight: bold;")
        layout.addWidget(auto_header)

        # *** Seed (parent) threshold ***
        # Only the tallest signal of an ensemble must exceed this to seed a new
        # ensemble; cofeatures still only need the (lower) "Minimum intensity".
        seed_thresh_label = QtWidgets.QLabel("Seed min intensity:")
        seed_thresh_tip = (
            "Only the tallest signal of an ensemble (the 'seed') must exceed\n"
            "this to seed a new ensemble; cofeatures still only need the\n"
            "(lower) 'Minimum intensity' above."
        )
        seed_thresh_label.setToolTip(seed_thresh_tip)
        layout.addWidget(seed_thresh_label)
        self.spinnerParentThreshold = QtWidgets.QDoubleSpinBox()
        self.spinnerParentThreshold.setMinimum(1)
        self.spinnerParentThreshold.setMaximum(1e9)
        self.spinnerParentThreshold.setStepType(
            QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType
        )
        self.spinnerParentThreshold.setValue(20000)
        self.spinnerParentThreshold.setToolTip(seed_thresh_tip)
        self.spinnerParentThreshold.valueChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.spinnerParentThreshold
        )

        # *** Extraction window strategy ***
        # 'peak_bounds' (non-fixed, follows the peak) is the default; 'fixed'
        # uses +-half-width scans around the apex.
        window_label = QtWidgets.QLabel("Extraction window:")
        window_tip = (
            "'Peak bounds' (default) follows the detected peak's own width;\n"
            "'Fixed width' instead uses a fixed +-half-width (in scans)\n"
            "around the apex, set below."
        )
        window_label.setToolTip(window_tip)
        layout.addWidget(window_label)
        self.comboWindow = QtWidgets.QComboBox()
        self.comboWindow.addItem("Peak bounds (auto width)", "peak_bounds")
        self.comboWindow.addItem("Fixed width", "fixed")
        self.comboWindow.setCurrentIndex(0)  # peak_bounds
        self.comboWindow.setToolTip(window_tip)
        self.comboWindow.currentIndexChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.comboWindow
        )

        # *** Fixed-window half-width (scans) ***
        # Only used when window strategy == 'fixed'.
        half_width_label = QtWidgets.QLabel("Fixed window half-width (scans):")
        half_width_tip = ("Only used when Extraction window == 'Fixed width'. \n"
                          "Pads Ensemble to this many scans")
        half_width_label.setToolTip(half_width_tip)
        layout.addWidget(half_width_label)
        self.spinnerHalfWidth = QtWidgets.QSpinBox()
        self.spinnerHalfWidth.setMinimum(1)
        self.spinnerHalfWidth.setMaximum(1000)
        self.spinnerHalfWidth.setValue(10)
        self.spinnerHalfWidth.setToolTip(half_width_tip)
        self.spinnerHalfWidth.valueChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.spinnerHalfWidth
        )

        # *** Adduct-aware grouping ***
        # Loosen the MS1 threshold to the value below when the seed<->candidate
        # delta-m/z matches a known adduct / isotope / neutral loss.
        self.checkAdductAware = QtWidgets.QCheckBox()
        self.checkAdductAware.setText(
            "Adduct-aware grouping"
        )
        self.checkAdductAware.setChecked(True)
        self.checkAdductAware.setToolTip(
            "Loosen the MS1 threshold to the value below when the\n"
            "seed<->candidate delta-m/z matches a known adduct, isotope,\n"
            "or neutral loss."
        )
        self.checkAdductAware.toggled.connect(
            self.param_changed
        )
        layout.addWidget(
            self.checkAdductAware
        )

        loose_thresh_label = QtWidgets.QLabel("Adduct (loosened) MS1 threshold:")
        loose_thresh_tip = (
            "MS1 threshold applied instead of the normal one above when\n"
            "adduct-aware grouping recognizes the delta-m/z. Only used\n"
            "when 'Adduct-aware grouping' is checked."
        )
        loose_thresh_label.setToolTip(loose_thresh_tip)
        layout.addWidget(loose_thresh_label)
        self.spinnerLooseThreshold = QtWidgets.QDoubleSpinBox()
        self.spinnerLooseThreshold.setSingleStep(0.01)
        self.spinnerLooseThreshold.setMinimum(0.01)
        self.spinnerLooseThreshold.setMaximum(0.999)
        self.spinnerLooseThreshold.setValue(0.80)
        self.spinnerLooseThreshold.setToolTip(loose_thresh_tip)
        self.spinnerLooseThreshold.valueChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.spinnerLooseThreshold
        )

        # Everything below fills the right column.
        layout = right_layout

        # ---------------------------------------------------------------------
        # Peak validation (is_peak). Gates whether a seed is a real
        # chromatographic peak vs. a noise spike / slope / persistent
        # background 'whine'. Only used by auto-generation.
        # ---------------------------------------------------------------------
        peak_header = QtWidgets.QLabel("Peak validation:")
        peak_header.setStyleSheet("font-weight: bold;")
        layout.addWidget(peak_header)

        # *** Validation method ***
        # "prominence": bilateral - both shoulders must descend to baseline
        # (strict; best for isolated peaks). "flank": unilateral, neighbour-
        # invariant - the steeper shoulder must descend; keeps peaks fused to a
        # taller neighbour while still rejecting whine.
        peak_method_label = QtWidgets.QLabel("Method:")
        peak_method_tip = (
            "'Prominence' (bilateral): both shoulders must descend to\n"
            "baseline. Strict; best for isolated peaks.\n"
            "'Steepest flank' (unilateral, neighbour-invariant): only the\n"
            "steeper shoulder must descend; keeps peaks fused to a taller\n"
            "neighbour while still rejecting persistent background whine."
        )
        peak_method_label.setToolTip(peak_method_tip)
        layout.addWidget(peak_method_label)
        self.comboPeakMethod = QtWidgets.QComboBox()
        self.comboPeakMethod.addItem("Prominence (both shoulders)", "prominence")
        self.comboPeakMethod.addItem("Steepest flank (keeps fused peaks)", "flank")
        self.comboPeakMethod.setCurrentIndex(0)  # prominence
        self.comboPeakMethod.setToolTip(peak_method_tip)
        self.comboPeakMethod.currentIndexChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.comboPeakMethod
        )

        # *** Minimum peak width (scans) ***
        min_peak_width_label = QtWidgets.QLabel("Min peak width (scans):")
        min_peak_width_tip = (
            "Peaks narrower than this many scans are rejected as noise."
        )
        min_peak_width_label.setToolTip(min_peak_width_tip)
        layout.addWidget(min_peak_width_label)
        self.spinnerMinPeakWidth = QtWidgets.QSpinBox()
        self.spinnerMinPeakWidth.setMinimum(1)
        self.spinnerMinPeakWidth.setMaximum(1000)
        self.spinnerMinPeakWidth.setValue(5)
        self.spinnerMinPeakWidth.setToolTip(min_peak_width_tip)
        self.spinnerMinPeakWidth.valueChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.spinnerMinPeakWidth
        )

        # *** Minimum prominence ***
        # Apex must rise this fraction above the surrounding baseline on its
        # weaker shoulder. 0.5 => apex >= 2x baseline. Raise to reject broad
        # background bumps.
        min_prominence_label = QtWidgets.QLabel("Min prominence (0-1):")
        min_prominence_tip = (
            "Apex must rise this fraction above the surrounding baseline\n"
            "on its weaker shoulder. 0.5 => apex must be >= 2x baseline. \n"
            " Raise to reject broad background bumps."
        )
        min_prominence_label.setToolTip(min_prominence_tip)
        layout.addWidget(min_prominence_label)
        self.spinnerMinProminence = QtWidgets.QDoubleSpinBox()
        self.spinnerMinProminence.setSingleStep(0.05)
        self.spinnerMinProminence.setMinimum(0.0)
        self.spinnerMinProminence.setMaximum(1.0)
        self.spinnerMinProminence.setValue(0.5)
        self.spinnerMinProminence.setToolTip(min_prominence_tip)
        self.spinnerMinProminence.valueChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.spinnerMinProminence
        )

        # *** Baseline percentile ***
        # Percentile used to estimate each shoulder's baseline. Higher judges
        # prominence against a higher surrounding level, so bumps on elevated
        # background get rejected more readily.
        baseline_pct_label = QtWidgets.QLabel("Baseline percentile (0-100):")
        baseline_pct_tip = (
            "Percentile used to estimate each shoulder's baseline. Higher\n"
            "judges prominence against a higher surrounding level, so\n"
            "bumps on elevated background get rejected more aggressively."
        )
        baseline_pct_label.setToolTip(baseline_pct_tip)
        layout.addWidget(baseline_pct_label)
        self.spinnerBaselinePct = QtWidgets.QDoubleSpinBox()
        self.spinnerBaselinePct.setSingleStep(5.0)
        self.spinnerBaselinePct.setMinimum(0.0)
        self.spinnerBaselinePct.setMaximum(100.0)
        self.spinnerBaselinePct.setValue(10.0)
        self.spinnerBaselinePct.setToolTip(baseline_pct_tip)
        self.spinnerBaselinePct.valueChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.spinnerBaselinePct
        )

        # *** Edge fraction ***
        # find_peak_boundaries stops descending when intensity falls below this
        # fraction of the apex. Smaller => wider segments.
        edge_fraction_label = QtWidgets.QLabel("Edge fraction (0-1):")
        edge_fraction_tip = (
            "Peak-boundary detection stops descending when intensity\n"
            "falls below this fraction of the apex. "
            "Smaller => wider segments."
        )
        edge_fraction_label.setToolTip(edge_fraction_tip)
        layout.addWidget(edge_fraction_label)
        self.spinnerEdgeFraction = QtWidgets.QDoubleSpinBox()
        self.spinnerEdgeFraction.setSingleStep(0.05)
        self.spinnerEdgeFraction.setMinimum(0.01)
        self.spinnerEdgeFraction.setMaximum(0.99)
        self.spinnerEdgeFraction.setValue(0.10)
        self.spinnerEdgeFraction.setToolTip(edge_fraction_tip)
        self.spinnerEdgeFraction.valueChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.spinnerEdgeFraction
        )

        # ---------------------------------------------------------------------
        # Background suppression. Two orthogonal gates aimed at persistent
        # contaminant signals (like 279 m/z from formic acid)
        # that pass otherwise peak validation:
        #    - a smoothing-survival test (kills spiky/jitter apices)
        #    - a per-lane persistence test
        #           (drops mass lanes present across most of the run).
        #
        # Only used by auto-generation.
        # ---------------------------------------------------------------------
        bg_header = QtWidgets.QLabel("Background suppression:")
        bg_header.setStyleSheet("font-weight: bold;")
        layout.addWidget(bg_header)

        # *** Smoothing-survival gate ***
        # After peak validation, require the apex's rise above baseline to
        # survive a narrow Gaussian smooth. A coherent band survives (~1); a
        # 1-2 scan spike or plateau jitter collapses toward baseline.
        self.checkSmoothingSurvival = QtWidgets.QCheckBox(
            "Require smoothing survival"
        )
        self.checkSmoothingSurvival.setChecked(True)
        self.checkSmoothingSurvival.setToolTip(
            "After peak validation, require the apex's rise above baseline\n"
            "to survive a narrow Gaussian smooth. A coherent band survives\n"
            "(~1); a 1-2 scan spike or plateau jitter collapses toward\n"
            "baseline."
        )
        self.checkSmoothingSurvival.stateChanged.connect(
            self.param_changed
        )
        layout.addWidget(self.checkSmoothingSurvival)

        # *** Smoothing sigma (scans) ***
        # Gaussian width. Keep well below a real peak's width (~1) so genuine
        # bands survive while narrow spikes don't.
        smoothing_sigma_label = QtWidgets.QLabel("Smoothing sigma (scans):")
        smoothing_sigma_tip = (
            "Gaussian width used by the smoothing-survival test. Keep well\n"
            "below a real peak's width (~1 scans) so genuine bands survive while\n"
            "narrow spikes don't."
        )
        smoothing_sigma_label.setToolTip(smoothing_sigma_tip)
        layout.addWidget(smoothing_sigma_label)
        self.spinnerSmoothingSigma = QtWidgets.QDoubleSpinBox()
        self.spinnerSmoothingSigma.setSingleStep(0.5)
        self.spinnerSmoothingSigma.setMinimum(0.1)
        self.spinnerSmoothingSigma.setMaximum(20.0)
        self.spinnerSmoothingSigma.setValue(1.0)
        self.spinnerSmoothingSigma.setToolTip(smoothing_sigma_tip)
        self.spinnerSmoothingSigma.valueChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.spinnerSmoothingSigma
        )

        # *** Min smoothing survival (0-1) ***
        # Fraction of the baseline-subtracted apex height that must remain after
        # smoothing. 0.5 => stays at least halfway between baseline and raw apex.
        min_smoothing_survival_label = QtWidgets.QLabel(
            "Min smoothing survival (0-1):"
        )
        min_smoothing_survival_tip = (
            "Fraction of the baseline-subtracted apex height that must\n"
            "remain after smoothing. 0.5 => stays at least halfway between\n"
            "baseline and raw apex."
        )
        min_smoothing_survival_label.setToolTip(min_smoothing_survival_tip)
        layout.addWidget(min_smoothing_survival_label)
        self.spinnerMinSmoothingSurvival = QtWidgets.QDoubleSpinBox()
        self.spinnerMinSmoothingSurvival.setSingleStep(0.05)
        self.spinnerMinSmoothingSurvival.setMinimum(0.0)
        self.spinnerMinSmoothingSurvival.setMaximum(1.0)
        self.spinnerMinSmoothingSurvival.setValue(0.5)
        self.spinnerMinSmoothingSurvival.setToolTip(min_smoothing_survival_tip)
        self.spinnerMinSmoothingSurvival.valueChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.spinnerMinSmoothingSurvival
        )

        # *** Persistent-lane rejection ***
        # A mass lane present in more than this fraction of the run's scans is
        # treated as background and never seeds OR recruits. Unchecking disables
        # the whole test (passes max_lane_persistence=None).
        self.checkPersistentLanes = QtWidgets.QCheckBox(
            "Reject persistent lanes"
        )
        self.checkPersistentLanes.setChecked(True)
        self.checkPersistentLanes.setToolTip(
            "A mass lane present in more than the fraction below of the\n"
            "run's scans is treated as background and never seeds or\n"
            "recruits into an ensemble."
        )
        self.checkPersistentLanes.stateChanged.connect(
            self.param_changed
        )
        layout.addWidget(self.checkPersistentLanes)

        # *** Max lane persistence (0-1) ***
        max_lane_persistence_label = QtWidgets.QLabel(
            "Max lane persistence (0-1):"
        )
        max_lane_persistence_tip = (
            "Fraction of the run's scans a mass lane can be present in\n"
            "before it's treated as background. Only used when 'Reject\n"
            "persistent lanes' is checked. A lane counts as 'present' in a\n"
            "scan when its intensity is at/above the shared 'Minimum\n"
            "intensity' above."
        )
        max_lane_persistence_label.setToolTip(max_lane_persistence_tip)
        layout.addWidget(max_lane_persistence_label)
        self.spinnerMaxLanePersistence = QtWidgets.QDoubleSpinBox()
        self.spinnerMaxLanePersistence.setSingleStep(0.05)
        self.spinnerMaxLanePersistence.setMinimum(0.01)
        self.spinnerMaxLanePersistence.setMaximum(1.0)
        self.spinnerMaxLanePersistence.setValue(0.5)
        self.spinnerMaxLanePersistence.setToolTip(max_lane_persistence_tip)
        self.spinnerMaxLanePersistence.valueChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.spinnerMaxLanePersistence
        )

        # (A lane counts as "present" in a scan when its intensity is at/above
        # the shared Min intensity — cofeature_threshold — so no separate floor
        # control is needed here.)

        # ---------------------------------------------------------------------
        # RT window. Restrict seeding to a retention-time range; everything
        # outside [start, end] is ignored when picking seeds. Unchecking passes
        # rt_range=None (whole run). Only used by auto-generation. Placed back in
        # the left column to balance the two columns' heights.
        # ---------------------------------------------------------------------
        layout = left_layout

        rt_header = QtWidgets.QLabel("RT window:")
        rt_header.setStyleSheet("font-weight: bold;")
        layout.addWidget(rt_header)

        self.checkRTWindow = QtWidgets.QCheckBox(
            "Restrict to RT window"
        )
        self.checkRTWindow.setChecked(False)
        self.checkRTWindow.setToolTip(
            "Restrict seeding to the retention-time range below; everything\n"
            "outside [start, end] is ignored when picking seeds. Unchecked\n"
            "uses the whole run."
        )
        self.checkRTWindow.stateChanged.connect(
            self.param_changed
        )
        layout.addWidget(self.checkRTWindow)

        rt_start_label = QtWidgets.QLabel("RT start (min):")
        rt_start_tip = "Start of the RT window. Only used when checked above."
        rt_start_label.setToolTip(rt_start_tip)
        layout.addWidget(rt_start_label)
        self.spinnerRTStart = QtWidgets.QDoubleSpinBox()
        self.spinnerRTStart.setSingleStep(0.5)
        self.spinnerRTStart.setMinimum(0.0)
        self.spinnerRTStart.setMaximum(100000.0)
        self.spinnerRTStart.setValue(0.0)
        self.spinnerRTStart.setToolTip(rt_start_tip)
        self.spinnerRTStart.valueChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.spinnerRTStart
        )

        rt_end_label = QtWidgets.QLabel("RT end (min):")
        rt_end_tip = "End of the RT window. Only used when checked above."
        rt_end_label.setToolTip(rt_end_tip)
        layout.addWidget(rt_end_label)
        self.spinnerRTEnd = QtWidgets.QDoubleSpinBox()
        self.spinnerRTEnd.setSingleStep(0.5)
        self.spinnerRTEnd.setMinimum(0.0)
        self.spinnerRTEnd.setMaximum(100000.0)
        self.spinnerRTEnd.setValue(0.0)
        self.spinnerRTEnd.setToolTip(rt_end_tip)
        self.spinnerRTEnd.valueChanged.connect(
            self.param_changed
        )
        layout.addWidget(
            self.spinnerRTEnd
        )

        # ---------------------------------------------------------------------
        # Save / Reset / Restore-defaults for the auto-generation settings,
        # mirroring the Formula Finder dialog. Save persists the current auto
        # params to the [auto_ensemble] config section; Reset reloads the saved
        # config; Restore Defaults reloads the shipped template.
        # ---------------------------------------------------------------------
        self.btnConfigBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Reset
            | QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults
        )
        self.btnConfigBox.clicked.connect(self.on_config_btn_pressed)
        main_layout.addWidget(self.btnConfigBox)

        # Populate the auto-generation controls from the persisted config.
        if self.config is not None:
            self._load_auto_params_from_config()


    def param_changed(self):
        self.sigSettingsChanged.emit(
            self.get_params()
        )


    def get_params(self) -> dict:
        """
        Parameters for *manual* single-ensemble extraction. Kept as the original
        subset so the manual extraction call site is unaffected.
        """
        return {
            'ms1_corr_threshold': float(self.spinnerMS1Threshold.value()),
            'ms2_corr_threshold': float(self.spinnerMS2Threshold.value()),
            'min_intsy': float(self.spinnerMinIntsy.value()),
            'use_rel_intsy': self.checkRelativeIntensity.isChecked(),
            'method': self.comboMethod.currentData(),
        }

    def get_auto_params(self) -> dict:
        """
        Parameters for *automated* ensemble generation, shaped to construct an
        AutoEnsembleParams (fields omitted here fall back to its defaults). The
        shared scoring settings (method, thresholds, min intensity, relative
        intensity) carry over from the manual controls above; the auto-only
        controls supply the rest.
        """
        return {
            # shared with manual extraction
            'ms1_corr_threshold': float(self.spinnerMS1Threshold.value()),
            'ms2_corr_threshold': float(self.spinnerMS2Threshold.value()),
            'cofeature_threshold': float(self.spinnerMinIntsy.value()),
            'use_rel_intsy': self.checkRelativeIntensity.isChecked(),
            'method': self.comboMethod.currentData(),
            # auto-specific
            'parent_threshold': float(self.spinnerParentThreshold.value()),
            'window_strategy': self.comboWindow.currentData(),
            'extraction_half_width': int(self.spinnerHalfWidth.value()),
            'adduct_aware': self.checkAdductAware.isChecked(),
            'loose_corr_threshold': float(self.spinnerLooseThreshold.value()),
            # peak validation (is_peak)
            'peak_method': self.comboPeakMethod.currentData(),
            'min_peak_width': int(self.spinnerMinPeakWidth.value()),
            'min_prominence': float(self.spinnerMinProminence.value()),
            'baseline_pct': float(self.spinnerBaselinePct.value()),
            'edge_fraction': float(self.spinnerEdgeFraction.value()),
            # background suppression
            'require_smoothing_survival': (
                self.checkSmoothingSurvival.isChecked()
            ),
            'smoothing_sigma': float(self.spinnerSmoothingSigma.value()),
            'min_smoothing_survival': (
                float(self.spinnerMinSmoothingSurvival.value())
            ),
            'max_lane_persistence': (
                float(self.spinnerMaxLanePersistence.value())
                if self.checkPersistentLanes.isChecked()
                else None
            ),
            # RT window (None = whole run)
            'rt_range': self._rt_range(),
        }

    def _rt_range(self) -> 'tuple[float, float] | None':
        """
        (start, end) RT window in SECONDS from the controls, or None when the
        restriction is off or the bounds are degenerate (end <= start), in which
        case the whole run is used.

        The spinners are in minutes (chromatograms are shown in minutes), but
        ScanArray.rt_arr — which rt_range is compared against — is in seconds, so
        convert here.
        """
        if not self.checkRTWindow.isChecked():
            return None
        start = float(self.spinnerRTStart.value())
        end = float(self.spinnerRTEnd.value())
        if end <= start:
            return None
        return (start * 60.0, end * 60.0)

    # -- Auto-generation config persistence --------

    def on_config_btn_pressed(
        self,
        button: 'QtWidgets.QAbstractButton',
    ) -> None:
        standard_button = self.btnConfigBox.standardButton(button)
        match standard_button:
            case QtWidgets.QDialogButtonBox.StandardButton.Save:
                self._save_auto_params_to_config()
            case QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults:
                # Populate from the shipped template, ignoring user overrides.
                self._load_auto_params_from_config(load_default_config())
            case QtWidgets.QDialogButtonBox.StandardButton.Reset:
                # Reload the last-saved user config.
                self._load_auto_params_from_config()

    def _load_auto_params_from_config(self, config=None) -> None:
        """
        Populate the auto-generation controls from `config`
        (defaults to the menu's own user config).
        Returns None if no config is available.
        """
        config = config if config is not None else self.config
        if config is None:
            return
        self._apply_auto_params(auto_params_from_config(config))

    def _save_auto_params_to_config(self) -> None:
        """
        Persist the current auto-generation controls to the config file.

        Starts from the stored config so fields the GUI doesn't expose
        (min_turn, min_window_halfwidth, adduct_ppm_tol, polarity) are preserved,
        overlays the GUI-controlled fields, then writes to disk.
        """
        if self.config is None:
            return
        merged = auto_params_from_config(self.config)._replace(
            **self.get_auto_params()
        )
        auto_params_to_config(self.config, merged)
        save_config(self.config)

    def _set_combo(self, combo: 'QtWidgets.QComboBox', data) -> None:
        idx = combo.findData(data)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _apply_auto_params(self, p: 'AutoEnsembleParams') -> None:
        """
        Set every auto-generation control from an AutoEnsembleParams. The
        two None-able composites are decoded to their checkbox + value here.
        """
        # shared scoring controls
        self.spinnerMS1Threshold.setValue(p.ms1_corr_threshold)
        self.spinnerMS2Threshold.setValue(p.ms2_corr_threshold)
        self.spinnerMinIntsy.setValue(p.cofeature_threshold)
        self.checkRelativeIntensity.setChecked(p.use_rel_intsy)
        self._set_combo(self.comboMethod, p.method)
        # auto core
        self.spinnerParentThreshold.setValue(p.parent_threshold)
        self._set_combo(self.comboWindow, p.window_strategy)
        self.spinnerHalfWidth.setValue(p.extraction_half_width)
        self.checkAdductAware.setChecked(p.adduct_aware)
        self.spinnerLooseThreshold.setValue(p.loose_corr_threshold)
        # peak validation
        self._set_combo(self.comboPeakMethod, p.peak_method)
        self.spinnerMinPeakWidth.setValue(p.min_peak_width)
        self.spinnerMinProminence.setValue(p.min_prominence)
        self.spinnerBaselinePct.setValue(p.baseline_pct)
        self.spinnerEdgeFraction.setValue(p.edge_fraction)
        # background suppression
        self.checkSmoothingSurvival.setChecked(p.require_smoothing_survival)
        self.spinnerSmoothingSigma.setValue(p.smoothing_sigma)
        self.spinnerMinSmoothingSurvival.setValue(p.min_smoothing_survival)
        self.checkPersistentLanes.setChecked(p.max_lane_persistence is not None)
        if p.max_lane_persistence is not None:
            self.spinnerMaxLanePersistence.setValue(p.max_lane_persistence)
        # RT window (stored/params in seconds; controls in minutes)
        self.checkRTWindow.setChecked(p.rt_range is not None)
        if p.rt_range is not None:
            self.spinnerRTStart.setValue(p.rt_range[0] / 60.0)
            self.spinnerRTEnd.setValue(p.rt_range[1] / 60.0)