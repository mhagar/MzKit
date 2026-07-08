"""
DDA precursor grouping — thin preset over the unified extraction engine.

The grouping logic now lives in ``core.cli.auto_extract_ensembles`` (shared with
the DIA/auto path). This module keeps the ``GroupingParams`` public surface and
maps it onto an ``ExtractionConfig`` via the ``dda_config`` preset.

Grouping principle (unchanged):
    1. Map each MS2 scan to the tallest MS1 peak in its isolation window, then
       cluster those into per-peak precursor features.
    2. Greedily seed on the tallest precursor feature; recruit coeluting
       precursor features by peak-shape cosine of their MS1 XICs, loosening the
       threshold when the seed<->candidate delta-m/z matches a known adduct /
       isotope / neutral-loss relationship.
    3. Emit an Ensemble per multi-MS2 group (>= min_ms2_scans unique MS2 scans).
"""
from typing import NamedTuple, TYPE_CHECKING

from core.cli.auto_extract_ensembles import extract_ensembles, dda_config

if TYPE_CHECKING:
    from core.data_structs import Ensemble, Injection


class GroupingParams(NamedTuple):
    """
    Parameters for precursor-feature grouping.

    min_intsy: Minimum intensity for a signal to count (peak in the isolation
        window, and MS2 lane activity).
    edge_fraction: Stop descending the seed peak when intensity falls below this
        fraction of apex (defines the correlation window). Passed to
        `find_peak_boundaries`.
    min_peak_width: Minimum seed-peak width in scans (rejects spikes).
    min_prominence: Seed-peak bilateral-prominence threshold — the peakiness knob
        that rejects slopes and smeared/constant background signals. 0.5 => apex
        must rise to >= 2x the surrounding baseline on its weaker side.
    min_iso_halfwidth: isolation window used when not specified in scan (in Da).
    tight_cosine: Default membership threshold (no recognized adduct/ISF delta m/z).
    loose_cosine: Membership threshold when the delta m/z is recognized (i.e. Na-H).
    adduct_ppm_tol: ppm tolerance for the delta m/z checking.
    use_rel_intsy: Normalize each XIC by its max before scoring.
    polarity: 1 positive, 0 negative. Default 1. Used by adduct table.
    """
    min_intsy: float
    edge_fraction: float = 0.1
    min_peak_width: int = 5
    min_prominence: float = 0.5
    min_iso_halfwidth: float = 0.5
    tight_cosine: float = 0.97
    loose_cosine: float = 0.85
    adduct_ppm_tol: float = 20.0
    use_rel_intsy: bool = True
    polarity: int = 1


def group_precursor_ensembles(
    injection: 'Injection',
    params: GroupingParams,
    progress_callback=None,  # injected by ProcessRunner; unused here
    cancel_event=None,       # injected by ProcessRunner
) -> list['Ensemble']:
    """
    Group precursor features originating from the same chemical entity into
    Ensembles (singletons left unassigned). Delegates to the unified engine.
    """
    config = dda_config(
        min_intsy=params.min_intsy,
        edge_fraction=params.edge_fraction,
        min_peak_width=params.min_peak_width,
        min_prominence=params.min_prominence,
        min_iso_halfwidth=params.min_iso_halfwidth,
        ms1_corr_threshold=params.tight_cosine,
        loose_corr_threshold=params.loose_cosine,
        adduct_ppm_tol=params.adduct_ppm_tol,
        use_rel_intsy=params.use_rel_intsy,
        polarity=params.polarity,
    )
    return extract_ensembles(
        injection, config,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
