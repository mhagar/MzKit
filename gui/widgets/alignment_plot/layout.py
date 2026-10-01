"""
Pure geometry / lookup helpers for the Alignment Viewer feature-map.

No Qt widgets here — just coordinate math, colour assignment and the
alignment -> ensemble resolution logic (lifted from the old
``AlignmentTableModel._get_ensemble``). Kept Qt-light so the numbers are
easy to reason about and tweak.
"""
from typing import Callable, NamedTuple, Optional, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF, QPointF
from PyQt5.QtGui import QColor

from core.utils.natural_sort import natural_sort_key

if TYPE_CHECKING:
    from core.data_structs import SampleUUID, Ensemble
    from core.interfaces.data_sources import SampleDataSource
    from core.data_structs.alignment import EnsembleAlignment, AlignedAnalyte
    from gui.widgets.alignment_plot.params import RenderParams


# Maps an ensemble to its x-coordinate on the feature map.
XAccessor = Callable[['Ensemble'], float]


def ensemble_rt(ensemble: 'Ensemble') -> float:
    """Default x-coordinate: the ensemble's peak retention time (s)."""
    return float(ensemble.peak_rt)


# Colour for singleton analytes (present in a single sample). All other
# tunable rendering values now live in RenderParams (see params.py).
SINGLETON_COLOR = QColor(140, 140, 140, 180)


class LaneMap(NamedTuple):
    """Ordered sample lanes plus their y-position + name lookups."""
    sample_uuids: list['SampleUUID']          # top-to-bottom lane order
    y_of_sample: dict['SampleUUID', float]    # sample -> lane y
    name_of_sample: dict['SampleUUID', str]


class HoverTarget(NamedTuple):
    """
    What the cursor is currently over. ``kind`` is one of
    ``'ensemble'``, ``'analyte'`` or ``'sample'``.
    """
    kind: str
    analyte_id: Optional[int] = None
    sample_uuid: Optional['SampleUUID'] = None
    ensemble: Optional['Ensemble'] = None
    analyte: Optional['AlignedAnalyte'] = None


def build_lane_map(
    alignment: 'EnsembleAlignment',
    data_source: 'SampleDataSource',
) -> LaneMap:
    """
    Resolve sample names and lay the lanes out top-to-bottom, natural-sorted
    by sample name (falls back to a uuid-tail label for missing samples).
    """
    names: dict['SampleUUID', str] = {}
    for uuid in alignment.sample_uuids:
        sample = data_source.get_sample(uuid)
        names[uuid] = sample.name if sample else f"...{str(uuid)[-5:]}"

    ordered = sorted(
        alignment.sample_uuids,
        key=lambda u: natural_sort_key(names[u]),
    )
    y_of_sample = {uuid: float(i) for i, uuid in enumerate(ordered)}
    return LaneMap(list(ordered), y_of_sample, names)


def resolve_ensemble(
    analyte: 'AlignedAnalyte',
    sample_uuid: 'SampleUUID',
    data_source: 'SampleDataSource',
) -> Optional['Ensemble']:
    """
    Look up the concrete Ensemble an analyte points at in a given sample
    (see `AlignedAnalyte.resolve_member`).
    """
    return analyte.resolve_member(sample_uuid, data_source.get_sample)


def _interp_log(
    value: float,
    floor: float,
    ceiling: float,
    out_lo: float,
    out_hi: float,
) -> float:
    """
    Log-interpolate ``value`` from the [floor, ceiling] intensity window onto
    [out_lo, out_hi], clipping outside the window.
    """
    floor = max(floor, 1.0)
    ceiling = max(ceiling, floor + 1.0)
    v = np.log10(max(value, 1.0))
    return float(np.interp(
        v,
        [np.log10(floor), np.log10(ceiling)],
        [out_lo, out_hi],
    ))


def scale_width(base_intsy: float, params: 'RenderParams') -> float:
    """Log-interpolate intensity -> strip width (seconds), clipped."""
    return _interp_log(
        base_intsy, params.intsy_floor, params.intsy_ceiling,
        params.width_min, params.width_max,
    )


def scale_height(base_intsy: float, params: 'RenderParams') -> float:
    """Log-interpolate intensity -> strip height (lane units), clipped."""
    return _interp_log(
        base_intsy, params.intsy_floor, params.intsy_ceiling,
        params.height_min, params.height_max,
    )


def scale_line_opacity(max_intsy: float, params: 'RenderParams') -> int:
    """Log-interpolate an analyte's max intensity -> line alpha (0..255)."""
    return int(round(_interp_log(
        max_intsy, params.intsy_floor, params.intsy_ceiling,
        params.line_opacity_min, params.line_opacity_max,
    )))


def strip_geometry(
    ensemble: 'Ensemble',
    lane_y: float,
    stagger_idx: int,
    params: 'RenderParams',
    x_of: XAccessor = ensemble_rt,
) -> tuple[QRectF, QPointF]:
    """
    Build the strip rect + its centre point for one ensemble.

    X-centre = `x_of(ensemble)` (peak RT by default); both width and height scale with intensity (on their
    own ranges). The strip is nudged vertically within its lane by
    ``stagger_idx`` so it stays clickable.
    """
    width = scale_width(ensemble.base_intsy, params)
    height = scale_height(ensemble.base_intsy, params)
    cx = x_of(ensemble)
    cy = (
        lane_y + params.stagger_base
        + (stagger_idx % params.stagger_count) * params.stagger_step
    )

    rect = QRectF(
        cx - 0.5 * width,
        cy - 0.5 * height,
        width,
        height,
    )
    return rect, QPointF(cx, cy)


def analyte_color(analyte_idx: int, is_singleton: bool) -> QColor:
    """
    Neutral grey for singletons; a stable, distinct colour per multi-sample
    analyte (keyed by its index in the alignment).
    """
    if is_singleton:
        return QColor(SINGLETON_COLOR)
    return pg.intColor(
        analyte_idx,
        hues=17,
        minValue=180,
        maxValue=255,
        sat=170,
    )
