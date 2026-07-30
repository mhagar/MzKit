"""
User-tunable rendering parameters for the Alignment Viewer feature-map.

These used to be module-level constants in ``layout.py``; they're now a
dataclass so the visualization menu can edit them and re-render. A single
intensity reference window (``intsy_floor`` .. ``intsy_ceiling``, log-space)
is mapped onto every intensity-driven visual extent (strip width, strip
height, connecting-line opacity).
"""
from dataclasses import dataclass, replace


@dataclass
class RenderParams:
    # Intensity reference window (linear values; mapped in log space).
    intsy_floor: float = 1e4
    intsy_ceiling: float = 5e7

    # Strip width (retention-time seconds).
    width_min: float = 0.1
    width_max: float = 45.0

    # Strip height (lane units; 1.0 == one whole sample lane).
    height_min: float = 0.04
    height_max: float = 0.10

    # Vertical stagger within a lane (keeps overlapping strips clickable).
    stagger_count: int = 7
    stagger_step: float = 0.13

    # Analyte connecting-line opacity (0..255), scaled by the analyte's
    # max ensemble intensity.
    line_opacity_min: int = 40
    line_opacity_max: int = 220

    @property
    def stagger_base(self) -> float:
        """Vertical offset that centres the stagger spread on the lane."""
        return -0.5 * (self.stagger_count - 1) * self.stagger_step

    def copy(self) -> "RenderParams":
        return replace(self)
