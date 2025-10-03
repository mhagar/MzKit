import numpy as np
import pyqtgraph as pg


class IntsyAxisItem(pg.AxisItem):
    """
    Custom AxisItem with sensible behaviour for mass spectra
    (Y-axis)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_ticks_per_plot = 10
        self.min_ticks_per_plot = 5

    def tickStrings(self, values, scale, spacing):
        return [f'{value:.1E}' for value in values]

    def tickValues(self, minVal, maxVal, size):
        # Get the original tick values
        original_ticks = super().tickValues(minVal, maxVal, size)
        if not original_ticks:
            return original_ticks

        # Get the major ticks (first element in the dictionary)
        major_ticks = original_ticks[0][1]
        num_ticks = len(major_ticks)

        if num_ticks == 1:
            return original_ticks

        if num_ticks > self.max_ticks_per_plot:
            # If too many ticks, reduce them
            step = len(major_ticks) // self.max_ticks_per_plot
            major_ticks = major_ticks[::step]
        elif num_ticks < self.min_ticks_per_plot:
            # If too few ticks, interpolate to add more
            current_range = major_ticks[-1] - major_ticks[0]
            desired_step = current_range / (self.min_ticks_per_plot - 1)
            new_ticks = np.arange(
                major_ticks[0],
                major_ticks[-1] + desired_step / 2,  # Add small offset to include endpoint
                desired_step
            )
            major_ticks = new_ticks

        return [(original_ticks[0][0], major_ticks)]


class MzAxisItem(pg.AxisItem):
    """
    Custom AxisItem implementing sensible behaviour for mass spectra
    (X-axis)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_ticks_per_plot = 100
        self.min_ticks_per_plot = 10

    def tickValues(self, minVal, maxVal, size):
        # Get the original tick values
        original_ticks = super().tickValues(minVal, maxVal, size)
        if not original_ticks:
            return original_ticks

        # Get the major ticks (first element in the dictionary)
        major_ticks = original_ticks[0][1]
        num_ticks = len(major_ticks)

        if num_ticks == 1:
            return original_ticks

        if num_ticks > self.max_ticks_per_plot:
            # If too many ticks, reduce them
            step = len(major_ticks) // self.max_ticks_per_plot
            major_ticks = major_ticks[::step]
        elif num_ticks < self.min_ticks_per_plot:
            # If too few ticks, interpolate to add more
            current_range = major_ticks[-1] - major_ticks[0]
            desired_step = current_range / (self.min_ticks_per_plot - 1)
            new_ticks = np.arange(
                major_ticks[0],
                major_ticks[-1] + desired_step / 2,  # Add small offset to include endpoint
                desired_step
            )
            major_ticks = new_ticks

        return [(original_ticks[0][0], major_ticks)]
