
from gui.widgets.ChromPlotWidget import ChromPlotWidget, ChromPlotItem, ChromViewBox

from gui.widgets.CustomAxisItems import IntsyAxisItem


class InteractiveChromPlotWidget(ChromPlotWidget):
    """
    An interactive version of the ChromPlotWidget that has interactivity
    - A slider fo selecting specific scans
    - A range/window slider for selecting a range of scans
    """
    def __init__(self, *args, **kwargs):
        super(ChromPlotWidget, self).__init__(
            *args,
            **kwargs,
            plotItem=InteractiveChromPlotItem(
                plot_widget=self,
                axisItems={
                    'left': IntsyAxisItem('left')
                },
            ),
        )
        self.pi: InteractiveChromPlotItem = self.getPlotItem()
        self.setBackground(None)

class InteractiveChromPlotItem(ChromPlotItem):
    """
    An interactive version of ChromPlotItem with interactivity features:
    - A slider fo selecting specific scans
    - A range/window slider for selecting a range of scans
    """
    def __init__(
        self,
        plot_widget: InteractiveChromPlotWidget,
        *args,
        **kwargs,
    ):
        super(ChromPlotItem, self).__init__(
            *args,
            **kwargs,
            viewBox=ChromViewBox(
                defaultPadding=0.0,
            ),
        ),
        self.plot_widget = plot_widget



