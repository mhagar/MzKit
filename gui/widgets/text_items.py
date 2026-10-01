"""
Helpers for pyqtgraph TextItems shared between plot widgets.
"""
import pyqtgraph as pg
from PyQt5 import QtCore


def center_textitem(
        textitem: pg.TextItem,
) -> pg.TextItem:
    """
    Bizarre hoops that must be jumped to center-align HTML labels
    in pyqtgraph.
    From https://stackoverflow.com/a/62602065
    """
    it = textitem.textItem
    option = it.document().defaultTextOption()
    option.setAlignment(QtCore.Qt.AlignCenter)
    it.document().setDefaultTextOption(option)
    it.setTextWidth(it.boundingRect().width())

    return textitem
