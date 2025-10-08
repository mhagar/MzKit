import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui

from typing import Optional

class TextOverlay(QtWidgets.QGraphicsWidget):
    """
    A widget that displays text and stays in view when plot is panned/scaled.

    Note! To use this, don't .addItem(); rather, should be like this:
    ```
    text_overlay = TextOverlay(text="Hello, Text Overlay", offset=(50, 0))
    text_overlay.setParentItem(plot_item)
    ```
    """
    def __init__(
            self,
            text: str,
            offset: Optional[tuple[int, int]] = None,
    ):
        super().__init__()
        self.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations)

        # Create layout and text label
        self.layout = QtWidgets.QGraphicsGridLayout()
        self.setLayout(self.layout)
        # self.label = pg.LabelItem(text)
        self.label = TextOverlayItem(text, parent=self)
        self.layout.addItem(self.label, 0, 0)

        # Set position relative to parent if offset provided
        if offset is not None:
            self.setPos(offset[0], offset[1])

    def boundingRect(self):
        return QtCore.QRectF(
            0,
            0,
            self.label.boundingRect().width(),
            self.label.boundingRect().height(),
        )

    def setText(
            self,
            text: str
    ):
        """
        Update the displayed text
        """
        self.label.setText(text)

    def setTextColor(
            self,
            color
    ):
        """
        Set the text color
        """
        self.label.setDefaultTextColor(
            QtGui.QColor(color)
        )


class TextOverlayItem(pg.GraphicsWidget):
    def __init__(
            self,
            text: str,
            parent=None,
            angle=0,
    ):
        pg.GraphicsWidget.__init__(
            self,
            parent
        )
        self.item = QtWidgets.QGraphicsTextItem(text, self)
        self.item.setRotation(angle)

    def setText(self, text):
        self.item.setHtml(
            text
        )

    def setTextColor(self, color):
        self.item.setDefaultTextColor(QtGui.QColor(color))

    def boundingRect(self):
        return self.item.boundingRect()