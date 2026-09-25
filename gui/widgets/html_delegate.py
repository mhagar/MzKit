"""
Table item delegate that renders cell text as HTML (e.g. formula subscripts).
"""
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QTextDocument


class HTMLDelegate(QtWidgets.QStyledItemDelegate):
    """
    Delegate that renders HTML content in table cells.
    Useful for displaying chemical formulae with subscripts.
    """

    def paint(self, painter, option, index):
        """Paint the cell with HTML rendering"""
        self.initStyleOption(option, index)

        # QTextDocument for rendering HTML
        doc = QTextDocument()
        doc.setHtml(option.text)
        doc.setTextWidth(option.rect.width())

        # Clear the text from the option to prevent default drawing
        option.text = ""

        # Draw the background and focus rect
        style = option.widget.style() if option.widget else QtWidgets.QStyle()
        style.drawControl(QtWidgets.QStyle.CE_ItemViewItem, option, painter, option.widget)

        # Draw the HTML content
        painter.save()
        painter.translate(option.rect.topLeft())
        doc.drawContents(painter)
        painter.restore()

    def sizeHint(self, option, index):
        """
        Calculate the size hint for the cell
        """
        doc = QTextDocument()
        doc.setHtml(index.data())
        doc.setTextWidth(option.rect.width() if option.rect.width() > 0 else 100)
        return QSize(int(doc.idealWidth()), int(doc.size().height()))
