from PyQt5.QtWidgets import QMdiArea, QMdiSubWindow
from PyQt5 import QtCore

class AppMdiArea(QMdiArea):
    """
    Custom MDI area for application
    """
    def __init__(
            self,
            parent=None,
    ):
        super().__init__(parent)
        self.setActivationOrder(
            QMdiArea.ActivationHistoryOrder
        )

    def add_sub_window(
            self,
            widget,
            title=""
    ) -> QMdiSubWindow:
        """
        Add a new subwindow containing the widget
        :param widget:
        :param title:
        :return: returns QMdiSubWindow
        """
        sub_window = QMdiSubWindow()
        sub_window.setWidget(widget)
        sub_window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        sub_window.setWindowTitle(title)

        self.addSubWindow(sub_window)
        sub_window.show()

        return sub_window
