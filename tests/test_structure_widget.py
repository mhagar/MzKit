"""Offscreen checks of the reusable structure thumbnail (gui/widgets/StructureWidget.py)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
from gui.widgets import StructureWidget  # noqa: E402

ERYTHROMYCIN = (
    'CC[C@@H]1[C@@]([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)'
    'O[C@H]2C[C@@]([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)O)C)C)O)(C)O'
)


def test_size_hint_ignores_the_molecule():
    view = StructureWidget.StructureView(size=(160, 120))
    view.set_structure(ERYTHROMYCIN, "erythromycin")
    assert view.sizeHint() == QtCore.QSize(160, 120)
    image = QtGui.QImage(160, 120, QtGui.QImage.Format_ARGB32)
    image.fill(QtCore.Qt.transparent)
    view.render(image)
    drawn = sum(image.pixelColor(x, y).alpha() > 0 for x in range(0, 160, 4) for y in range(0, 120, 4))
    assert drawn > 0


def test_hover_shows_and_hides_popup():
    view = StructureWidget.StructureView()
    view.set_structure("CCO")
    view.resize(160, 120)
    view.show()
    QtWidgets.QApplication.sendEvent(view, QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove, QtCore.QPointF(10, 10), QtCore.Qt.NoButton,
        QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
    ))
    assert StructureWidget.popup().isVisible()
    QtWidgets.QApplication.sendEvent(view, QtCore.QEvent(QtCore.QEvent.Leave))
    assert not StructureWidget.popup().isVisible()


def test_copy_menu_fills_clipboard():
    clipboard = QtWidgets.QApplication.clipboard()
    menu = StructureWidget.copy_menu(None, "CCO", "ethanol")
    smiles, name, image = menu.actions()
    smiles.trigger()
    assert clipboard.text() == "CCO"
    name.trigger()
    assert clipboard.text() == "ethanol"
    image.trigger()
    mime = clipboard.mimeData()
    assert mime.hasImage() and b"<svg" in bytes(mime.data('image/svg+xml'))


def test_delegate_hover_and_ctrl_c():
    table = QtWidgets.QTableWidget(1, 2)
    table.setItemDelegateForColumn(0, StructureWidget.StructureDelegate(table))
    item = QtWidgets.QTableWidgetItem()
    item.setData(QtCore.Qt.UserRole, "CCO")
    table.setItem(0, 0, item)
    table.setItem(0, 1, QtWidgets.QTableWidgetItem("ethanol"))
    table.resize(300, 200)
    table.show()
    center = table.visualItemRect(item).center()
    QtWidgets.QApplication.sendEvent(table.viewport(), QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove, QtCore.QPointF(center), QtCore.Qt.NoButton,
        QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
    ))
    assert StructureWidget.popup().isVisible()
    QtWidgets.QApplication.sendEvent(table.viewport(), QtCore.QEvent(QtCore.QEvent.Leave))
    assert not StructureWidget.popup().isVisible()

    table.setCurrentCell(0, 0)
    QtWidgets.QApplication.sendEvent(table, QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress, QtCore.Qt.Key_C, QtCore.Qt.ControlModifier, "c",
    ))
    assert QtWidgets.QApplication.clipboard().text() == "CCO"
