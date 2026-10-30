"""
Reusable 2D structure display.

- StructureView: a SMILES drawn at the widget's own (thumbnail) size
- StructureDelegate: the same, in a table column (SMILES in Qt.UserRole,
  optional name in NAME_ROLE)

Thumbnails skip atom labels and enlarge (with labels) in a shared popup on
hover, and copy the SMILES / name / image from a right-click menu
(StructureView also has a corner button for it).
`[structures]` config: follow_palette (RDKit dark mode on dark palettes),
popup_width / popup_height.
"""
from functools import lru_cache
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtSvg import QSvgRenderer

from core.utils.config import load_config
from core.utils.render_molecule import smiles_to_svg

NAME_ROLE = QtCore.Qt.UserRole + 10
SECTION = 'structures'


@lru_cache(maxsize=1)
def _settings() -> tuple[bool, QtCore.QSize]:
    config = load_config()
    return (
        config.getboolean(SECTION, 'follow_palette', fallback=True),
        QtCore.QSize(config.getint(SECTION, 'popup_width', fallback=400),
                     config.getint(SECTION, 'popup_height', fallback=300)),
    )


def _dark() -> bool:
    follow, _ = _settings()
    palette = QtWidgets.QApplication.palette()
    return follow and palette.color(QtGui.QPalette.Window).lightness() < 128


@lru_cache(maxsize=512)
def _svg(smiles: str, atoms: tuple, width: int, height: int, dark: bool, labels: bool) -> Optional[bytes]:
    try:
        svg, _ = smiles_to_svg(smiles, list(atoms) or None, size=(width, height), dark=dark, labels=labels)
    except ValueError:
        return None
    return svg.encode()


def render(
        painter: QtGui.QPainter, rect: QtCore.QRect, smiles: str, atoms: tuple = (), labels: bool = True,
) -> bool:
    """Draw `smiles` into `rect` at its pixel size; False if it can't be parsed."""
    if rect.width() < 8 or rect.height() < 8:
        return False
    svg = _svg(smiles, tuple(atoms), rect.width(), rect.height(), _dark(), labels)
    if svg is None:
        return False
    QSvgRenderer(QtCore.QByteArray(svg)).render(painter, QtCore.QRectF(rect))
    return True


def _image(smiles: str, atoms: tuple, size: QtCore.QSize) -> QtGui.QImage:
    image = QtGui.QImage(size, QtGui.QImage.Format_ARGB32)
    image.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(image)
    render(painter, image.rect(), smiles, atoms)
    painter.end()
    return image


class _Popup(QtWidgets.QWidget):
    """The shared enlarged rendering shown beside the cursor."""

    def __init__(self):
        super().__init__(None, QtCore.Qt.ToolTip | QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._smiles, self._atoms = "", ()

    def show_at(self, global_pos: QtCore.QPoint, smiles: str, atoms: tuple = ()):
        _, size = _settings()
        if (smiles, atoms) != (self._smiles, self._atoms):
            self._smiles, self._atoms = smiles, atoms
            self.update()
        self.resize(size)
        # Below-right of the cursor, flipped to stay on screen
        screen = QtWidgets.QApplication.screenAt(global_pos)
        area = screen.availableGeometry() if screen else QtCore.QRect(global_pos, size)
        x, y = global_pos.x() + 16, global_pos.y() + 16
        if x + size.width() > area.right():
            x = global_pos.x() - 16 - size.width()
        if y + size.height() > area.bottom():
            y = global_pos.y() - 16 - size.height()
        self.move(max(x, area.left()), max(y, area.top()))
        self.show()
        self.raise_()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtCore.Qt.black if _dark() else QtCore.Qt.white)
        render(painter, self.rect(), self._smiles, self._atoms)
        painter.setPen(self.palette().color(QtGui.QPalette.Mid))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


_popup: Optional[_Popup] = None


def popup() -> _Popup:
    global _popup
    if _popup is None:
        _popup = _Popup()
    return _popup


def copy_menu(parent: QtWidgets.QWidget, smiles: str, name: str = "", atoms: tuple = ()) -> QtWidgets.QMenu:
    """Copy SMILES / name / image (PNG + SVG) of a structure."""
    clipboard = QtWidgets.QApplication.clipboard()
    menu = QtWidgets.QMenu(parent)
    menu.addAction("Copy SMILES", lambda: clipboard.setText(smiles))
    menu.addAction("Copy name", lambda: clipboard.setText(name)).setEnabled(bool(name))

    def copy_image():
        _, size = _settings()
        mime = QtCore.QMimeData()
        mime.setImageData(_image(smiles, atoms, size))
        svg = _svg(smiles, tuple(atoms), size.width(), size.height(), _dark(), True)
        if svg:
            mime.setData('image/svg+xml', svg)
        clipboard.setMimeData(mime)

    menu.addAction("Copy image", copy_image)
    return menu


class StructureView(QtWidgets.QWidget):
    """A structure thumbnail: hover enlarges it, right-click / corner button copies."""

    def __init__(self, parent=None, size: tuple[int, int] = (160, 120)):
        super().__init__(parent)
        self._size_hint = QtCore.QSize(*size)
        self._smiles, self._name, self._atoms = "", "", ()
        self._caption = ""
        self.enlarge_on_hover = True
        self.setMouseTracking(True)
        self.copy_btn = QtWidgets.QToolButton(self)
        self.copy_btn.setText("⧉")
        self.copy_btn.setToolTip("Copy SMILES, name or image")
        self.copy_btn.setAutoRaise(True)
        self.copy_btn.clicked.connect(
            lambda: self._menu(self.copy_btn.mapToGlobal(self.copy_btn.rect().bottomLeft()))
        )
        self.copy_btn.hide()

    def set_structure(self, smiles: str, name: str = "", highlight_atoms=None):
        self._smiles, self._name = smiles or "", name or ""
        self._atoms = tuple(highlight_atoms or ())
        if self.underMouse() and popup().isVisible():
            popup().show_at(QtGui.QCursor.pos(), self._smiles, self._atoms)
        self.update()

    def set_caption(self, text: str):
        """Text (may be multi-line) drawn in the top-right corner; "" clears it."""
        self._caption = text
        self.update()

    def smiles(self) -> str:
        return self._smiles

    def sizeHint(self):
        return self._size_hint

    def minimumSizeHint(self):
        return QtCore.QSize(40, 30)

    def paintEvent(self, event):
        if self._smiles:
            painter = QtGui.QPainter(self)
            # A thumbnail that enlarges skips atom labels: they're legible in the popup
            render(painter, self.rect(), self._smiles, self._atoms, labels=not self.enlarge_on_hover)
            if self._caption:
                painter.setPen(QtCore.Qt.white if _dark() else QtCore.Qt.black)
                painter.drawText(self.rect().adjusted(3, 3, -3, -3),
                                 QtCore.Qt.AlignTop | QtCore.Qt.AlignRight, self._caption)
            painter.end()

    def resizeEvent(self, event):
        self.copy_btn.move(self.width() - self.copy_btn.sizeHint().width(), 0)

    def _enlarges(self) -> bool:
        _, size = _settings()
        return self.enlarge_on_hover and bool(self._smiles) and (self.width() < size.width() * 0.8 or self.height() < size.height() * 0.8)

    def enterEvent(self, event):
        self.copy_btn.setVisible(bool(self._smiles))

    def mouseMoveEvent(self, event):
        if self._enlarges():
            popup().show_at(event.globalPos(), self._smiles, self._atoms)

    def leaveEvent(self, event):
        self.copy_btn.hide()
        popup().hide()

    def contextMenuEvent(self, event):
        self._menu(event.globalPos())

    def _menu(self, global_pos):
        if self._smiles:
            popup().hide()
            copy_menu(self, self._smiles, self._name, self._atoms).exec_(global_pos)


class StructureDelegate(QtWidgets.QStyledItemDelegate):
    """
    Draws the SMILES in the cell's Qt.UserRole; only painted (visible) rows get
    rendered. Hovering a cell enlarges it, right-clicking any cell of the row
    offers copy, and Ctrl+C on the structure cell copies its SMILES.
    `extend_menu(menu, index)` adds the view's own actions to that menu.
    """

    def __init__(self, view: QtWidgets.QAbstractItemView, extend_menu=None):
        super().__init__(view)
        self._view, self._viewport = view, view.viewport()
        self._extend_menu = extend_menu
        view.setMouseTracking(True)
        self._viewport.installEventFilter(self)
        view.installEventFilter(self)

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        if smiles := index.data(QtCore.Qt.UserRole):
            render(painter, option.rect.adjusted(2, 2, -2, -2), smiles, labels=False)

    def _structure_index(self, index: QtCore.QModelIndex) -> QtCore.QModelIndex:
        """The structure cell in `index`'s row."""
        if not index.isValid():
            return index
        for col in range(index.model().columnCount()):
            if self._view.itemDelegateForColumn(col) is self:
                return index.sibling(index.row(), col)
        return QtCore.QModelIndex()

    def eventFilter(self, obj, event):
        kind = event.type()
        if obj is self._viewport:
            if kind == QtCore.QEvent.MouseMove:
                index = self._view.indexAt(event.pos())
                smiles = index.data(QtCore.Qt.UserRole) if self._view.itemDelegate(index) is self else None
                if smiles:
                    popup().show_at(event.globalPos(), smiles)
                else:
                    popup().hide()
            elif kind == QtCore.QEvent.Leave:
                popup().hide()
            elif kind == QtCore.QEvent.ContextMenu:
                index = self._view.indexAt(event.pos())
                cell = self._structure_index(index)
                if smiles := cell.data(QtCore.Qt.UserRole):
                    popup().hide()
                    menu = copy_menu(self._view, smiles, cell.data(NAME_ROLE) or "")
                    if self._extend_menu is not None:
                        self._extend_menu(menu, index)
                    menu.exec_(event.globalPos())
                    return True
        elif kind == QtCore.QEvent.KeyPress and event.matches(QtGui.QKeySequence.Copy):
            index = self._view.currentIndex()
            if self._view.itemDelegate(index) is self and (smiles := index.data(QtCore.Qt.UserRole)):
                QtWidgets.QApplication.clipboard().setText(smiles)
                return True
        return False
