# Copyright (C) 2026  Mostafa Hagar
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


import sys
import os

# Force X11 on Linux (Wayland causes issues) - must be set before importing Qt
if sys.platform == 'linux':
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

from gui.controllers import MainController
from core.utils.config import load_config
from PyQt5.QtWidgets import QApplication, QProxyStyle, QStyle
import pyqtgraph as pg

COMPACT_QSS = """
QTableView::item { padding: 0px 3px; }
QHeaderView::section { padding: 1px 4px; }
QPushButton, QToolButton { padding: 2px 6px; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { padding: 1px 3px; min-height: 0; }
QTabBar::tab { padding: 3px 8px; }
QGroupBox { margin-top: 6px; }
"""


class CompactStyle(QProxyStyle):
    def pixelMetric(self, metric, option=None, widget=None):
        # Qt clamps this up to the header's minimum (font height + padding), so rows hug the text
        if metric == QStyle.PM_HeaderDefaultSectionSizeVertical:
            return 0
        return super().pixelMetric(metric, option, widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Dark mode (qdarkstyle) disabled for now in favour of a compact layout
    app.setStyle(CompactStyle(app.style()))
    font = app.font()
    font.setPointSizeF(font.pointSizeF() - 1)
    app.setFont(font)
    app.setStyleSheet(COMPACT_QSS)

    # pg.setConfigOption('background', 'w')
    # pg.setConfigOption('foreground', 'k')

    config = load_config()

    controller = MainController(
        app=app,
        config=config,
    )

    controller.main_view.show()

    sys.exit(app.exec_())