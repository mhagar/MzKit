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
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)

    config = load_config()

    controller = MainController(
        app=app,
        config=config,
    )

    controller.main_view.show()

    sys.exit(app.exec_())