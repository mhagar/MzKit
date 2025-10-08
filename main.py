from gui.controllers import MainController
from core.utils.config import load_config

from PyQt5.QtWidgets import QApplication

import sys
import os

# Force X11 (Wayland is a problem)
os.environ['QT_QPA_PLATFORM'] = 'xcb'

if __name__ == "__main__":
    app = QApplication(sys.argv)

    config = load_config()

    controller = MainController(
        app=app,
        config=config,
    )

    controller.main_view.show()

    sys.exit(app.exec_())