# from core.controllers.MainController import MainController
from gui.controllers import MainController
from core.utils.config import load_config

from PyQt5.QtWidgets import QApplication

import sys

print(
        f"Hi. __name__: {__name__}"
)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    config = load_config()

    controller = MainController(
        app=app,
        config=config,
    )

    controller.main_view.show()

    sys.exit(app.exec_())
