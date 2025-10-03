from PyQt5 import QtWidgets, QtCore

from gui.resources.FormulaFinderWindow import Ui_Form


class FormulaFinderDialog(
    QtWidgets.QWidget,
    Ui_Form,
):

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self._connect_signals()
        self._setup_statusbar()

        # State stuff
        self.search_query: list[tuple[float, float]] = []

    def _connect_signals(self):
        self.btnAddSignal.clicked.connect(
            self.tableInput.add_row
        )

        self.btnRemoveSignal.clicked.connect(
            self.tableInput.remove_last_row
        )

        self.btnClearSignals.clicked.connect(
            self.tableInput.clear_rows
        )

        self.btnSearch.clicked.connect(
            self.on_search_execute
        )

    def _setup_statusbar(self):
        self.statusbar = QtWidgets.QStatusBar()
        # self.statusbar.setMaximumHeight(15)  # pixels
        self.verticalLayout.addWidget(
            self.statusbar
        )


    def on_search_execute(self):
        self._retrieve_table_input()

    def _retrieve_table_input(self):
        """
        Retrieves the user input from the table
        """
        user_input = self.tableInput.get_user_input()
        print(
            f"user_input: {user_input}"
        )

        if user_input:
            self.search_query = user_input

        else:
            # Something went wrong; show warning
            self.statusbar.showMessage(
                "Invalid search query"
            )

            return






