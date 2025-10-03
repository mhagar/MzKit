from PyQt5 import QtWidgets, QtCore, QtGui
import pandas as pd
import logging

logger = logging.getLogger()


class FormulaFinderInputTable(QtWidgets.QTableWidget):
    # Signal for programmatic data updates
    sigDataUpdated = QtCore.pyqtSignal(list)

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent)
        self._setup_table()
        self._connect_signals()

    def _setup_table(self):
        """
        Configure table appearance
        """
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(
            ["m/z", "Intensity"]
        )

        # Hide vertical header (row numbers)
        self.verticalHeader().setVisible(False)

        # Initialize with single row
        self.setRowCount(1)
        self.add_empty_row(0)
        self._update_intensity_column_state()

        # Make table fill available space
        self.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )

        # Show grid lines and banded rows
        self.setStyleSheet(
            "gridline-color: gray;"
        )

        self.setAlternatingRowColors(True)

    def add_empty_row(
        self,
        idx: int,
    ):
        """
        Add empty items to a row
        """
        for col in range(2):
            item = QtWidgets.QTableWidgetItem("")
            self.setItem(idx, col, item)

    def add_row(self):
        """
        Adds a new row for another MS signal
        """
        # Auto-fill first intensity cell when adding second row
        if self.rowCount() == 1:
            first_intensity_item = self.item(0, 1)
            if first_intensity_item and not first_intensity_item.text().strip():
                first_intensity_item.setText("100.0")

        current_rows = self.rowCount()
        self.setRowCount(current_rows + 1)
        self.add_empty_row(current_rows)
        self._update_intensity_column_state()

    def remove_last_row(self):
        """
        Remove the last row (but keep at least one)
        """
        if self.rowCount() > 1:
            self.setRowCount(
                self.rowCount() - 1
            )
            self._update_intensity_column_state()

    def clear_rows(self):
        self.setRowCount(1)
        self._update_intensity_column_state()

    def _connect_signals(self):
        # Validation signal
        self.itemChanged.connect(
            self._validate_item
        )

        # New data signal
        self.sigDataUpdated.connect(
            self.populate_table
        )

    def _update_intensity_column_state(self):
        """
        Make intensity column non-editable when only one row exists
        (i.e. no isotope envelope to define!)
        """
        is_single_row = self.rowCount() == 1

        for row in range(self.rowCount()):
            intensity_item = self.item(row, 1)
            if intensity_item:
                if is_single_row:
                    intensity_item.setFlags(
                        intensity_item.flags() & ~QtCore.Qt.ItemIsEditable
                    )
                else:
                    intensity_item.setFlags(
                        intensity_item.flags() | QtCore.Qt.ItemIsEditable
                    )

    def _validate_item(
        self,
        item: QtWidgets.QTableWidgetItem,
    ):
        """
        Validates that the item contains a valid float. Used for direct
        feedback when user is typing.

        There's a second, 'batch' validation that's executed with
        `self.get_user_input()`
        """
        text = item.text().strip()

        if not text:  # Empty is OK
            item.setData(
                QtCore.Qt.ForegroundRole, None
            )  # Clear custom styling
            return

        try:
            value = float(text)
            if value <= 0:
                raise ValueError("Must be positive non-zero")

            item.setData(QtCore.Qt.ForegroundRole, None)  # Clear custom styling
            item.setText(str(value))  # Normalize the display
        except ValueError:
            item.setForeground(QtCore.Qt.red)  # Red text for invalid input

    def get_user_input(
        self,
    ) -> list[tuple[float, float]]:
        """
        Retrieves all valid m/z, intensity pairs
        from input table
        """
        signals = []
        for row in range(self.rowCount()):
            mz_item = self.item(
                row,
                0,
            )

            intsy_item = self.item(
                row,
                1,
            )

            if mz_item and intsy_item:
                try:
                    mz = float(mz_item.text() or 0.0)
                    intsy = float(intsy_item.text() or 0.0)

                    signals.append(
                        (mz, intsy)
                    )

                except (ValueError, AttributeError) as e:
                    # Skip invalid rows
                    logger.warning(e)
                    continue

        if self._validate_user_input(signals):
            return signals

        return []  # Interpreted as a failed input


    def _validate_user_input(
        self,
        values: list[tuple[float, float]],
    ) -> bool:
        # If there are no signals, invalidate
        if not values:
            return False

        # If there's only one row, then only check m/z value
        if len(values) == 1:
            if values[0][0] != 0.0:
                return True

        # I there's more than one row, return false if any values are 0.0
        for mz, intsy in values:
            if mz == 0.0 or intsy == 0.0:
                return False

        return True

    def keyPressEvent(
        self,
        event: QtGui.QKeyEvent,
    ):
        """
        Handle keyboard events (i.e. Ctrl+V for paste)
        """
        if event.key() == QtCore.Qt.Key_V and event.modifiers() == QtCore.Qt.ControlModifier:
            self._paste_from_clipboard()
        else:
            super().keyPressEvent(event)

    def _paste_from_clipboard(self):
        """
        Paste data from clipboard using pandas
        """
        try:
            # Read clipboard data
            df = pd.read_clipboard(
                engine='python'
            )

            # Ensure at least 2 columns
            if df.shape[1] < 2:
                logger.warning(
                    "Clipboard data must have at least 2 columns (m/z, intensity)"
                )
                return

            # Take first two columns as m/z and intensity
            data = df.iloc[:, :2].values

            # Clear current table and populate with clipboard data
            self.setRowCount(len(data))

            for row, (mz, intensity) in enumerate(data):
                # Create items and populate
                mz_item = QtWidgets.QTableWidgetItem(str(mz))
                intensity_item = QtWidgets.QTableWidgetItem(str(intensity))

                self.setItem(row, 0, mz_item)
                self.setItem(row, 1, intensity_item)

                # Validate the items
                self._validate_item(mz_item)
                self._validate_item(intensity_item)

            # Update intensity column state
            self._update_intensity_column_state()

        except Exception as e:
            logger.warning(
                f"Failed to paste from clipboard: {e}"
            )

    def populate_table(
        self,
        data: list[tuple[float, float]]
    ):
        """
        Used to programmatically populate the table with data
        """
        if not data:
            return

        self.setRowCount(
            len(data)
        )

        for row, (mz, intensity) in enumerate(data):
            mz_item = QtWidgets.QTableWidgetItem(str(mz))
            intensity_item = QtWidgets.QTableWidgetItem(str(intensity))

            self.setItem(row, 0, mz_item)
            self.setItem(row, 1, intensity_item)

            # Validate the items
            self._validate_item(mz_item)
            self._validate_item(intensity_item)

        # Update intensity column state
        self._update_intensity_column_state()

    