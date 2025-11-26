from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QTextDocument
from PyQt5.QtCore import QSize
import pandas as pd
import logging

logger = logging.getLogger()

# Hardcoded for now
MAX_MASS = 4000

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

        # Set selection behaviour to select *Rows*
        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
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
        has_single_row = self.rowCount() == 1

        for row in range(self.rowCount()):
            intensity_item = self.item(row, 1)

            if intensity_item:

                if has_single_row:
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

        # If any signal is larger than MAX_MASS, invalidate
        if any([x[0] > MAX_MASS for x in values]):
            return False

        # If there's only one row, then only check m/z value
        if len(values) == 1:
            if values[0][0] != 0.0:
                return True

        # If there's more than one row, return false if any values are 0.0
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

        Handles two cases:
        - Single number: Sets as m/z with intensity 100.0
        - Table with 2+ columns: Uses first two columns as m/z and intensity
        """
        try:
            # Read clipboard data without expecting headers
            df = pd.read_clipboard(
                engine='python',
                header=None
            )

            # Handle single value case (1x1 DataFrame)
            if df.shape == (1, 1):
                single_value = float(df.iloc[0, 0])
                data = [(single_value, 100.0)]

            # Reject single column with multiple rows
            elif df.shape[1] == 1 and df.shape[0] > 1:
                logger.warning(
                    "Clipboard contains single column with multiple rows. "
                    "Please provide both m/z and intensity values for multiple entries."
                )
                return

            # Handle table case (2+ columns)
            elif df.shape[1] >= 2:
                # Take first two columns as m/z and intensity
                data = df.iloc[:, :2].values

            else:
                logger.warning(
                    "Clipboard data format not recognized."
                )
                return

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

        except pd.errors.EmptyDataError:
            logger.warning(
                "Clipboard is empty or contains no valid data."
            )
        except ValueError as e:
            logger.warning(
                f"Clipboard contains invalid numeric data: {e}"
            )
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

        self.setRowCount(0)  # Clear table

        self.setRowCount(
            len(data)
        )

        for row, (mz, intensity) in enumerate(data):
            mz_item = QtWidgets.QTableWidgetItem(f"{mz:.5f}")
            mz_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter
            )

            intensity_item = QtWidgets.QTableWidgetItem(str(intensity))
            intensity_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter
            )

            self.setItem(row, 0, mz_item)
            self.setItem(row, 1, intensity_item)

            # Validate the items
            self._validate_item(mz_item)
            self._validate_item(intensity_item)

        # Update intensity column state
        self._update_intensity_column_state()


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
