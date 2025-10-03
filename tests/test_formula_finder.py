"""
Test script for FormulaFinderDialog.
"""
import sys
from PyQt5.QtWidgets import QApplication
from gui.dialogues.formula_finder import FormulaFinderDialog


if __name__ == "__main__":
    app = QApplication(sys.argv)

    dialog = FormulaFinderDialog()
    dialog.show()

    sys.exit(app.exec_())

