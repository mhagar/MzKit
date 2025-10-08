"""
Proxy model for SampleListModel with filtering and natural sorting capabilities
"""
import re
from PyQt5.QtCore import QSortFilterProxyModel, Qt
from typing import Optional, TYPE_CHECKING

from core.utils.natural_sort import natural_sort_key
from .sample_list_model import get_sample_content_types

if TYPE_CHECKING:
    from PyQt5.QtCore import QModelIndex
    from .sample_list_model import SampleListModel


class SampleProxyModel(QSortFilterProxyModel):
    """
    Proxy model that adds filtering and natural sorting to SampleListModel
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._filter_text: str = ""
        self._show_injections: bool = True
        self._show_fingerprints: bool = True
        self.setDynamicSortFilter(True)
        self.setSortCaseSensitivity(Qt.CaseInsensitive)
        
    def set_filter_text(self, text: str) -> None:
        """Set the search filter text"""
        self._filter_text = text.lower()
        self.invalidateFilter()

    def set_filter_criteria(
        self,
        text: str,
        show_injections: bool,
        show_fingerprints: bool,
    ) -> None:
        """
        Set filter criteria (text and data types)
        """
        print(
            f"Setting filter criteria, text: {text}, inj: {show_injections}, fp: {show_fingerprints}"
        )
        self._filter_text = text.lower()
        self._show_injections = show_injections
        self._show_fingerprints = show_fingerprints
        self.invalidateFilter()
    
    def filterAcceptsRow(
        self,
        source_row: int,
        source_parent: 'QModelIndex',
    ) -> bool:
        """
        Override to implement custom filtering logic
        """
        source_model: 'SampleListModel' = self.sourceModel()
        if not source_model:
            return True

        # Check if the row is valid
        if source_row < 0 or source_row >= source_model.rowCount(source_parent):
            return False

        try:
            index = source_model.index(source_row, 0, source_parent)
            sample = source_model.getSampleAtIndex(index)

            if not sample:
                return True  # Accept rows that might be in transition

            # Text filtering
            if self._filter_text:
                sample_name = sample.name.lower() if sample.name else ""
                if self._filter_text not in sample_name:
                    return False

            # Data type filtering
            has_injection, has_fingerprint = get_sample_content_types(sample)

            # If neither injection nor fingerprint buttons are checked, show nothing
            if not self._show_injections and not self._show_fingerprints:
                return False

            # Show sample if it matches the requested data types
            if self._show_injections and has_injection:
                return True
            if self._show_fingerprints and has_fingerprint:
                return True

            # If sample has no data but we're showing both types, show it
            if self._show_injections and self._show_fingerprints and not has_injection and not has_fingerprint:
                return True

            return False

        except (IndexError, AttributeError):
            # Accept rows during model transitions
            return True
    
    def lessThan(
        self,
        left: 'QModelIndex',
        right: 'QModelIndex',
    ) -> bool:
        """
        Override to implement natural sorting
        """

        source_model: 'SampleListModel' = self.sourceModel()
        if not source_model:
            return super().lessThan(left, right)

        try:
            left_sample = source_model.getSampleAtIndex(left)
            right_sample = source_model.getSampleAtIndex(right)

            if not left_sample or not right_sample:
                return super().lessThan(left, right)

            left_name = left_sample.name or ""
            right_name = right_sample.name or ""

            return natural_sort_key(left_name) < natural_sort_key(right_name)

        except (IndexError, AttributeError):
            # Fall back to default comparison during model transitions
            return super().lessThan(left, right)
