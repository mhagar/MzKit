"""
Qt model for representing samples in the data registry
"""
from PyQt5.QtCore import (
    QAbstractListModel,
    QModelIndex,
    Qt,
)

from PyQt5.QtGui import QIcon

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from core.data_structs import (
        DataRegistry,
        Sample,
        SampleUUID,
    )


class SampleListModel(QAbstractListModel):
    """
    Model representing Samples in DataRegistry
    """
    def __init__(
        self,
        registry: 'DataRegistry',
        parent=None,
    ):
        super().__init__(parent)
        self.registry: 'DataRegistry' = registry
        self._sample_uuids: list['SampleUUID'] = (
            self.registry.get_all_sample_uuids()
        )

        self.registry.sigSampleAdded.connect(
            self.onSampleAdded
        )
        self.registry.sigSampleRemoved.connect(
            self.onSampleRemoved
        )

    def data(
        self,
        index: QModelIndex,
        role=Qt.DisplayRole,
    ):
        sample = self.getSampleAtIndex(index)
        if not sample:
            # Invalid index
            return

        match role:
            case Qt.DisplayRole:
                return sample.name

            case Qt.DecorationRole:
                return get_data_type_icon(
                    sample
                )

            case Qt.ToolTipRole:
                return get_sample_tooltip(
                    sample
                )


    def rowCount(
        self,
        parent=QModelIndex(),
    ):
        return len(self._sample_uuids)


    def getSampleAtIndex(
        self,
        index: QModelIndex,
    ) -> Optional['Sample']:

        if ( not index.isValid()
            or index.row() >= self.rowCount()):
            return None

        uuid = self._sample_uuids[index.row()]
        return self.registry.get_sample(uuid)


    def getAllSamples(
        self,
    ) -> list['Sample']:
        return self.registry.get_all_samples()


    def getSampleByUUID(
        self,
        uuid: 'SampleUUID',
    ) -> Optional['Sample']:
        return self.registry.get_sample(uuid)


    def onSampleAdded(
        self,
        sample: 'Sample',
    ):
        """
        Update Qt model to reflect registry changes.
        Inserts sample at end
        """
        row = len(self._sample_uuids)  # Use current length, not rowCount()
        self.beginInsertRows(
            QModelIndex(),
            row,
            row,
        )

        self._sample_uuids.append(sample.uuid)

        self.endInsertRows()


    def onSampleRemoved(
        self,
        sample: 'Sample',
    ):
        """
        Update Qt model to reflect registry changes
        """
        try:
            row = self._sample_uuids.index(sample.uuid)
            self.beginRemoveRows(
                QModelIndex(),
                row,
                row,
            )
            self._sample_uuids.pop(row)
            self.endRemoveRows()
        except ValueError:
            # UUID not found; possibly already removed
            pass


def get_sample_content_types(
    sample: 'Sample'
) -> tuple[bool, bool]:
    """
    Returns tuple of (has_injection, has_fingerprint)
    """
    has_injection = sample.injection is not None
    has_fingerprint = sample.fingerprint is not None
    return has_injection, has_fingerprint


def get_data_type_icon(
    sample: 'Sample'
) -> QIcon:
    has_injection, has_fingerprint = get_sample_content_types(sample)

    if has_injection and has_fingerprint:
        return QIcon(
            "gui/resources/icons/sample_fingerprint_and_injection.svg"
        )
    elif has_injection:
        return QIcon(
            "gui/resources/icons/sample_injection_only.svg"
        )
    elif has_fingerprint:
        return QIcon(
            "gui/resources/icons/sample_fingerprint_only.svg"
        )
    else:
        return QIcon("gui/resources/icons/sample_empty.svg")


def get_sample_tooltip(
    sample: 'Sample'
) -> str:
    tooltip = (
        f"Injection: {'false' if sample.injection is None else 'true'}, "
        f"Fingerprint: {'false' if sample.fingerprint is None else 'true'}"
    )
    return tooltip