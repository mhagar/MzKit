from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem

from core.data_structs import SampleUUID, Sample, Injection, Fingerprint
from core.interfaces.data_sources import SampleDataSource


class SampleViewerItemModel(
    QStandardItemModel,
):
    """
    Stores info about what to extract/display from raw data
    (Which is stored in the DataRegistry, not here!)
    """
    UuidRole = Qt.UserRole + 1

    def __init__(
        self,
        sample_data_source: 'SampleDataSource',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.sample_data_source = sample_data_source

        self.setHorizontalHeaderLabels(
            ['Samples']
        )


    def getSample(
        self,
        uuid: 'SampleUUID',
    ) -> Optional['Sample']:
        return self.sample_data_source.get_sample(uuid)


    def getInjection(
        self,
        uuid: 'SampleUUID',
    ) -> 'Injection':
        return self.getSample(uuid).injection


    def getFingerprint(
        self,
        uuid: 'SampleUUID',
    ) -> 'Fingerprint':
        return self.getSample(uuid).fingerprint

    def addSample(
        self,
        uuid: 'SampleUUID',
        visible: bool = True,
    ):
        """
        Adds a Sample to the viewer.

        If visible == True, initializes it pre-checked
        """
        if self.sampleAlreadyLoaded(uuid):
            return

        sample = self.getSample(uuid)

        item = QStandardItem(sample.name)

        item.setCheckable(True)
        item.setFlags(
            Qt.ItemIsEnabled |
            Qt.ItemIsSelectable |
            Qt.ItemIsDragEnabled |
            Qt.ItemIsUserCheckable
        )

        item.setData(
            uuid,
            self.UuidRole,  # SampleUUID
        )
        if visible:
            item.setCheckState(Qt.Checked)

        self.appendRow(item)


    def removeSample(
        self,
        uuid: 'SampleUUID'
    ):
        """
        Removes a sample from the viewer
        """
        for row in range(self.rowCount()):
            for col in range(self.columnCount()):
                item = self.item(row, col)

                if item and item.data(self.UuidRole) == uuid:
                    self.removeRow(row)


    def sampleAlreadyLoaded(
        self,
        uuid: 'SampleUUID',
    ):
        """
        Given a UUID, checks if the sample is already loaded
        :param uuid:
        :return:
        """
        for row_idx in range(0, self.rowCount()):
            if uuid == self.item(row_idx).data(self.UuidRole):
                return True

        return False

    def getSampleUuidAtRow(
        self,
        row: int,
    ) -> Optional[ 'SampleUUID' ]:
        item = self.item(row)
        if item:
            return item.data(self.UuidRole)
        return None
