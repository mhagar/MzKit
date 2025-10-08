from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QVariant,
    QSortFilterProxyModel,
)
from PyQt5.QtGui import (
    QStandardItem,
    QStandardItemModel,
    QBrush,
)
import pyqtgraph as pg

from gui.resources.FingerprintViewerWindow import Ui_Form
from gui.utils.fingerprint_graphics import array_to_pixmap

from typing import Optional, Literal, TYPE_CHECKING
if TYPE_CHECKING:
        from core.data_structs import (
            Sample, SampleUUID,
        )
        from core.interfaces.data_sources import SampleDataSource


class FingerprintViewerWindow(
    QtWidgets.QWidget,
    Ui_Form,
):
    """
    Window housing a Fingerprint-centric Sample Viewer
    """
    def __init__(
        self,
        data_source: 'SampleDataSource'
    ):
        super().__init__()
        self.setupUi(self)

        self.sample_data_source = data_source
        self.sample_data_source.subscribe_to_changes(
            change_type='Sample',
            addition_callback=self.onSampleAdded,
            removal_callback=self.onSampleRemoved,
            update_callback=self.onSampleUpdated,
        )

        # Main table view
        self.table_model = FingerprintTableModel(
            sample_data_source=data_source,
        )
        
        # Proxy model to filter based on sample/metadata selection
        self.proxy_model = FingerprintTableProxyModel()
        self.proxy_model.setSourceModel(self.table_model)
        
        self.viewTableFingerprints.setModel(
            self.proxy_model
        )
        self.viewTableFingerprints.setMouseTracking(True)
        self.fingerprint_delegate = FingerprintDelegate()

        # Selection views
        self.sample_selection_model = SampleSelectionModel()
        
        # Proxy model to filter sample selection list by text
        self.sample_filter_proxy = SampleSelectionFilterProxyModel()
        self.sample_filter_proxy.setSourceModel(self.sample_selection_model)
        
        self.viewListSamples.setModel(
            self.sample_filter_proxy
        )

        self.metadata_selection_model = MetadataSelectionModel()
        self.viewListMetadata.setModel(
            self.metadata_selection_model,
        )
        
        # Connect selection models to proxy model filtering
        self.proxy_model.setSampleSelectionModel(self.sample_selection_model)
        self.proxy_model.setMetadataSelectionModel(self.metadata_selection_model)
        
        # Connect signals to refresh filtering when selections change
        self.sample_selection_model.itemChanged.connect(
            self.proxy_model.invalidateFilter
        )
        self.metadata_selection_model.itemChanged.connect(
            self.proxy_model.invalidateFilter
        )

    def _update_state(self):
        # TODO: Refactor this so it's not being called for every new sample
        self.viewTableFingerprints.setItemDelegateForColumn(
            self.table_model.columnCount() - 1,
            self.fingerprint_delegate,
        )

    def onSampleAdded(
        self,
        sample: 'Sample',
    ):
        self._update_state()
        self.sample_selection_model.addSample(sample)
        self.metadata_selection_model.addMetadataFieldsFromSample(sample)

    def onSampleRemoved(
        self,
        sample: 'Sample',
    ):
        self._update_state()
        self.sample_selection_model.removeSample(sample)

    def onSampleUpdated(
        self,
        sample: 'Sample',
    ):
        self._update_state()

    def onFinishedInputSampleFilter(self):
        """
        Called when user finishes typing into the filter textbox
        """
        pattern: str = self.lineFilterSamples.text()
        self.sample_filter_proxy.setFilterWildcard(pattern)

    def _showFilteredSamples(
        self,
        checked: bool
    ):
        """
        Check or uncheck all selected rows in the view
        """
        for proxy_index in self.viewListSamples.selectionModel().selectedIndexes():
            source_index = self.sample_filter_proxy.mapToSource(proxy_index)

            # Get the item from the source model and set its check state
            item = self.sample_selection_model.itemFromIndex(source_index)
            if item:
                check_state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                item.setCheckState(check_state)

    def _showFilteredMetadata(
        self,
        checked: bool
    ):
        """
        Uncheck or check all selected rows in the view
        """
        for index in self.viewListMetadata.selectionModel().selectedIndexes():
            item = self.metadata_selection_model.itemFromIndex(index)

            if item:
                check_state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                item.setCheckState(check_state)


    def onSelectionButtonPush(self):
        sender_name: str = self.sender().objectName()

        selection_type, action_type = identify_filter_button(sender_name)

        match selection_type:
            case 'sample':
                match action_type:
                    case 'show':
                        self._showFilteredSamples(True)
                    case 'hide':
                        self._showFilteredSamples(False)

            case 'metadata':
                match action_type:
                    case 'show':
                        self._showFilteredMetadata(True)

                    case 'hide':
                        self._showFilteredMetadata(False)


def identify_filter_button(
    sender_name: str,
) -> tuple[
    Literal['sample', 'metadata'],
    Literal['show', 'hide'],
]:
    """
    Given a sender name, returns a tuple of strings indicating what
    kind of action should be executed
    """
    sender_name = sender_name.lower()

    selection_type: Literal['sample', 'metadata']
    if 'sample' in sender_name:
        selection_type = 'sample'
    elif 'metadata' in sender_name:
        selection_type = 'metadata'
    else:
        raise ValueError(
            f"Neither 'sample' nor 'metadata' found in sender_name: "
            f"{sender_name}"
        )

    action_type: Literal['show', 'hide']
    if 'show' in sender_name:
        action_type = 'show'
    elif 'hide' in sender_name:
        action_type = 'hide'
    else:
        raise ValueError(
            f"Neither 'show' nor 'hide' found in sender_name: "
            f"{sender_name}"
        )

    return selection_type, action_type


class SampleSelectionFilterProxyModel(QSortFilterProxyModel):
    """
    Proxy model that filters the SampleSelectionModel based on text input
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.setFilterKeyColumn(0)  # Filter on the first (name) column


class FingerprintTableProxyModel(QSortFilterProxyModel):
    """
    Proxy model that filters the FingerprintTableModel
    based on sample and metadata selection

    Note: I have not implemented Metadata column selection yet
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sample_selection_model: Optional['SampleSelectionModel'] = None
        self.metadata_selection_model: Optional['MetadataSelectionModel'] = None
    
    def setSampleSelectionModel(
        self,
        selection_model: 'SampleSelectionModel',
    ):
        self.sample_selection_model = selection_model

    def setMetadataSelectionModel(
        self,
        selection_model: 'MetadataSelectionModel',
    ):
        self.metadata_selection_model = selection_model

    def filterAcceptsRow(
        self,
        source_row: int,
        source_parent,
    ):
        """
        Only show rows for samples that are checked in the selection model
        """
        if not self.sample_selection_model:
            return True
            
        # Get the sample UUID from the source model
        source_model = self.sourceModel()
        source_model: FingerprintTableModel

        if not hasattr(source_model, '_row_to_sample_uuid'):
            return True
            
        if source_row not in source_model.get_rows():
            return True
            
        sample_uuid = source_model.get_sample_uuid_by_row_num(source_row)
        
        # Check if this sample is selected in the selection model
        item = self.sample_selection_model.getItemBySampleUUID(sample_uuid)
        if item:
            return item.checkState() == Qt.CheckState.Checked
            
        return True


# Model for keeping track of samples loaded into this window
class FingerprintTableModel(
    QAbstractTableModel,
):
    def __init__(
        self,
        sample_data_source: 'SampleDataSource',
        colormap: str = 'CET-D1A',
        parent=None,
    ):
        super().__init__(parent)
        self.sample_data_source = sample_data_source
        self.sample_data_source.subscribe_to_changes(
            self.onSampleAdded,
            self.onSampleRemoved,
            self.onSampleUpdated,
            change_type='Sample',
        )

        self._row_to_sample_uuid: dict[int, 'SampleUUID'] = {}
        self._sample_uuid_to_row: dict['SampleUUID', int] =  {}

        self._meta_fields: set[str] = set( [] )
        self._col_to_meta_field: dict[int, str] = {}

        self._descriptors: set[str] = set([])

        # # Tracks which columns should actually be visible in table
        # self._visible_array_idxs: list[int] = []

        self.colormap = pg.colormap.get(colormap)
        print(
            "TODO: Using default colormap range, expose to user"
        )


    def _update_sample(
        self,
        sample: 'SampleUUID',
    ):
        pass


    def rowCount(
        self,
        parent=QModelIndex(),
    ):
        return self.sample_data_source.sample_count()


    def columnCount(
        self,
        parent=QModelIndex(),
    ):
        return len(self.get_metadata_fields()) + 1  # plus one column for fingerprint


    def data(
        self,
        index: QModelIndex,
        role=Qt.DisplayRole,
    ):
        if not index.isValid():
            return QVariant()

        if not self._row_to_sample_uuid:
            # No data loaded yet
            return QVariant()

        row = index.row()
        col = index.column()

        sample = self.sample_data_source.get_sample(
            uuid=self._row_to_sample_uuid[row]
        )

        # Different behaviour depending on whether column
        # is metadata or fprint
        match self.col_to_type(col):
            case 'metadata':
                field = self._col_to_meta_field.get(col)

                match role:
                    case Qt.DisplayRole:
                        return sample.metadata.get(field)


            case 'fingerprint':
                match role:
                    case Qt.DisplayRole:
                        if sample.fingerprint:
                            return array_to_pixmap(
                                array=sample.fingerprint.array,
                                colormap=self.colormap,
                            )

                    case Qt.UserRole:
                        if sample.fingerprint:
                            return array_to_pixmap(
                                array=sample.fingerprint.array,
                                colormap=self.colormap,
                            )

        return QVariant()


    def headerData(
        self,
        section,
        orientation,
        role=Qt.DisplayRole,
    ):
        # Setting horizontal header:
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:

            match self.col_to_type(section):
                case 'metadata':
                    if section:
                        return self.get_metadata_field_by_col_num(section)

                case 'fingerprint':
                    return QVariant()

        # Setting vertical header:
        elif role == Qt.DisplayRole and orientation == Qt.Vertical:
            sample = self.sample_data_source.get_sample(
                uuid=self.get_sample_uuid_by_row_num(section)
            )
            return sample.name

        return QVariant()


    def _value_to_color(
        self,
        value: float,
    ) -> QBrush:
        return QBrush(
            self.colormap.map(
                value,
                mode=pg.ColorMap.QCOLOR,
            )
        )


    def set_color_range(
        self,
        min_val: float,
        max_val: float,
    ):
        """
        Update the color map range
        :param min_val:
        :param max_val:
        :return:
        """
        pass


    def onSampleAdded(
        self,
        sample: 'Sample',
    ):
        # Add sample to end of table
        # row = self.sample_data_source.sample_count() + 1
        row = len(self._row_to_sample_uuid)

        self.beginInsertRows(
            QModelIndex(),
            row,
            row,
        )

        # Add UUID to memory
        self._row_to_sample_uuid[row] = sample.uuid
        self._sample_uuid_to_row[sample.uuid] = row

        # Store its fingerprint descriptors
        if sample.fingerprint:
            self._descriptors.update(
                sample.fingerprint.descriptors
            )

        self.endInsertRows()

        # Store its metadata fields and update columns
        if not self.get_metadata_fields():
            self.set_metadata_fields(
                list(sample.metadata.keys())
            )

            self.beginInsertColumns(
                QModelIndex(),
                0,
                len(self.get_metadata_fields()) - 1,
            )
            self.endInsertColumns()


    def onSampleRemoved(
        self,
        sample: 'Sample',
    ):
        row = self._sample_uuid_to_row.pop(sample.uuid)

        self.beginRemoveRows(
            QModelIndex(),
            row,
            row,
        )

        removed_sample = self._row_to_sample_uuid.pop(row)

        # self._update_state()

        self.endRemoveRows()

    def onSampleUpdated(
        self,
        sample: 'Sample',
    ):
        self.beginResetModel()

        self._update_sample(
            sample.uuid
        )

        self.endResetModel()


    def col_to_type(
        self,
        col: int,
    ) -> Literal['metadata', 'fingerprint']:
        """
        Given a column index, returns a string indicating
        what kind of data should be in the column
        :param col:
        :return:
        """
        if col == len(self._meta_fields):
            return 'fingerprint'

        return 'metadata'


    def get_rows(
        self,
    ) -> list[int]:
        return list(self._row_to_sample_uuid.keys())

    def get_sample_uuid_by_row_num(
        self,
        row: int,
    ) -> Optional[ 'SampleUUID' ]:
        return self._row_to_sample_uuid.get(row)


    def set_metadata_fields(
        self,
        fields: list[str],
    ):
        self._meta_fields.update(
            fields
        )
        for idx, field in enumerate(
            sorted(self._meta_fields)
        ):
            self._col_to_meta_field[idx] = field


    def get_metadata_fields(
        self,
    ) -> list[str]:
        return sorted(self._meta_fields)

    def get_metadata_field_by_col_num(
        self,
        col: int,
    ) -> Optional[str]:
        return self._col_to_meta_field.get(col)


class FingerprintDelegate(
    QtWidgets.QStyledItemDelegate
):
    """
    Handles drawing the fingerprint graphic inside the table view
    """
    sigFingerprintHovered = QtCore.pyqtSignal(
        int,  # row
        int,  # fingerprint idx
    )

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent)
        self.zoom_level: float = 1.0
        self.scroll_offset: int = 0

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QModelIndex,
    ):
        painter.setRenderHint(
            QtGui.QPainter.SmoothPixmapTransform, False
        )  # Nearest neighbour scaling

        pixmap = index.data(Qt.UserRole)
        if pixmap and isinstance(pixmap, QtGui.QPixmap):
            # target_rect -> where image will be painted
            target_rect = option.rect  # Cell's drawing area

            # source_rect -> what part of pixmap to paint from
            # handle zooming/panning here
            pixmap_width: int = pixmap.width()
            visible_width: int = int(pixmap_width / self.zoom_level)
            start_x = self.scroll_offset

            # clamp to pixmap boundaries
            start_x: int = max(
                0,
                min(
                    start_x,
                    pixmap_width - visible_width,
                )
            )
            end_x = min(
                start_x + visible_width,
                pixmap_width,
            )

            source_rect = QtCore.QRect(
                start_x, 0,         # x,y top left corner
                end_x - start_x, 1  # width/height of selection
            )

            painter.drawPixmap(
                target_rect,
                pixmap,
                source_rect,
            )

        else:
            # Fall back to default rendering for non-bitmap cells
            super().paint(
                painter,
                option,
                index,
            )

    def sizeHint(
        self,
        option,
        index: QModelIndex,
    ):
        """
        Control cell size
        """
        return QtCore.QSize(200, 50)

    def editorEvent(
        self,
        event: QtCore.QEvent,
        model: QtCore.QAbstractItemModel,
        option: QtWidgets.QStyleOptionViewItem,
        index: QModelIndex,
    ):
        if event.type() == QtCore.QEvent.MouseMove:
            # Convert mouse position to fingerprint array idx
            mouse_x = event.pos().x()
            cell_rect: QtCore.QRect = option.rect
            pixmap: QtGui.QPixmap = index.data(Qt.UserRole)

            if not pixmap:
                return super().editorEvent(event, model, option, index)

            fingerprint_idx: int = self._mouse_to_fingerprint_idx(
                mouse_x=mouse_x,
                cell_rect=cell_rect,
                pixmap_width=pixmap.width(),
            )

            if fingerprint_idx:
                self.sigFingerprintHovered.emit(
                    index.row(),
                    fingerprint_idx,
                )


        return super().editorEvent(event, model, option, index)

    def _mouse_to_fingerprint_idx(
        self,
        mouse_x: float,
        cell_rect: QtCore.QRect,
        pixmap_width: int,
    ) -> Optional[int]:
        """
        Convert mouse X coordinate into fingerprint array index
        :param mouse_x:
        :param cell_rect:
        :param pixmap_width:
        :return:
        """
        relative_x = (mouse_x - cell_rect.left()) / cell_rect.width()

        # Compensate for zoom/scroll
        visible_width = pixmap_width / self.zoom_level
        start_x = self.scroll_offset

        fingerprint_x = start_x + (relative_x * visible_width)
        fingerprint_idx = int(fingerprint_x)

        # Clamp to valid range
        if 0 <= fingerprint_idx < pixmap_width:
            return fingerprint_idx

        return None


# Models for keeping track of which Samples/Features/Metadata user
#  wants to see in the table
class SampleSelectionModel(
    QStandardItemModel,
):
    """
    Stores info about which fingerprints to display in the main table.
    Note: this doesn't store the actual fingerprints - just information
    about which ones the user wishes to see
    """
    UuidRole = Qt.UserRole + 1

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def addSample(
        self,
        sample: 'Sample',
        selected: bool = True,
    ):
        if self.sampleAlreadyLoaded(
            sample.uuid
        ):
            return

        item = QStandardItem(
            sample.name
        )

        item.setCheckable(True)
        item.setFlags(
            Qt.ItemIsEnabled |
            Qt.ItemIsSelectable |
            Qt.ItemIsUserCheckable
        )

        item.setData(
            sample.uuid,
            self.UuidRole
        )

        if selected:
            item.setCheckState(Qt.CheckState.Checked)

        self.appendRow(item)


    def sampleAlreadyLoaded(
        self,
        uuid: 'SampleUUID',
    ) -> bool:
        """
        Given a UUID, checks if the sample is already loaded
        :param uuid:
        :return:
        """

        if self.getItemBySampleUUID(uuid):
            return True

        return False


    def removeSample(
        self,
        sample: 'Sample',
    ):
        item = self.getItemBySampleUUID(sample.uuid)

        if item:
            self.removeRow(item.row())


    def getItemBySampleUUID(
        self,
        uuid: 'SampleUUID',
    ) -> Optional[QtGui.QStandardItem]:
        for row_idx in range(0, self.rowCount()):
            if uuid == self.item(row_idx).data(
                self.UuidRole,
            ):
                return self.item(row_idx)

        return None


class MetadataSelectionModel(
    QStandardItemModel,
):
    """
    Stores info about which metadata to display in the main table
    Note: this doesn't store the actual metadata: just information about
    which ones the user wishes to see
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._metadata_fields: set[str] = set([])


    def addMetadataField(
        self,
        metadata_name: str,
        selected: bool = True,
    ):
        if self.metadataAlreadyLoaded(
            metadata_name
        ):
            return

        item = QStandardItem(
            metadata_name
        )

        self._metadata_fields.add(metadata_name)

        item.setCheckable(True)
        item.setFlags(
            Qt.ItemIsEnabled |
            Qt.ItemIsSelectable |
            Qt.ItemIsUserCheckable
        )

        item.setData(
            metadata_name,
        )

        if selected:
            item.setCheckState(Qt.CheckState.Checked)

        self.appendRow(item)


    def addMetadataFieldsFromSample(
        self,
        sample: 'Sample',
    ):
        """
        Given a Sample, extracts its metadata field names and
        adds them to the model (only if they have not been seen yet)
        """
        sample_metadata_fields: set[str] = set(
            sample.metadata.keys()
        )

        diff = sample_metadata_fields.difference(self._metadata_fields)
        for new_field in diff:
            self.addMetadataField(
                metadata_name=new_field
            )


    def metadataAlreadyLoaded(
        self,
        metadata_name: str,
    ) -> bool:
        if metadata_name in self._metadata_fields:
            return True

        return False


    def getItemByMetadataName(
        self,
        metadata_name: str,
    ) -> Optional[QStandardItem]:
        for row_idx in range(0, self.rowCount()):
            if metadata_name == self.item(row_idx).data():
                return self.item(row_idx)

        return None
