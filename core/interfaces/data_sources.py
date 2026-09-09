from typing import Protocol, Optional, TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from core.data_structs import (
        Sample,
        SampleUUID,
        AlignmentUUID,
        EnsembleUUID,
    )
    from core.data_structs.alignment import EnsembleAlignment
    from core.data_structs.formula_assignment import FormulaAssignment

class SampleDataSource(Protocol):
    """
    Structural contract for the read/subscribe surface that
     GUI views and models depend on.

    DataRegistry satisfies this by structure - it cannot inherit from it directly
     (see note in core/data_structs/data_registry.py)

    Scope is deliberately limited to Samples. DataRegistry also stores
    EnsembleAlignments and FormulaAssignments, but no consumer reaches those
    through this protocol, so they are intentionally excluded here.
    """
    def get_sample(
        self,
        uuid: 'SampleUUID',
    ) -> Optional['Sample']:
        ...

    def get_all_samples(self) -> list['Sample']:
        ...

    def get_all_sample_uuids(self) -> list['SampleUUID']:
        ...

    def sample_count(self) -> int:
        ...

    def notify_sample_updated(
        self,
        uuid: 'SampleUUID',
    ) -> None:
        """
        Ask the source to re-emit its "sample updated" notification for the
        given UUID, so subscribed models/views refresh.
        """
        ...

    def subscribe_to_changes(
        self,
        addition_callback: Callable[..., None],
        removal_callback: Callable[..., None],
        update_callback: Optional[Callable[..., None]] = None,
        change_type: Literal['Sample', 'Alignment', 'Assignment'] = 'Sample',
    ):
        """
        Subscribe to notifications when the registry changes.

        :param addition_callback: Called when an object is added.
        :param removal_callback: Called when an object is removed.
        :param update_callback: Called when an object is updated/changed.
            Only honoured for change_type='Sample'; optional.
        :param change_type: Which family of changes to subscribe to.
            'Sample' subscribes to Sample add/remove/update signals;
            'Alignment' and 'Assignment' subscribe to add/remove only.
        """
        ...


class AlignmentDataSource(Protocol):
    """
    Structural contract for the read/subscribe surface
     over EnsembleAlignments
    """
    def get_alignment(
        self,
        uuid: 'AlignmentUUID',
    ) -> Optional['EnsembleAlignment']:
        ...

    def get_all_alignment_uuids(self) -> list['AlignmentUUID']:
        ...

    def alignment_count(self) -> int:
        ...

    def subscribe_to_changes(
        self,
        addition_callback: Callable[..., None],
        removal_callback: Callable[..., None],
        update_callback: Optional[Callable[..., None]] = None,
        change_type: Literal['Sample', 'Alignment', 'Assignment'] = 'Sample',
    ):
        """
        Subscribe to alignment change notifications; call with
        change_type='Alignment'. See SampleDataSource.subscribe_to_changes
        for the full parameter contract (update_callback is unused for
        alignments, which emit add/remove only).
        """
        ...


class AssignmentDataSource(Protocol):
    """
    Structural contract for the read/subscribe surface over FormulaAssignments.

    Assignments are keyed by their SOURCE uuid (an EnsembleUUID today),
    so the lookups here take a source uuid rather than an assignment uuid.
    """
    def get_assignment_for_source(
        self,
        source_uuid: 'EnsembleUUID',
    ) -> Optional['FormulaAssignment']:
        ...

    def get_all_assignment_source_uuids(self) -> list['EnsembleUUID']:
        ...

    def assignment_count(self) -> int:
        ...

    def subscribe_to_changes(
        self,
        addition_callback: Callable[..., None],
        removal_callback: Callable[..., None],
        update_callback: Optional[Callable[..., None]] = None,
        change_type: Literal['Sample', 'Alignment', 'Assignment'] = 'Sample',
    ):
        """
        Subscribe to assignment change notifications; call with
        change_type='Assignment'. See SampleDataSource.subscribe_to_changes
        for the full parameter contract (update_callback is unused for
        assignments, which emit add/remove only).
        """
        ...
