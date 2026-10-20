from typing import Protocol, Optional, TYPE_CHECKING, Callable, Iterable, Literal

if TYPE_CHECKING:
    from core.data_structs import (
        Sample,
        SampleUUID,
        AlignmentUUID,
        EnsembleUUID,
        AnalyteUUID,
        GroupUUID,
    )
    from core.data_structs.alignment import EnsembleAlignment
    from core.data_structs.analyte_group import AnalyteGroup
    from core.data_structs.formula_assignment import FormulaAssignment
    from core.data_structs.similarity_matrix import SimilarityMatrix

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
        change_type: Literal['Sample', 'Alignment', 'Assignment', 'Group', 'Matrix'] = 'Sample',
    ):
        """
        Subscribe to notifications when the registry changes.

        :param addition_callback: Called when an object is added.
        :param removal_callback: Called when an object is removed.
        :param update_callback: Called when an object is updated/changed.
            Honoured for 'Sample', 'Assignment' and 'Group'; optional.
        :param change_type: Which family of changes to subscribe to.
            'Sample' subscribes to Sample add/remove/update signals;
            'Alignment' subscribes to add/remove only; 'Assignment' and
            'Group' also honour update_callback.
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
        change_type: Literal['Sample', 'Alignment', 'Assignment', 'Group', 'Matrix'] = 'Sample',
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

    def set_chosen_candidate(
        self,
        source_uuid: 'EnsembleUUID',
        chosen_idx: Optional[int],
        chosen_by: str = 'user',
    ):
        """
        Accept a different candidate of a registered assignment (None clears
        the choice); notifies update subscribers.
        """
        ...

    def register_assignment(
        self,
        assignment: 'FormulaAssignment',
    ):
        ...

    def remove_assignment(
        self,
        source_uuid: 'EnsembleUUID',
    ):
        ...

    def subscribe_to_changes(
        self,
        addition_callback: Callable[..., None],
        removal_callback: Callable[..., None],
        update_callback: Optional[Callable[..., None]] = None,
        change_type: Literal['Sample', 'Alignment', 'Assignment', 'Group', 'Matrix'] = 'Sample',
    ):
        """
        Subscribe to assignment change notifications; call with
        change_type='Assignment'. See SampleDataSource.subscribe_to_changes
        for the full parameter contract; update_callback fires when an
        assignment's chosen candidate changes.
        """
        ...


class GroupDataSource(Protocol):
    """
    Structural contract for the read/subscribe surface over AnalyteGroups.
    """
    def get_group(
        self,
        uuid: 'GroupUUID',
    ) -> Optional['AnalyteGroup']:
        ...

    def get_all_groups(self) -> list['AnalyteGroup']:
        ...

    def groups_for_analytes(
        self,
        analyte_uuids: Iterable['AnalyteUUID'],
    ) -> list['AnalyteGroup']:
        """Groups with any member in `analyte_uuids`."""
        ...

    def register_group(
        self,
        group: 'AnalyteGroup',
    ):
        ...

    def remove_group(
        self,
        uuid: 'GroupUUID',
    ):
        ...

    def notify_group_updated(
        self,
        group: 'AnalyteGroup',
    ):
        """
        Groups are edited in place; call this after every edit so
        update subscribers refresh.
        """
        ...

    def subscribe_to_changes(
        self,
        addition_callback: Callable[..., None],
        removal_callback: Callable[..., None],
        update_callback: Optional[Callable[..., None]] = None,
        change_type: Literal['Sample', 'Alignment', 'Assignment', 'Group', 'Matrix'] = 'Sample',
    ):
        """
        Subscribe to group change notifications; call with
        change_type='Group'. See SampleDataSource.subscribe_to_changes
        for the full parameter contract.
        """
        ...


class MatrixDataSource(Protocol):
    """
    Structural contract for the read/subscribe surface over
     SimilarityMatrices (computed or imported).
    """
    def get_matrix(
        self,
        uuid: int,
    ) -> Optional['SimilarityMatrix']:
        ...

    def get_all_matrices(self) -> list['SimilarityMatrix']:
        ...

    def matrices_for_analytes(
        self,
        analyte_uuids: Iterable['AnalyteUUID'],
    ) -> list['SimilarityMatrix']:
        """Matrices sharing any id with `analyte_uuids`."""
        ...

    def find_matrix(
        self,
        provenance: dict,
        alignment: 'EnsembleAlignment',
    ) -> Optional['SimilarityMatrix']:
        """A matrix with this provenance covering all of `alignment`."""
        ...

    def register_matrix(
        self,
        matrix: 'SimilarityMatrix',
    ):
        ...

    def remove_matrix(
        self,
        uuid: int,
    ):
        ...

    def subscribe_to_changes(
        self,
        addition_callback: Callable[..., None],
        removal_callback: Callable[..., None],
        update_callback: Optional[Callable[..., None]] = None,
        change_type: Literal['Sample', 'Alignment', 'Assignment', 'Group', 'Matrix'] = 'Sample',
    ):
        """
        Subscribe to matrix change notifications; call with
        change_type='Matrix' (add/remove only).
        """
        ...
