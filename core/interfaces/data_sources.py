from typing import Any, Protocol, Optional, TYPE_CHECKING, Callable, Iterable, Literal

if TYPE_CHECKING:
    from core.data_structs import (
        Ensemble,
        Sample,
        SampleUUID,
        AlignmentUUID,
        EnsembleUUID,
        AnalyteUUID,
        GroupUUID,
    )
    from core.data_structs.alignment import EnsembleAlignment
    from core.data_structs.ensemble_group import EnsembleGroup
    from core.data_structs.similarity_matrix import SimilarityMatrix

class SampleDataSource(Protocol):
    """
    Structural contract for the read/subscribe surface that
     GUI views and models depend on.

    DataRegistry satisfies this by structure - it cannot inherit from it directly
     (see note in core/data_structs/data_registry.py)

    Scope is deliberately limited to Samples; the other protocols below
    cover alignments, ensemble annotations, groups and matrices.
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
        change_type: Literal['Sample', 'Alignment', 'Group', 'Matrix'] = 'Sample',
    ):
        """
        Subscribe to notifications when the registry changes.

        :param addition_callback: Called when an object is added.
        :param removal_callback: Called when an object is removed.
        :param update_callback: Called when an object is updated/changed.
            Honoured for 'Sample' and 'Group'; optional.
        :param change_type: Which family of changes to subscribe to.
            'Sample' subscribes to Sample add/remove/update signals;
            'Alignment' subscribes to add/remove only. Ensemble edits
            are signalled by `sigEnsembleAnnotated` instead
            (see EnsembleDataSource).
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
        change_type: Literal['Sample', 'Alignment', 'Group', 'Matrix'] = 'Sample',
    ):
        """
        Subscribe to alignment change notifications; call with
        change_type='Alignment'. See SampleDataSource.subscribe_to_changes
        for the full parameter contract (update_callback is unused for
        alignments, which emit add/remove only).
        """
        ...


class EnsembleDataSource(Protocol):
    """
    Structural contract for reading and editing what's stored on Ensembles
    (formula assignment, structure annotation). Edits go through `update_ensemble` /
    `set_chosen_candidate`, which emit `sigEnsembleAnnotated(ensemble_uuid)`.
    """
    sigEnsembleAnnotated: Any   # pyqtSignal(EnsembleUUID)

    def get_ensemble(
        self,
        uuid: 'EnsembleUUID',
    ) -> Optional['Ensemble']:
        ...

    def update_ensemble(
        self,
        uuid: 'EnsembleUUID',
        **changes,
    ):
        ...

    def remove_ensembles(
        self,
        uuids: Iterable['EnsembleUUID'],
    ):
        """Delete ensembles from their injections (and from groups)"""
        ...

    def set_chosen_candidate(
        self,
        uuid: 'EnsembleUUID',
        chosen_idx: Optional[int],
        chosen_by: str = 'user',
    ):
        """
        Accept a different candidate of an ensemble's formula assignment
        (None clears the choice)
        """
        ...


class GroupDataSource(Protocol):
    """
    Structural contract for the read/subscribe surface over EnsembleGroups.
    """
    def get_group(
        self,
        uuid: 'GroupUUID',
    ) -> Optional['EnsembleGroup']:
        ...

    def get_all_groups(self) -> list['EnsembleGroup']:
        ...

    def groups_for_ensembles(
        self,
        ensemble_uuids: Iterable['EnsembleUUID'],
    ) -> list['EnsembleGroup']:
        """Groups with any member in `ensemble_uuids`."""
        ...

    def register_group(
        self,
        group: 'EnsembleGroup',
    ):
        ...

    def remove_group(
        self,
        uuid: 'GroupUUID',
    ):
        ...

    def notify_group_updated(
        self,
        group: 'EnsembleGroup',
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
        change_type: Literal['Sample', 'Alignment', 'Group', 'Matrix'] = 'Sample',
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
        change_type: Literal['Sample', 'Alignment', 'Group', 'Matrix'] = 'Sample',
    ):
        """
        Subscribe to matrix change notifications; call with
        change_type='Matrix' (add/remove only).
        """
        ...
