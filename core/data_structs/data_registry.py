"""
Central data registry for performing UUID <-> object look-ups.

Dependent on PyQt5 for signals/slots, but should not otherwise involve
any Qt.
"""

from PyQt5 import QtCore

from core.interfaces.data_sources import SampleDataSource

from typing import Iterable, Optional, TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from core.data_structs import (
        Sample,
        SampleUUID,
        InjectionUUID,
        FingerprintUUID,
        AlignmentUUID,
        EnsembleUUID,
        AnalyteUUID,
        GroupUUID,
    )
    from core.data_structs.alignment import EnsembleAlignment
    from core.data_structs.analyte_group import AnalyteGroup
    from core.data_structs.analyte_annotation import AnalyteAnnotation
    from core.data_structs.formula_assignment import FormulaAssignment
    from core.data_structs.similarity_matrix import SimilarityMatrix

class DataRegistry(
    QtCore.QObject,
    # NOTE: don't inherit SampleDataSource here.
    # Because it's a typing.Protocol, mixing it w/ QObject
    # raises a metaclass conflict
    # Just make sure DataRegistry satisfies SampleDataSource by shape
):
    """
    Central data registry. Stores Samples,
     and performs UUID <-> object look-ups

    # TODO: Implement global Ensemble searching here
    """
    sigSampleAdded = QtCore.pyqtSignal(
        object  # Sample
    )
    sigSampleRemoved = QtCore.pyqtSignal(
        object  # Sample
    )
    sigSampleUpdated = QtCore.pyqtSignal(
        object  # Sample
    )
    sigAlignmentAdded = QtCore.pyqtSignal(
        object  # EnsembleAlignment
    )
    sigAlignmentRemoved = QtCore.pyqtSignal(
        object  # EnsembleAlignment
    )
    sigAssignmentAdded = QtCore.pyqtSignal(
        object  # FormulaAssignment
    )
    sigAssignmentRemoved = QtCore.pyqtSignal(
        object  # FormulaAssignment
    )
    sigAssignmentUpdated = QtCore.pyqtSignal(
        object  # FormulaAssignment (e.g. its chosen candidate changed)
    )
    sigGroupAdded = QtCore.pyqtSignal(
        object  # AnalyteGroup
    )
    sigGroupRemoved = QtCore.pyqtSignal(
        object  # AnalyteGroup
    )
    sigGroupUpdated = QtCore.pyqtSignal(
        object  # AnalyteGroup (edited in place: title, notes, membership)
    )
    sigAnnotationAdded = QtCore.pyqtSignal(
        object  # AnalyteAnnotation
    )
    sigAnnotationRemoved = QtCore.pyqtSignal(
        object  # AnalyteAnnotation
    )
    sigAnnotationUpdated = QtCore.pyqtSignal(
        object  # AnalyteAnnotation (replaced one for the same analyte)
    )
    sigMatrixAdded = QtCore.pyqtSignal(
        object  # SimilarityMatrix
    )
    sigMatrixRemoved = QtCore.pyqtSignal(
        object  # SimilarityMatrix
    )

    def __init__(self):
        self._samples: dict['SampleUUID', 'Sample'] = {}
        self._sample_name_to_uuid: dict[str, 'SampleUUID'] = {}
        self._alignments: dict['AlignmentUUID', 'EnsembleAlignment'] = {}
        # Keyed by SOURCE uuid (an EnsembleUUID today); latest assignment wins.
        self._assignments: dict['EnsembleUUID', 'FormulaAssignment'] = {}
        self._groups: dict['GroupUUID', 'AnalyteGroup'] = {}
        # Reverse index for groups_for_analytes; rebuilt on every group change
        self._groups_by_analyte: dict['AnalyteUUID', set['GroupUUID']] = {}
        self._annotations: dict['AnalyteUUID', 'AnalyteAnnotation'] = {}
        self._matrices: dict[int, 'SimilarityMatrix'] = {}
        super().__init__()

    def subscribe_to_changes(
        self,
        addition_callback,
        removal_callback,
        update_callback=None,
        change_type: Literal[
            'Sample', 'Alignment', 'Assignment', 'Group', 'Annotation', 'Matrix'
        ] = 'Sample',
    ):
        match change_type:
            case 'Sample':
                self.sigSampleAdded.connect(
                    addition_callback
                )

                self.sigSampleRemoved.connect(
                    removal_callback
                )

                if update_callback:
                    self.sigSampleUpdated.connect(
                        update_callback
                    )

            case 'Alignment':
                self.sigAlignmentAdded.connect(
                    addition_callback
                )

                self.sigAlignmentRemoved.connect(
                    removal_callback
                )

            case 'Assignment':
                self.sigAssignmentAdded.connect(
                    addition_callback
                )

                self.sigAssignmentRemoved.connect(
                    removal_callback
                )

                if update_callback:
                    self.sigAssignmentUpdated.connect(
                        update_callback
                    )

            case 'Group':
                self.sigGroupAdded.connect(
                    addition_callback
                )

                self.sigGroupRemoved.connect(
                    removal_callback
                )

                if update_callback:
                    self.sigGroupUpdated.connect(
                        update_callback
                    )

            case 'Annotation':
                self.sigAnnotationAdded.connect(
                    addition_callback
                )

                self.sigAnnotationRemoved.connect(
                    removal_callback
                )

                if update_callback:
                    self.sigAnnotationUpdated.connect(
                        update_callback
                    )

            case 'Matrix':
                self.sigMatrixAdded.connect(
                    addition_callback
                )

                self.sigMatrixRemoved.connect(
                    removal_callback
                )

    def register_sample(
        self,
        sample: 'Sample',
    ):
        """
        Registers a Sample into the database.

        If there already exists a Sample with the same name, but
        with compatible contents (i.e. one has MS data and the other
        has fingerprints), merges the two.
        :param sample:
        :return:
        """
        self.validate_new_sample(sample)

        matched_sample_uuid: Optional['SampleUUID'] = self.match_samplename(
            sample.name
        )

        if matched_sample_uuid:
            # Sample with same name already exists
            self.merge_samples(
                source=sample,
                destination_uuid=matched_sample_uuid
            )

        else:
            # Brand new sample
            self._samples[sample.uuid] = sample
            self._sample_name_to_uuid[sample.name] = sample.uuid
            self.sigSampleAdded.emit(
                sample
            )

    def register_samples(
        self,
        samples: list['Sample'],
    ):
        for sample in samples:
            self.register_sample(sample)

    def get_sample(
        self,
        uuid: 'SampleUUID',
    ) -> Optional['Sample']:
        return self._samples.get(uuid)

    def get_all_samples(self) -> list['Sample']:
        return list(self._samples.values())

    def get_all_sample_uuids(self) -> list['SampleUUID']:
        return list(self._samples.keys())

    def remove_sample(
        self,
        uuid: 'SampleUUID',
    ):
        if uuid in self._samples:
            sample_to_remove = self._samples[uuid]
            self.sigSampleRemoved.emit(
                sample_to_remove
            )
            del self._samples[uuid]
            del self._sample_name_to_uuid[sample_to_remove.name]

            return

        raise ValueError(
            f"No sample with UUID {uuid} registered"
        )

    def validate_new_sample(
        self,
        sample: 'Sample'
    ):
        """
        Validates a sample for registration. This check must
        succeed before registering a sample.

        Currently enforces uuid uniqueness and that a Sample carries at
        least one of (Injection, Fingerprint). Fingerprint cross-sample
        consistency checks (matching array size / descriptors / metadata
        fields) are not yet ported from the old fingerprint model.

        :param sample:
        :return:
        """
        if sample.uuid in self._samples:
            raise ValueError(
                f"Sample {sample.name} is already registered with uuid: "
                f"{self._samples[sample.uuid].uuid}"
            )

        if not sample.injection and not sample.fingerprint:
            raise ValueError(
                f"Sample must have either a Fingerprint or Injection assigned"
                f"before being registered. Exception caused by sample: {sample}"
            )

    def match_samplename(
        self,
        name: str,
    ) -> Optional['SampleUUID']:
        """
        Given a name, checks whether there already exists
        a Sample with the same name.

        This is used to merge imported fingerprints and MS data

        :param name:
        :return:
        """
        if name in self._sample_name_to_uuid:
            return self._sample_name_to_uuid[name]

        return None

    def merge_samples(
        self,
        source: 'Sample',
        destination_uuid: 'SampleUUID',
    ) -> None:
        """
        Given a source sample and a destination UUID, merges
        the two.

        This is used if two samples have identical names, but one
        only has MS data, and the other has only fingerprint data.
        """
        destination = self.get_sample(destination_uuid)

        if source.name != destination.name:
            raise ValueError(
                f"Source and destination samples have different names: "
                f"{source.name} vs {destination.name}"
            )

        if not _merge_is_valid(source, destination):
            raise ValueError(
                f"Incompatible merge attempted:\n"
                f"Source: {source}\n"
                f"Destination: {destination}"
            )

        # Merge the fingerprint/injection, whichever is appropriate
        if not destination.fingerprint:
            destination.set_fingerprint(
                source.fingerprint
            )
            self.sigSampleUpdated.emit(destination)

        if not destination.injection:
            destination.set_injection(
                source.injection
            )
            self.sigSampleUpdated.emit(destination)

        # Merge the metadata attribute
        destination.metadata.update(source.metadata)

    def clear(self) -> None:
        """
        Empties the registry of all Samples and Alignments, emitting the
        corresponding removal signals so that every subscribed model/view
        resets to an empty state.

        The DataRegistry instance itself is preserved (only its contents
        are dropped), so all existing references and signal/slot
        connections held by controllers, models and views remain valid.

        Assignments and alignments are removed first, since they reference
        Samples (via source/ensemble UUIDs). Removing the last alignment
        orphans every group, so groups go with it.
        """
        for source_uuid in self.get_all_assignment_source_uuids():
            self.remove_assignment(source_uuid)

        for uuid in self.get_all_alignment_uuids():
            self.remove_alignment(uuid)

        # Groups/matrices not caught by orphan collection (e.g. no alignments)
        for group in self.get_all_groups():
            self.remove_group(group.uuid)
        for annotation in self.get_all_annotations():
            self.remove_annotation(annotation.analyte_uuid)
        for matrix in self.get_all_matrices():
            self.remove_matrix(matrix.uuid)

        for uuid in self.get_all_sample_uuids():
            self.remove_sample(uuid)

    def sample_count(self) -> int:
        return len(self._samples)

    def notify_sample_updated(
        self,
        uuid: 'SampleUUID',
    ):
        """
        Emits sigSampleUpdated
        """
        self.sigSampleUpdated.emit(
            self.get_sample(uuid)
        )

    def update_sample_metadata(
        self,
        uuid: 'SampleUUID',
        metadata: dict[str, any],
    ):
        """
        Given a UUID and a metadata dictionary, updates the
        sample's "metadata" attribute and signals that it was
        changed.
        """
        sample = self.get_sample(uuid)

        for key, value in metadata.items():
            sample.metadata[key] = value

        self.sigSampleUpdated.emit(sample)

    # --- Alignment methods ---

    def register_alignment(
        self,
        alignment: 'EnsembleAlignment',
    ):
        self._alignments[alignment.uuid] = alignment
        self.sigAlignmentAdded.emit(alignment)

    def remove_alignment(
        self,
        uuid: 'AlignmentUUID',
    ):
        if uuid in self._alignments:
            alignment = self._alignments[uuid]
            self.sigAlignmentRemoved.emit(alignment)
            del self._alignments[uuid]
            self._collect_orphans()

    def _collect_orphans(self):
        """
        Drop every group / annotation / similarity matrix none of whose analytes appear
        in any remaining alignment (run after an alignment is removed).
        """
        live = {
            analyte.uuid
            for alignment in self._alignments.values()
            for analyte in alignment.analytes
        }
        for group in self.get_all_groups():
            if live.isdisjoint(group.analyte_uuids):
                self.remove_group(group.uuid)
        for annotation in self.get_all_annotations():
            if annotation.analyte_uuid not in live:
                self.remove_annotation(annotation.analyte_uuid)
        live_ids = {str(uuid) for uuid in live}
        for matrix in self.get_all_matrices():
            if live_ids.isdisjoint(matrix.ids):
                self.remove_matrix(matrix.uuid)

    def get_alignment(
        self,
        uuid: 'AlignmentUUID',
    ) -> Optional['EnsembleAlignment']:
        return self._alignments.get(uuid)

    def get_all_alignment_uuids(
        self,
    ) -> list['AlignmentUUID']:
        return list(self._alignments.keys())

    def alignment_count(self) -> int:
        return len(self._alignments)

    # --- Formula assignment methods ---

    def register_assignment(
        self,
        assignment: 'FormulaAssignment',
    ):
        """
        Registers a FormulaAssignment, keyed by its source UUID. Replaces any
        existing assignment for the same source (latest wins).
        """
        self._assignments[assignment.source_uuid] = assignment
        self.sigAssignmentAdded.emit(assignment)

    def remove_assignment(
        self,
        source_uuid: 'EnsembleUUID',
    ):
        if source_uuid in self._assignments:
            assignment = self._assignments[source_uuid]
            self.sigAssignmentRemoved.emit(assignment)
            del self._assignments[source_uuid]

    def get_assignment_for_source(
        self,
        source_uuid: 'EnsembleUUID',
    ) -> Optional['FormulaAssignment']:
        return self._assignments.get(source_uuid)

    def set_chosen_candidate(
        self,
        source_uuid: 'EnsembleUUID',
        chosen_idx: Optional[int],
        chosen_by: str = 'user',
    ):
        """
        Accept a different candidate of a registered assignment (None clears
        the choice), recording who chose it.
        """
        assignment = self._assignments.get(source_uuid)
        if assignment is None:
            raise KeyError(f"No formula assignment for source {source_uuid}")
        if chosen_idx is not None and not 0 <= chosen_idx < len(assignment.candidates):
            raise IndexError(
                f"Candidate {chosen_idx} out of range "
                f"({len(assignment.candidates)} candidates)"
            )

        assignment.chosen_idx = chosen_idx
        assignment.chosen_by = chosen_by if chosen_idx is not None else None
        self.sigAssignmentUpdated.emit(assignment)

    def user_chosen_sources(self) -> set['EnsembleUUID']:
        """Sources whose formula the user chose (auto runs must keep them)."""
        return {
            uuid for uuid, a in self._assignments.items() if a.user_chosen
        }

    def chosen_formulas(self) -> dict['EnsembleUUID', str]:
        """Accepted formula per source, for exports."""
        return {
            uuid: a.chosen.formula_str
            for uuid, a in self._assignments.items()
            if a.chosen is not None
        }

    def get_all_assignment_source_uuids(
        self,
    ) -> list['EnsembleUUID']:
        return list(self._assignments.keys())

    def assignment_count(self) -> int:
        return len(self._assignments)

    # --- Analyte annotation methods ---

    def register_annotation(
        self,
        annotation: 'AnalyteAnnotation',
    ):
        """Add, or replace its analyte's existing annotation."""
        existed = annotation.analyte_uuid in self._annotations
        self._annotations[annotation.analyte_uuid] = annotation
        if existed:
            self.sigAnnotationUpdated.emit(annotation)
        else:
            self.sigAnnotationAdded.emit(annotation)

    def remove_annotation(
        self,
        analyte_uuid: 'AnalyteUUID',
    ):
        if analyte_uuid in self._annotations:
            self.sigAnnotationRemoved.emit(self._annotations.pop(analyte_uuid))

    def get_annotation(
        self,
        analyte_uuid: 'AnalyteUUID',
    ) -> Optional['AnalyteAnnotation']:
        return self._annotations.get(analyte_uuid)

    def get_all_annotations(self) -> list['AnalyteAnnotation']:
        return list(self._annotations.values())

    # --- Analyte group methods ---

    def register_group(
        self,
        group: 'AnalyteGroup',
    ):
        self._groups[group.uuid] = group
        self._rebuild_group_index()
        self.sigGroupAdded.emit(group)

    def remove_group(
        self,
        uuid: 'GroupUUID',
    ):
        if uuid in self._groups:
            group = self._groups.pop(uuid)
            self._rebuild_group_index()
            self.sigGroupRemoved.emit(group)

    def get_group(
        self,
        uuid: 'GroupUUID',
    ) -> Optional['AnalyteGroup']:
        return self._groups.get(uuid)

    def get_all_groups(self) -> list['AnalyteGroup']:
        return list(self._groups.values())

    def notify_group_updated(
        self,
        group: 'AnalyteGroup',
    ):
        """
        Groups are edited in place; every edit path must call this so the
        index and subscribed views refresh.
        """
        self._rebuild_group_index()
        self.sigGroupUpdated.emit(group)

    def groups_for_analytes(
        self,
        analyte_uuids: Iterable['AnalyteUUID'],
    ) -> list['AnalyteGroup']:
        """Groups with any member in `analyte_uuids`, in registration order."""
        hits = set()
        for uuid in analyte_uuids:
            hits |= self._groups_by_analyte.get(uuid, set())
        return [g for u, g in self._groups.items() if u in hits]

    def group_titles(self) -> dict['AnalyteUUID', str]:
        """
        Titles of every group each analyte belongs to (oldest group first,
        joined by ';'), for exports. Untitled groups are skipped.
        """
        titles: dict['AnalyteUUID', list[str]] = {}
        for group in sorted(self._groups.values(), key=lambda g: g.created_at):
            if not group.title:
                continue
            for uuid in group.analyte_uuids:
                titles.setdefault(uuid, []).append(group.title)
        return {uuid: ';'.join(t) for uuid, t in titles.items()}

    # --- Similarity matrix methods ---

    def register_matrix(
        self,
        matrix: 'SimilarityMatrix',
    ):
        self._matrices[matrix.uuid] = matrix
        self.sigMatrixAdded.emit(matrix)

    def remove_matrix(
        self,
        uuid: int,
    ):
        if uuid in self._matrices:
            matrix = self._matrices.pop(uuid)
            self.sigMatrixRemoved.emit(matrix)

    def get_matrix(
        self,
        uuid: int,
    ) -> Optional['SimilarityMatrix']:
        return self._matrices.get(uuid)

    def get_all_matrices(self) -> list['SimilarityMatrix']:
        return list(self._matrices.values())

    def matrices_for_analytes(
        self,
        analyte_uuids: Iterable['AnalyteUUID'],
    ) -> list['SimilarityMatrix']:
        """Matrices sharing any id with `analyte_uuids`, in registration order."""
        ids = {str(uuid) for uuid in analyte_uuids}
        return [m for m in self._matrices.values() if not ids.isdisjoint(m.ids)]

    def find_matrix(
        self,
        provenance: dict,
        alignment: 'EnsembleAlignment',
    ) -> Optional['SimilarityMatrix']:
        """
        A registered matrix computed the same way (`provenance` source and
        params match) that covers every analyte of `alignment` (each is in
        its ids or uncovered), for reuse instead of recomputing.
        """
        needed = {str(a.uuid) for a in alignment.analytes}
        for matrix in self._matrices.values():
            if (
                matrix.provenance.get('source') == provenance.get('source')
                and matrix.provenance.get('params') == provenance.get('params')
                and needed <= set(matrix.ids).union(matrix.uncovered)
            ):
                return matrix
        return None

    def _rebuild_group_index(self):
        self._groups_by_analyte = {}
        for group in self._groups.values():
            for uuid in group.analyte_uuids:
                self._groups_by_analyte.setdefault(uuid, set()).add(group.uuid)

def _merge_is_valid(
    source: 'Sample',
    destination: 'Sample',
) -> bool:
    source_pattern = (
        source.fingerprint is not None,
        source.injection is not None,
    )

    destination_pattern = (
        destination.fingerprint is not None,
        destination.injection is not None,
    )

    match (source_pattern, destination_pattern):
        case (
        ( (True, False),
          (False, True) )
        |
        ( (False, True),
          (True, False) )
        ):
            return True

        case _:
            return False


