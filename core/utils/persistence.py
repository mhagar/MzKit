"""
Implementation of project persistence
(i.e. saving/loading workspaces)
"""
from core.data_structs import (
    DataRegistry,
    Sample,
    Injection,
    Ensemble,
    Fingerprint,
    ScanArray,
)
from core.data_structs.scan_array import ScanArrayParameters
from core.data_structs.ensemble import (
    AdductNetwork,
    EnsembleAnnotation,
    GenericAnnotation,
    IonAnnotation,
    IonPairAnnotation,
    MzDiffAnnotation,
)
from core.data_structs.alignment import (
    EnsembleAlignment,
    AlignedAnalyte,
    AlignmentParams,
)
from core.data_structs.formula_assignment import (
    FormulaAssignment,
    AssignedCandidate,
)
from core.data_structs.ensemble_group import EnsembleGroup
from core.data_structs.similarity_matrix import ImportParams, SimilarityMatrix

import logging
import zipfile
import pickle
import io
import json
from pathlib import Path
from dataclasses import asdict

import numpy as np

from typing import NamedTuple, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    pass

__version__ = "2.0.0"
logger = logging.getLogger(__name__)


class ProjectData(NamedTuple):
    """Everything load_project() reconstitutes from an .mzk file."""
    samples: list[Sample]
    alignments: list[EnsembleAlignment]
    groups: list[EnsembleGroup]
    matrices: list[SimilarityMatrix]

def save_project(
    filepath: Path,
    data_registry: 'DataRegistry',
    progress_callback=None,  # injected by ProcessRunner; unused here
    cancel_event=None,       # injected by ProcessRunner; unused here
) -> None:
    """
    Serializes the Sample objects, then stores in a zip file.
    """
    with zipfile.ZipFile(
        file=filepath,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        project_metadata = {
            'format_version': __version__,
            # 'embedded_mzml': [],
        }

        zf.writestr(
            "project_metadata.json",
            data=json.dumps(
                project_metadata,
                indent=2,
            )
        )

        samples: list[Sample] = data_registry.get_all_samples()
        logger.info(
            f"Packaging {len(samples)} Samples"
        )
        for sample in samples:
            savepath = f"samples/{sample.name}"

            logger.debug(
                f"Packaging: {sample}"
            )

            serialize_sample_primitives(
                sample,
                savepath,
                zf,
            )

            if sample.injection:
                serialize_injection_primitives(
                    sample,
                    savepath,
                    zf,
                )
                serialize_injection_scanarrays(
                    sample,
                    savepath,
                    zf,
                )
                serialize_injection_ensembles(
                    sample,
                    savepath,
                    zf,
                )

            if sample.fingerprint:
                serialize_fingerprint_primitives(
                    sample,
                    savepath,
                    zf,
                )
                serialize_fingerprint_arrays(
                    sample,
                    savepath,
                    zf,
                )

        # Serialize alignments
        alignments = [
            data_registry.get_alignment(uuid)
            for uuid in data_registry.get_all_alignment_uuids()
        ]
        if alignments:
            logger.info(
                f"Packaging {len(alignments)} alignments"
            )
            for alignment in alignments:
                serialize_alignment(alignment, zf)

        # Serialize analyte groups
        groups = data_registry.get_all_groups()
        if groups:
            logger.info(
                f"Packaging {len(groups)} analyte groups"
            )
            for group in groups:
                serialize_group(group, zf)

        # Serialize similarity matrices (computed and imported, raw values)
        matrices = data_registry.get_all_matrices()
        if matrices:
            logger.info(
                f"Packaging {len(matrices)} similarity matrices"
            )
            for matrix in matrices:
                serialize_matrix(matrix, zf)

    logger.info(
        f"Project saved: {filepath.absolute()}"
    )


def serialize_alignment(
    alignment: 'EnsembleAlignment',
    zf: 'zipfile.ZipFile',
):
    analytes_data = []
    for analyte in alignment.analytes:
        analytes_data.append({
            'uuid': analyte.uuid,
            'ensemble_map': {
                str(k): v for k, v in analyte.ensemble_map.items()
            },
            'consensus_rt': analyte.consensus_rt,
            'consensus_mz': analyte.consensus_mz,
        })

    alignment_dict = {
        'uuid': alignment.uuid,
        'name': alignment.name,
        'sample_uuids': list(alignment.sample_uuids),
        'parameters': alignment.parameters._asdict(),
        'analytes': analytes_data,
    }

    zf.writestr(
        f"alignments/{alignment.uuid}.json",
        data=json.dumps(alignment_dict, indent=2),
    )


def deserialize_alignment(
    alignment_path: str,
    zf: 'zipfile.ZipFile',
) -> 'EnsembleAlignment':
    data = json.loads(zf.read(alignment_path))

    analytes = []
    for a in data['analytes']:
        analytes.append(AlignedAnalyte(
            uuid=a['uuid'],
            ensemble_map={int(k): v for k, v in a['ensemble_map'].items()},
            consensus_rt=a['consensus_rt'],
            consensus_mz=a['consensus_mz'],
        ))

    return EnsembleAlignment(
        uuid=data['uuid'],
        name=data.get('name', ''),
        sample_uuids=tuple(data['sample_uuids']),
        # Drop keys of retired params (e.g. pre-MS1-only ms2_* / weights)
        parameters=AlignmentParams(**{
            k: v for k, v in data['parameters'].items()
            if k in AlignmentParams._fields
        }),
        analytes=analytes,
    )


def _generic_annot(d: dict) -> GenericAnnotation:
    # Pre-2.0 files: one `cofeature_idx`, and a `source` key
    d = {k: v for k, v in d.items() if k != 'source'}
    if 'cofeature_idx' in d:
        d['cofeature_idxs'] = [d.pop('cofeature_idx')]
    return GenericAnnotation(**d)


def deserialize_assignment(data: Optional[dict]) -> Optional['FormulaAssignment']:
    """
    Inverse of asdict(FormulaAssignment): it's primitive-only
    (AssignedCandidate included), so asdict captures every score term losslessly
    """
    if data is None:
        return None
    data = dict(data)  # don't mutate the source
    candidates = [AssignedCandidate(**c) for c in data.pop('candidates')]
    return FormulaAssignment(candidates=candidates, **data)


def serialize_group(
    group: 'EnsembleGroup',
    zf: 'zipfile.ZipFile',
):
    # Primitive-only, so asdict round-trips it
    zf.writestr(
        f"groups/{group.uuid}.json",
        data=json.dumps(asdict(group), indent=2),
    )


def deserialize_group(
    group_path: str,
    zf: 'zipfile.ZipFile',
) -> Optional['EnsembleGroup']:
    """None for a pre-2026-11 group of analytes (not migrated)."""
    data = json.loads(zf.read(group_path))
    if 'ensemble_uuids' not in data:
        return None
    return EnsembleGroup(**data)


def serialize_matrix(
    matrix: 'SimilarityMatrix',
    zf: 'zipfile.ZipFile',
):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, values=matrix.values)
    zf.writestr(f"similarity/{matrix.uuid}.npz", data=buffer.getvalue())
    matrix_dict = {
        'uuid': matrix.uuid,
        'name': matrix.name,
        'ids': list(matrix.ids),
        'provenance': matrix.provenance,
        # NamedTuple -> dict (json.dumps would write a bare list)
        'default_transform': matrix.default_transform._asdict(),
        'uncovered': list(matrix.uncovered),
    }
    zf.writestr(
        f"similarity/{matrix.uuid}.json",
        data=json.dumps(matrix_dict, indent=2),
    )


def deserialize_matrix(
    json_path: str,
    zf: 'zipfile.ZipFile',
) -> 'SimilarityMatrix':
    data = json.loads(zf.read(json_path))
    npz_path = json_path.removesuffix('.json') + '.npz'
    with np.load(io.BytesIO(zf.read(npz_path)), allow_pickle=False) as npz:
        values = npz['values']
    return SimilarityMatrix(
        name=data['name'],
        ids=tuple(data['ids']),
        values=values,
        provenance=data['provenance'],
        default_transform=ImportParams(**data['default_transform']),
        uncovered=tuple(data.get('uncovered', ())),
        uuid=data['uuid'],
    )


def serialize_fingerprint_arrays(
    sample: 'Sample',
    savepath: str,
    zf: 'zipfile.ZipFile',
):
    # Write arrays (Pickle is OK; numpy arrays)
    data = {
        'array':       sample.fingerprint.array,
        'descriptors': sample.fingerprint.descriptors,
    }
    zf.writestr(
        f"{savepath}/fingerprint_data.pkl",
        data=pickle.dumps(
            data,
        )
    )


def serialize_fingerprint_primitives(
    sample: 'Sample',
    savepath: str,
    zf: 'zipfile.ZipFile',
):
    # Write fingerprint primitives
    fp_metadata = {
        'uuid': sample.fingerprint.uuid
    }
    zf.writestr(
        f"{savepath}/fingerprint.json",
        data=json.dumps(
            fp_metadata,
            indent=2,
        ),
    )


def serialize_injection_scanarrays(
    sample: 'Sample',
    savepath: str,
    zf: 'zipfile.ZipFile',
):
    # Write ScanArrays. Using Pickle is OK here
    ms1_scan_array_dict = sample.injection.scan_array_ms1.__dict__
    zf.writestr(
        f"{savepath}/ms1_scan_array.pkl",
        data=pickle.dumps(
            ms1_scan_array_dict
        )
    )
    if sample.injection.scan_array_ms2:
        ms2_scan_array_dict = sample.injection.scan_array_ms2.__dict__
        zf.writestr(
            f"{savepath}/ms2_scan_array.pkl",
            data=pickle.dumps(
                ms2_scan_array_dict
            )
        )


def serialize_injection_primitives(
    sample: 'Sample',
    savepath: str,
    zf: 'zipfile.ZipFile',
):
    # Write injection primitives
    injection_metadata = {
        'filename':          sample.injection.filename,
        'ms1_scan_array_params': sample.injection.scan_array_parameters[0].__dict__,
        'ms2_scan_array_params': sample.injection.scan_array_parameters[1].__dict__ or None,
        'uuid':              sample.injection.uuid,
        'acquisition_mode':  sample.injection.acquisition_mode,
    }
    zf.writestr(
        f"{savepath}/injection.json",
        data=json.dumps(
            injection_metadata,
            indent=2,
        )
    )


def serialize_injection_ensembles(
    sample: 'Sample',
    savepath: str,
    zf: 'zipfile.ZipFile',
):
    """
    Serialize ensembles separately from injection objects
    """
    if not sample.injection.ensembles:
        return

    ensemble_data: list[dict] = []
    for ensemble in sample.injection.ensembles.values():
        # Must be serialized without injection reference
        ensemble_dict = {
            'uuid': ensemble.uuid,
            'ms1_cofeatures': ensemble.ms1_cofeatures,
            'ms2_cofeatures': ensemble.ms2_cofeatures,
            # Annotations are primitive-only dataclasses
            'mz_diffs': [asdict(a) for a in ensemble.mz_diffs],
            'ion_annots': {k: asdict(a) for k, a in ensemble.ion_annots.items()},
            'ion_pair_annots': [asdict(a) for a in ensemble.ion_pair_annots],
            'generic_annots': {
                k: asdict(a) for k, a in ensemble.generic_annots.items()
            },
            'formula_assignment': (
                asdict(ensemble.formula_assignment)
                if ensemble.formula_assignment is not None else None
            ),
            'adduct_network': (
                asdict(ensemble.adduct_network)
                if ensemble.adduct_network is not None else None
            ),
            'annotation': (
                asdict(ensemble.annotation)
                if ensemble.annotation is not None else None
            ),
            # User-editable properties
            'identity': ensemble.identity,
            'user_metadata': ensemble.user_metadata,
            # DDA precursor info
            'precursor_mz': ensemble.precursor_mz,
            'precursor_charge': ensemble.precursor_charge,
            'saturation_threshold': ensemble.saturation_threshold,
            # Injection reference not serialized - will be assigned on loading
        }

        ensemble_data.append(ensemble_dict)

    # Going to just use pickle for now. Lazy!!
    zf.writestr(
        f"{savepath}/ensembles.pkl",
        data=pickle.dumps(ensemble_data)
    )


def serialize_sample_primitives(
    sample: 'Sample',
    savepath: str,
    zf: 'zipfile.ZipFile',
):
    # Serialize simple data types
    sample_primitives = {
        k: v for k, v in sample.__dict__.items()
        if type(v) in [int, float, str, bool, type(None)]
    }

    sample_primitives['metadata'] = sample.metadata

    zf.writestr(
        f"{savepath}/primitives.json",
        data=json.dumps(
            sample_primitives,
            indent=2,
        )
    )



# def _get_mzml_paths(
#     injection_model: 'InjectionListModel',
# ) -> list[Path]:
#     mzml_paths: list[Path] = []
#     for injection in injection_model.getAllInjections():
#
#         mzml_path = Path(injection.exp.getLoadedFilePath())
#
#         if mzml_path not in mzml_paths:
#             mzml_paths.append(
#                 mzml_path
#             )
#
#     return mzml_paths


def load_project(
    filepath: Path,
    progress_callback=None,  # injected by ProcessRunner; unused here
    cancel_event=None,       # injected by ProcessRunner; unused here
) -> ProjectData:
    """
    Given a path to an .mzk file generated using save_project(),
    reconstitutes Samples (their ensembles carrying formula assignments),
    EnsembleAlignments and EnsembleGroups.

    :param filepath: Path to .mzk file
    :return: ProjectData(samples, alignments, groups, matrices)
    """
    _sanity_checks(filepath)

    samples: list[Sample] = []
    alignments: list[EnsembleAlignment] = []
    groups: list[EnsembleGroup] = []
    matrices: list[SimilarityMatrix] = []

    with zipfile.ZipFile(
        filepath,
        mode='r',
    ) as zf:
        # I don't do anything with project_metadata atm
        project_metadata: dict = json.loads(
            zf.read(
                'project_metadata.json'
            )
        )
        logger.info(
            f"Loading project from {filepath}, format version:"
            f"{project_metadata.get('format_version', 'unknown')}"
        )

        # Get directories of each Sample 'primitives.json' file
        sample_dirs: list[str] = [name for name in zf.namelist()
                          if name.startswith('samples/')
                          and name.endswith('/primitives.json')]

        for sample_primitives_path in sample_dirs:
            sample = deserialize_empty_sample(
                sample_primitives_path,
                zf,
            )
            loadpath = f"samples/{sample.name}"

            if f"{loadpath}/injection.json" in zf.namelist():
                # Sample has an Injection
                logger.debug(
                    f"Building Injection for Sample: {sample.name}"
                )
                injection = deserialize_injection(loadpath, zf)
                sample.set_injection(injection)

            if f"{loadpath}/fingerprint.json" in zf.namelist():
                # Sample has a Fingerprint
                logger.debug(
                    f"Building Fingerprint for Sample: {sample.name}"
                )
                fingerprint = deserialize_fingerprint(loadpath, zf)
                sample.set_fingerprint(fingerprint)

            samples.append(sample)

        # Load alignments
        alignment_paths = [
            name for name in zf.namelist()
            if name.startswith('alignments/')
            and name.endswith('.json')
        ]
        for alignment_path in alignment_paths:
            try:
                alignment = deserialize_alignment(alignment_path, zf)
            except KeyError:
                # Pre-1.1.0 alignments have no analyte uuids; these are
                # deliberately not migrated (re-align instead)
                logger.warning(
                    f"Skipping alignment {alignment_path}: saved by an older "
                    f"format version; please re-run the alignment."
                )
                continue
            alignments.append(alignment)

        # Pre-2026-11 files kept assignments in their own folder; not migrated
        if any(name.startswith('assignments/') for name in zf.namelist()):
            logger.warning(
                "This project's formula assignments were saved by an older "
                "format version and are not loaded; please re-run find-mfs."
            )

        # Load ensemble groups (absent before 1.4.0)
        group_paths = [
            name for name in zf.namelist()
            if name.startswith('groups/')
            and name.endswith('.json')
        ]
        for group_path in group_paths:
            group = deserialize_group(group_path, zf)
            if group is None:
                logger.warning(
                    f"Skipping group {group_path}: a group of analytes saved by "
                    f"an older format version; please re-create it."
                )
                continue
            groups.append(group)

        # Load similarity matrices (absent before 1.4.0)
        matrices = [
            deserialize_matrix(name, zf) for name in zf.namelist()
            if name.startswith('similarity/') and name.endswith('.json')
        ]

        # Pre-2026-11 files kept analyte annotations in their own folder
        if any(name.startswith('annotations/') for name in zf.namelist()):
            logger.warning(
                "This project's structure annotations were saved by an older "
                "format version and are not loaded."
            )

    logger.info(
        f"Loaded {len(samples)} samples, "
        f"{len(alignments)} alignments, "
        f"{len(groups)} groups, "
        f"{len(matrices)} similarity matrices."
    )

    return ProjectData(
        samples=samples,
        alignments=alignments,
        groups=groups,
        matrices=matrices,
    )


def deserialize_injection(
    loadpath: str,
    zf: 'zipfile.ZipFile',
) -> 'Injection':
    injection_primitives = json.loads(
        zf.read(f"{loadpath}/injection.json")
    )

    ms1_scan_array_params = ScanArrayParameters(
        **injection_primitives['ms1_scan_array_params']
    )

    ms2_scan_array_params = None
    if injection_primitives['ms2_scan_array_params']:
        ms2_scan_array_params = ScanArrayParameters(
            **injection_primitives['ms2_scan_array_params']
        )

    # Load MS1 scan array
    ms1_scan_array_dict: dict = pickle.loads(
        zf.read(
            f"{loadpath}/ms1_scan_array.pkl"
        )
    )
    ms1_scan_array = ScanArray(**ms1_scan_array_dict)

    # Load MS2 scan array (if it exists)
    ms2_scan_array = None
    ms2_path = f"{loadpath}/ms2_scan_array.pkl"
    if ms2_path in zf.namelist():
        ms2_scan_array_dict: dict = pickle.loads(
            zf.read(
                ms2_path
            )
        )
        ms2_scan_array = ScanArray(**ms2_scan_array_dict)

    # Finally, assemble into Injection object
    injection = Injection(
        scan_array_ms1=ms1_scan_array,
        scan_array_ms2=ms2_scan_array,
        filename=injection_primitives['filename'],
        uuid=injection_primitives['uuid'],
        scan_array_parameters=(
            ms1_scan_array_params,
            ms2_scan_array_params,
        ),
        acquisition_mode=injection_primitives.get('acquisition_mode', 'ms1_only'),
    )

    # Load Ensembles (if they exist)
    deserialize_injection_ensembles(
        injection=injection,
        loadpath=loadpath,
        zf=zf,
    )

    return injection


def deserialize_injection_ensembles(
    injection: 'Injection',
    loadpath: str,
    zf: 'zipfile.ZipFile',
):
    """
    Loads ensembles and assigns them to the Injection object.
    """
    ensembles_path = f"{loadpath}/ensembles.pkl"
    if ensembles_path not in zf.namelist():
        return

    ensemble_data: list[dict] = pickle.loads(zf.read(ensembles_path))

    for e_dict in ensemble_data:
        ensemble = Ensemble(
            uuid=e_dict['uuid'],
            ms1_cofeatures=e_dict['ms1_cofeatures'],
            ms2_cofeatures=e_dict['ms2_cofeatures'],
            mz_diffs=[MzDiffAnnotation(**d) for d in e_dict['mz_diffs']],
            ion_annots={
                k: IonAnnotation(**d) for k, d in e_dict['ion_annots'].items()
            },
            ion_pair_annots=[
                IonPairAnnotation(**d) for d in e_dict['ion_pair_annots']
            ],
            generic_annots={
                k: _generic_annot(d)
                for k, d in e_dict['generic_annots'].items()
                # find-mfs adduct labels now live in adduct_network
                if d.get('source') != 'auto_adduct'
            },
            formula_assignment=deserialize_assignment(
                e_dict.get('formula_assignment')
            ),
            adduct_network=(
                AdductNetwork(**e_dict['adduct_network'])
                if e_dict.get('adduct_network') else None
            ),
            annotation=EnsembleAnnotation.from_dict(e_dict.get('annotation')),
            # User-editable properties (with defaults for backward compat)
            identity=e_dict.get('identity'),
            user_metadata=e_dict.get('user_metadata', {}),
            # DDA precursor info (None for pre-DDA .mzk files)
            precursor_mz=e_dict.get('precursor_mz'),
            precursor_charge=e_dict.get('precursor_charge'),
        )
        # Pre-saturation .mzk files keep the config default
        if 'saturation_threshold' in e_dict:
            ensemble.saturation_threshold = e_dict['saturation_threshold']

        # This will call ensemble.set_injection():
        injection.add_ensemble(ensemble)


def deserialize_fingerprint(
    loadpath: str,
    zf: 'zipfile.ZipFile',
) -> 'Fingerprint':
    fingerprint_primitives = json.loads(
        zf.read(f"{loadpath}/fingerprint.json")
    )

    fp_data = pickle.loads(
        zf.read(
            f"{loadpath}/fingerprint_data.pkl"
        )
    )

    # Assemble into Fingerprint object
    fingerprint = Fingerprint(
        **fingerprint_primitives,
        array=fp_data['array'],
        descriptors=fp_data['descriptors'],
    )

    return fingerprint


def deserialize_empty_sample(
    sample_primitives_path: str,
    zf: 'zipfile.ZipFile',
):
    # Load primitives
    sample_primitives = json.loads(
        zf.read(sample_primitives_path)
    )
    sample = Sample(**sample_primitives)
    return sample


def _sanity_checks(
    filepath: Path,
) -> None:
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    if not filepath.suffix.lower() == '.mzk':
        raise ValueError(
            f"Invalid filetype: {filepath} \n"
            f"Should be .mzk file"
        )
