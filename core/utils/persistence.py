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
    IonAnnotation,
    IonPairAnnotation,
    MzDiffAnnotation,
)

import logging
import zipfile
import pickle
import json
from pathlib import Path
from dataclasses import asdict

from find_mfs import FormulaCandidate
from molmass import Formula

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass

__version__ = "1.0.0"
logger = logging.getLogger(__name__)

def save_project(
    filepath: Path,
    data_registry: 'DataRegistry',
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

    logger.info(
        f"Project saved: {filepath.absolute()}"
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

        # Convert ion_annots dict values from dataclass to dict
        serialized_ion_annots = {}
        for key, ion_annot in ensemble.ion_annots.items():
            annot_dict = asdict(ion_annot)
            # Convert FormulaCandidate to a serializable format
            annot_dict['formula'] = {
                'formula_str': str(ion_annot.formula.formula),
                'error_ppm': ion_annot.formula.error_ppm,
            }
            serialized_ion_annots[key] = annot_dict

        # Convert ion_pair_annots list from dataclass to dict
        serialized_ion_pair_annots = []
        for ion_pair_annot in ensemble.ion_pair_annots:
            annot_dict = asdict(ion_pair_annot)
            # Convert Formula to string
            annot_dict['formula_diff'] = str(ion_pair_annot.formula_diff)
            serialized_ion_pair_annots.append(annot_dict)

        # Convert mz_diffs to dicts (these should be fine but convert to be safe)
        serialized_mz_diffs = [asdict(mz_diff) for mz_diff in ensemble.mz_diffs]

        # Must be serialized without injection reference
        ensemble_dict = {
            'uuid': ensemble.uuid,
            'ms1_cofeatures': ensemble.ms1_cofeatures,
            'ms2_cofeatures': ensemble.ms2_cofeatures,
            'mz_diffs': serialized_mz_diffs,
            'ion_annots': serialized_ion_annots,
            'ion_pair_annots': serialized_ion_pair_annots,
            # User-editable properties
            'proposed_formula': ensemble.proposed_formula,
            'identity': ensemble.identity,
            'user_metadata': ensemble.user_metadata,
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
) -> list[Sample]:
    """
    Given a path to an .mzk file generate dusing save_project(),
    reconstitutes a list of Injections and Fingerprints
    :param filepath:
    :return:
    """
    _sanity_checks(filepath)

    samples: list[Sample] = []

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

    logger.info(
        f"Loaded {len(samples)} samples."
    )

    return samples


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
        )
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

    print('hi')
    print(f"ensemble_data:\n"
          f" {ensemble_data}")

    for e_dict in ensemble_data:
        # Reconstruct ion_annots from serialized dicts
        reconstructed_ion_annots = {}
        for key, annot_dict in e_dict['ion_annots'].items():
            # Reconstruct FormulaCandidate
            formula_data = annot_dict['formula']
            formula_candidate = FormulaCandidate(
                formula=Formula(formula_data['formula_str']),
                error_ppm=formula_data.get('error_ppm'),
                error_da=formula_data.get('error_da'),
                rdbe=formula_data.get('rdbe'),
            )

            # Reconstruct IonAnnotation dataclass
            ion_annot = IonAnnotation(
                cofeature_idxs=annot_dict['cofeature_idxs'],
                ms_level=annot_dict['ms_level'],
                formula=formula_candidate,
                uuid=annot_dict['uuid'],
                user_label=annot_dict.get('user_label'),
            )
            reconstructed_ion_annots[key] = ion_annot

        # Reconstruct ion_pair_annots from serialized dicts
        reconstructed_ion_pair_annots = []
        for annot_dict in e_dict['ion_pair_annots']:
            # Reconstruct Formula from string
            formula_diff = Formula(annot_dict['formula_diff'])

            # Reconstruct IonPairAnnotation dataclass
            ion_pair_annot = IonPairAnnotation(
                ion_a_uuid=annot_dict['ion_a_uuid'],
                ion_b_uuid=annot_dict['ion_b_uuid'],
                relationship=annot_dict['relationship'],
                formula_diff=formula_diff,
                user_label=annot_dict.get('user_label'),
            )
            reconstructed_ion_pair_annots.append(ion_pair_annot)

        # Reconstruct mz_diffs from serialized dicts
        reconstructed_mz_diffs = [
            MzDiffAnnotation(**mz_diff_dict)
            for mz_diff_dict in e_dict['mz_diffs']
        ]

        # Instantiate ensemble with reconstructed annotations
        ensemble = Ensemble(
            uuid=e_dict['uuid'],
            ms1_cofeatures=e_dict['ms1_cofeatures'],
            ms2_cofeatures=e_dict['ms2_cofeatures'],
            mz_diffs=reconstructed_mz_diffs,
            ion_annots=reconstructed_ion_annots,
            ion_pair_annots=reconstructed_ion_pair_annots,
            # User-editable properties (with defaults for backward compat)
            proposed_formula=e_dict.get('proposed_formula'),
            identity=e_dict.get('identity'),
            user_metadata=e_dict.get('user_metadata', {}),
        )

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
