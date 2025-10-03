"""
Implementation of project persistence
(i.e. saving/loading workspaces)
"""
from core.data_structs import (
    DataRegistry,
    Sample,
    Injection,
    Fingerprint,
    ScanArray,
)
from core.data_structs.scan_array import ScanArrayParameters

import logging
import zipfile
import pickle
import json
from pathlib import Path

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
    sample,
    savepath,
    zf,
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
    sample,
    savepath,
    zf,
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
    sample,
    savepath,
    zf,
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
    sample,
    savepath,
    zf,
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
    return injection


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


def deserialize_empty_sample(sample_primitives_path, zf):
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
