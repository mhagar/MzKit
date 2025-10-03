"""
Implementation of project persistence
(i.e. saving/loading workspaces)
"""
import pyopenms as oms

from core.data_structs import Injection, Fingerprint, ScanArray

import logging
import zipfile
import pickle
import json
import tempfile
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.models import FingerprintListModel, InjectionListModel

__version__ = "1.0.0"
logger = logging.getLogger(__name__)

def save_project(
    filepath: Path,
    injection_model: 'InjectionListModel',
    fingerprint_model: 'FingerprintListModel',
) -> None:
    """
    Serializes the Injection and Fingerprint objects,
    then stores it in a zip file.
    """
    with zipfile.ZipFile(
        file=filepath,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        metadata = {
            'format_version': __version__,
            'embedded_mzml': [],
        }

        # Embed all mzML files
        mzml_paths = _get_mzml_paths(injection_model)
        logger.info(
            f"Embedding {len(mzml_paths)} .mzML files"
        )
        for mzml_path in mzml_paths:
            zf.write(
                mzml_path,
                Path("mzml_files") / mzml_path.name
            )
            metadata['embedded_mzml'].append(
                {
                    'filename': mzml_path.name,
                    'size_bytes': mzml_path.stat().st_size,
                }
            )

        zf.writestr(
            "metadata.json",
            data=json.dumps(
                metadata,
                indent=2,
            )
        )

        # Save Injection objects
        injections: list[Injection] = injection_model.getAllInjections()
        logger.info(
            f"Packaging {len(injections)} Injections"
        )
        for injection in injections:
            uuid = injection.uuid

            logger.debug(
                f"Packaging Injection uuid: {uuid}"
            )

            # Write metadata
            inj_metadata = {
                k: v for k, v in injection.__dict__.items()
                if type(v) in [int, float, str, type(None)]
            }

            zf.writestr(
                f"injections/{uuid}/metadata.json",
                data=json.dumps(
                    inj_metadata,
                    indent=2,
                ),
            )

            # Write ScanArrays. Using pickle is OK with numpy arrays
            ms1_scan_array_dict = injection.scan_array_ms1.__dict__
            zf.writestr(
                f"injections/{uuid}/ms1_scan_array.pkl",
                data=pickle.dumps(
                    ms1_scan_array_dict
                )
            )

            if injection.scan_array_ms2:
                ms2_scan_array_dict = injection.scan_array_ms2.__dict__
                zf.writestr(
                    f"injections/{uuid}/ms2_scan_array.pkl",
                    data=pickle.dumps(
                        ms2_scan_array_dict
                    )
                )


        # Save Fingerprint objects
        fingerprints: list[Fingerprint] = fingerprint_model.getAllFingerprints()
        logger.info(
            f"Packaging {len(fingerprints)} Fingerprints"
        )
        for fingerprint in fingerprints:
            uuid = fingerprint.uuid

            logger.debug(
                f"Packaging Fingerprint uuid: {uuid}"
            )

            # Write metadata
            fp_metadata = {
                k: v for k, v in fingerprint.__dict__.items()
                if type(v) in [int, float, str, type(None)]
            }

            zf.writestr(
                f"fingerprints/{uuid}/metadata.json",
                data=json.dumps(
                    fp_metadata,
                    indent=2,
                ),
            )

            # Write other data
            data = {
                'array': fingerprint.array,
                'descriptors': fingerprint.descriptors,
            }

            zf.writestr(
                f"fingerprints/{uuid}/fingerprint_data.pkl",
                data=pickle.dumps(
                    data,
                )
            )

    logger.info(
        f"Project saved: {filepath.absolute()}"
    )


def _get_mzml_paths(
    injection_model: 'InjectionListModel',
) -> list[Path]:
    mzml_paths: list[Path] = []
    for injection in injection_model.getAllInjections():

        mzml_path = Path(injection.exp.getLoadedFilePath())

        if mzml_path not in mzml_paths:
            mzml_paths.append(
                mzml_path
            )

    return mzml_paths


def load_project(
    filepath: Path,
) -> tuple[list[Injection], list[Fingerprint]]:
    """
    Given a path to an .mkz file generate dusing save_project(),
    reconstitutes a list of Injections and Fingerprints
    :param filepath:
    :return:
    """
    _sanity_checks(filepath)

    injections: list[Injection] = []
    fingerprints: list[Fingerprint] = []

    with zipfile.ZipFile(
        filepath,
        mode='r',
    ) as zf:
        metadata: dict = json.loads(
            zf.read(
                'metadata.json'
            )
        )

        logger.info(
            f"Reading project containing {len(metadata['embedded_mzml'])} "
            f".mzML files"
        )

        # Create temp directory to extract .mzML files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path =Path(temp_dir)

            mzml_name_to_path: dict[str, Path] = {}
            for mzml_info in metadata['embedded_mzml']:
                filename = mzml_info['filename']
                temp_mzml_path = temp_path / filename
                with open(temp_mzml_path, 'wb') as f:

                    embedded_path = f"mzml_files/{filename}"
                    f.write(
                        zf.read(embedded_path)
                    )

                mzml_name_to_path[filename] = temp_mzml_path

            # Load Injection objects
            injection_dirs: list[str] = [name for name in zf.namelist()
                              if name.startswith('injections/')
                              and name.endswith('/metadata.json')]

            for inj_metadata_path in injection_dirs:
                uuid = inj_metadata_path.split('/')[1]

                # Load metadata
                inj_metadata = json.loads(
                    zf.read(inj_metadata_path)
                )

                logger.debug(
                    f"Reading ScanArrays for Injection uuid: {uuid}"
                )
                # Load MS1 scan array
                ms1_scan_array_dict: dict = pickle.loads(
                    zf.read(
                        f"injections/{uuid}/ms1_scan_array.pkl"
                    )
                )
                ms1_scan_array = ScanArray(**ms1_scan_array_dict)

                # Load MS2 scan array (if it exists)
                ms2_scan_array = None
                ms2_path = f"injections/{uuid}/ms2_scan_array.pkl"
                if ms2_path in zf.namelist():
                    ms2_scan_array_dict: dict = pickle.loads(
                        zf.read(
                            ms2_path
                        )
                    )
                    ms2_scan_array = ScanArray(**ms2_scan_array_dict)

                # Reconstruct MSExperiment object
                logger.debug(
                    f"Reading .mzMLs for Injection uuid: {uuid}"
                )
                mzml_path = mzml_name_to_path[
                    inj_metadata['filename']
                ]
                exp = oms.MSExperiment()
                oms.MzMLFile().load(
                    str(mzml_path),
                    exp,
                )

                # Finally, assemble into Injection object
                injection = Injection(
                    exp=exp,
                    scan_array_ms1=ms1_scan_array,
                    scan_array_ms2=ms2_scan_array,
                    **inj_metadata,  # Other args
                )

                injections.append(injection)


            # Load Fingerprint objects
            fingerprint_dirs: list[str] = [name for name in zf.namelist()
                               if name.startswith('fingerprints/')
                               and name.endswith('/metadata.json')]

            for fp_metadata_path in fingerprint_dirs:
                uuid = fp_metadata_path.split('/')[1]

                # Load metadata
                fp_metadata = json.loads(
                    zf.read(
                        fp_metadata_path
                    )
                )

                # Load fingerprint data
                fp_data = pickle.loads(
                    zf.read(
                        f"fingerprints/{uuid}/fingerprint_data.pkl"
                    )
                )

                # Finally, assemble into Fingerprint object
                fingerprint = Fingerprint(
                    array=fp_data['array'],
                    descriptors=fp_data['descriptors'],
                    **fp_metadata,
                )

                fingerprints.append(fingerprint)

    logger.info(
        f"Loaded {len(injections)} injections, {len(fingerprints)} fingerprints"
    )

    return injections, fingerprints



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
