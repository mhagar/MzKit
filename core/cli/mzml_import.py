"""
Imports .mzML files as Sample objects
"""
import pyopenms as oms

from core.utils.filesystem import all_filepaths_exist
from core.data_structs.injection import Injection
from core.data_structs.sample import Sample

import argparse
import logging
from pathlib import Path
# TESTING
import time

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from core.data_structs.scan_array import ScanArrayParameters


# Set up logger for this module
logger = logging.getLogger(__name__)


def mzml_to_injection(
        input_filepath: Path,
        scan_array_params: tuple[
            'ScanArrayParameters', Optional['ScanArrayParameters']
        ],
        verbose: bool = False,
) -> Injection:
    """
    Given an .mzML file, reads and converts into ScanArrays

    :param input_filepath: Path to input file
    :param scan_array_params:
    :param verbose: Whether to print verbose output

    :return: Dictionary with processing results
    """
    if verbose:
        logger.setLevel(logging.DEBUG)



    exp = oms.MSExperiment()
    oms.MzMLFile().load(
        str(input_filepath),
        exp,
    )

    injection = Injection(
        exp=exp,
        filename=input_filepath.name,
        scan_array_parameters=scan_array_params,
    )

    return injection


# This function is called by ProcessRunner
def main(
        input_filepaths: str | list[str],
        sample_names: str | list[str],
        scan_array_params: tuple[
            'ScanArrayParameters', Optional['ScanArrayParameters']
        ],
        verbose: bool = False,
) -> list[Sample]:
    """
    Main entry point for both CLI and program use

    :param input_filepaths: Either a list of file paths, or a single path
    :param sample_names: Either a list of sample names, or a single sample name
    :param scan_array_params: tuple of two ScanArrayParameters, one for MS1 and
    one for MS2. If None is passed for MS2, only imports MS1 data
    :param verbose:
    :return:
    """
    # Configure logging for CLI mode
    if __name__ == "__main__":
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )

    # Handle being given a single file:
    if isinstance(input_filepaths, str):
        input_filepaths = [input_filepaths]
        sample_names = [sample_names]

    input_filepaths = [Path(x) for x in input_filepaths]
    sample_names = [x for x in sample_names]

    # Validate inputs
    _validate(input_filepaths, sample_names)

    # Generate Sample objects
    samples: list[Sample] = []
    for sample_name, filepath in zip(
        sample_names,
        input_filepaths,
    ):
        try:

            logger.info(
                f"Importing {filepath.name}"
            )
            t0 = time.perf_counter()

            injection = mzml_to_injection(
                    input_filepath=filepath,
                    scan_array_params=scan_array_params,
                )

            samples.append(
                Sample(
                    name=sample_name,
                    injection=injection,
                )
            )

            t1 = time.perf_counter()
            logger.info(
                f"Done ({t1 - t0:.1f} sec)"
            )


        except ValueError as e:
            logger.warning(
                f"Error processing {sample_name}: \n"
                f"{e}"
            )

    return samples


def _validate(input_filepaths, sample_names):
    if not all_filepaths_exist(input_filepaths):
        raise FileNotFoundError(
            "At least one of selected files is missing"
        )
    if len(input_filepaths) != len(sample_names):
        raise ValueError(
            f"input_filepaths and sample_names are not the same length. "
            f"({len(input_filepaths)} vs {len(sample_names)}"
        )
    if len(input_filepaths) != len(set(input_filepaths)):
        raise ValueError(
            "input_filepaths contains duplicates"
        )
    if len(sample_names) != len(set(sample_names)):
        raise ValueError(
            "sample_names contains duplicates"
        )


# This part only runs when the script is executed directly from the CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example data processing script",
    )
    parser.add_argument(
        "--input-file", "-i",
        help="Path to input file",
    )

    parser.add_argument(
        "--output-file", "-o",
        help="Path to output file",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # TODO: Implement a way to load arguments as 'ScanArrayParameters',
    # This script can't be used from CLI in current state

    result = main(
        input_filepaths=args.input_file,
        verbose=args.verbose,
    )