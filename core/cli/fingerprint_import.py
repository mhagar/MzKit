"""
Imports fingerprint .csv's as Sample objects
"""
import pandas as pd

from core.utils.filesystem import all_filepaths_exist
from core.data_structs import Sample, Fingerprint

import argparse
import logging
# TESTING
import time
from datetime import timedelta

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.data_structs.fingerprint import FingerprintImportParams


# Set up logger for this module
logger = logging.getLogger(__name__)


def csv_to_fingerprint(
        params: 'FingerprintImportParams',
        verbose: bool = False,
) -> list[Sample]:
    """
    Given an .mzML file, reads and converts into ScanArrays

    :param params:
    core.data_structs.fingerprint.FingerprintImportParams object
    :param verbose: Whether to print verbose output
    :return: Dictionary with processing results
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    start = time.perf_counter()

    df = pd.read_csv(
        params.csv_path,
        index_col=0,
    )

    if not params.samples_in_rows:
        df = df.T

    if params.descriptors:
        df = df[params.descriptors]

    if params.sample_names:
        df = df.loc[params.sample_names]

    # Iterate over dataframe and construct Fingerprints
    samples: list[Sample] = []
    for samplename, array in df.iterrows():
        samplename: str
        array: pd.Series
        
        fingerprint = Fingerprint(
            array=array.to_numpy(),  # type: ignore
            descriptors=list(df.columns),
        )

        sample = Sample(
            name=samplename,
            fingerprint=fingerprint,
        )

        samples.append(sample)

    duration = timedelta(
        seconds=time.perf_counter() - start
    )
    print(f"Elapsed: {duration}")
    logger.info(
        f"Imported fingerprints for {len(samples)} samples."
    )

    return samples


# This function is called by ProcessRunner
def main(
        params: 'FingerprintImportParams',
        verbose: bool = False,
) -> list[Sample]:
    """
    Main entry point for both CLI and program use

    :param params: core.data_structs.fingerprint.FingerprintImportParams object
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

    _validate(params)

    # Generate Sample objects
    samples: list[Sample] = csv_to_fingerprint(
        params=params,
        verbose=verbose,
    )

    return samples


def _validate(params: 'FingerprintImportParams'):
    if not all_filepaths_exist([params.csv_path]):
        raise FileNotFoundError(
            "At least one of selected files is missing"
        )
    # if len(params.sample_names) != len(set(params.sample_names)):
    #     raise ValueError(
    #         "sample_names contains duplicates"
    #     )


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