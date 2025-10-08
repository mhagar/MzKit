"""
Imports metadata in a .csv and writes it into Samples with the same name
"""
from core.utils.import_sample_metadata import read_metadata_csv

from pathlib import Path
import argparse
import logging

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.data_structs import Sample, SampleUUID

# Set up logger for this module
logger = logging.getLogger(__name__)

# This function is called by ProcessRunner
def main(
    csv_filepath: Path,
    samplename_column: str,
    metadata_columns: list[str],
    samples: list['Sample'],
    verbose: bool = False,
) -> dict[ 'SampleUUID', dict[str, any] ]:
    """
    Main entry point for both CLI and program use
    :return:
    """

    # Configure logging for CLI mode
    if __name__ == "__main__":
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )

    results = read_metadata_csv(
        csv_filepath=Path(csv_filepath),
        samplename_column=samplename_column,
        metadata_columns=metadata_columns,
        samples=samples,
    )

    return results


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