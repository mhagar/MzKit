"""
Imports an analyte_table.csv
"""
import pandas as pd

from pathlib import Path
import argparse
import logging

from core.data_structs import AnalyteTable

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass

# Set up logger for this module
logger = logging.getLogger(__name__)

# This function is called by ProcessRunner
def main(
    analyte_table_csv_filepath: Path,
    analyte_id_column: str,
    sample_name_columns: list[str],
    metadata_table_csv_filepath: Path,
    metadata_id_column: str,
    field_columns: list[str],
    verbose: bool = False,
) -> AnalyteTable:
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

    # Read in analyte table dataframe
    analyte_table_df = pd.read_csv(analyte_table_csv_filepath)
    analyte_table_df = analyte_table_df.set_index(
        analyte_id_column
    )

    sample_name_columns.append('m/z')
    sample_name_columns.append('rt')

    analyte_table_df = analyte_table_df[sample_name_columns]

    # Read in metadata table if given
    metadata_table_df = pd.read_csv(metadata_table_csv_filepath)
    metadata_table_df = metadata_table_df.set_index(
        metadata_id_column
    )
    metadata_table_df = metadata_table_df[field_columns]

    return AnalyteTable(
        data=analyte_table_df,
        metadata=metadata_table_df,
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