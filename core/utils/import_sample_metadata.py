"""
Script for reading in a .csv file and writing into the 'metadata'
fields of Sample objects
"""
import pandas as pd

from pathlib import Path
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from core.data_structs import (
        Sample,
        SampleUUID,
    )


def read_metadata_csv(
    csv_filepath: Path,
    samplename_column: str,
    metadata_columns: Optional[list[str]],
    samples: list['Sample'],
    samples_in_rows: bool = True,
) -> dict['SampleUUID',
dict[str, any]
]:
    """
    Given a .csv file containing sample metadata, reads the contents
    and outputs this format:
    >>> {
    >>> 	19028124 : {  # SampleUUID
    >>> 		'organism': 'marinius spongus',
    >>> 		'fraction': '30% MeOH',
    >>> 	},
    >>> 	95843329 : {  # SampleUUID
    >>> 		'organism': 'marinius spongus',
    >>> 		'fraction': '90% MeOH',
    >>> 	},
    >>> }
    """
    df = pd.read_csv(
        csv_filepath
    )

    if not samples_in_rows:
        df = df.T

    df = sanity_check(df, metadata_columns, samplename_column)

    if metadata_columns:
        df = df[metadata_columns]

    results = {}
    for sample in samples:
        results[sample.uuid] = {}

        if sample.name in df.index:
            row_dict = df.loc[sample.name].to_dict()

            for key, value in row_dict.items():
                results[sample.uuid][key] = value

        else:
            # Fill the sample metadata field with empty values
            for key in df.columns:
                results[sample.uuid][key] = None

    return results


def sanity_check(df, metadata_columns, samplename_column):
    if samplename_column not in df.columns:
        raise ValueError(
            f"samplename_column: {samplename_column} not found in"
            f" table. Columns found: {df.columns}"
        )
    df = df.set_index(samplename_column)
    if len(df.index.unique()) != len(df.index):
        raise ValueError(
            "Table contains rows with duplicate sample names"
        )
    for metadata_field in metadata_columns:
        if metadata_field not in df.columns:
            raise ValueError(
                f"Table does not contain metadata field '{metadata_field}'"
            )
    return df


