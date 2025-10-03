from core.utils import persistence

from core.cli.mzml_import import mzml_to_injection
from core.data_structs import DataRegistry, Sample
from core.data_structs.scan_array import ScanArrayParameters

import logging
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.data_structs import (
        Injection, Fingerprint,
    )

logging.basicConfig(
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

mzml_paths = [
    'test_files/hifan_fractions/20250708_L97117_5_6.mzML',
    # 'test_files/hifan_fractions/20250708_L97117_5_7.mzML',
    # 'test_files/hifan_fractions/20250708_L97117_5_8.mzML',
]

fingerprint_csv_path = 'test_files/L97117_5_1_to_19_biological_response.csv'


def test_save_project(
    data_registry: DataRegistry,
):
    persistence.save_project(
        filepath=Path('test_project.mzk'),
        data_registry=data_registry
    )


def test_load_project(
    data_registry: DataRegistry,
):
    samples: list['Sample'] = persistence.load_project(
        filepath=Path('test_project.mzk')
    )

    # Check if loaded and saved samples are indeed the same
    sample_uuids_original = data_registry.get_all_sample_uuids()
    sample_uuids_loaded = [ x.uuid for x in samples ]

    assert sample_uuids_loaded == sample_uuids_original


def test_populate_data_registry(
) -> DataRegistry:
    """
    Populates a DataRegistry using test files
    :return:
    """
    data_registry = DataRegistry()

    scan_array_params = ScanArrayParameters(
        ms_level=1,
        mz_tolerance=0.03,
        scan_gap_tolerance=3,
        min_intsy=3000,
        scan_nums=None,
    )

    for mzml_path in mzml_paths:
        injection: 'Injection' = mzml_to_injection(
            input_filepath=Path(mzml_path),
            scan_array_params=( scan_array_params, scan_array_params),
            verbose=True,
        )

        sample = Sample(
            name="Test Sample",
            injection=injection,
        )

        data_registry.register_sample(
            sample
        )

    return data_registry


if __name__ == "__main__":
    data_registry = test_populate_data_registry()

    test_save_project(data_registry)
    test_load_project(data_registry)





