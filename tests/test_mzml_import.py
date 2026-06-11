from typing import Optional, TYPE_CHECKING
import logging

from core.data_structs.scan_array import ScanArrayParameters
import core.cli.mzml_import as mzml_import

logging.basicConfig()
if TYPE_CHECKING:
    from core.data_structs import (
        Sample,
    )

def test_mzml_import(
    input_filepaths: list[str],
    sample_names: list[str],
    scan_array_params: tuple[
        'ScanArrayParameters', Optional['ScanArrayParameters']
    ],
) -> list['Sample']:
    samples = mzml_import.main(
        input_filepaths=input_filepaths,
        sample_names=sample_names,
        scan_array_params=scan_array_params,
    )

    assert samples
    assert len(samples) == len(sample_names)

FILEPATHS = [
    'tests/test_files/hifan_fractions/20250708_L97117_5_1.mzML',
    'tests/test_files/hifan_fractions/20250708_L97117_5_3.mzML',
    'tests/test_files/hifan_fractions/20250708_L97117_5_5.mzML',
    'tests/test_files/hifan_fractions/20250708_L97117_5_7.mzML',
]

SAMPLENAMES = [
    # 'Blank',
    # 'L7161_3_3',
    # 'L7161_3_6',
    # 'L7161_3_7',
    # 'L7161_3_12',
    # 'L7161_3_16',
    # 'L7161_3_17',
    'L97117_5_1',
    'L97117_5_3',
    'L97117_5_5',
    'L97117_5_7',
]

ms1_params = ScanArrayParameters(
    ms_level=1,
    mz_tolerance=0.05,
    scan_gap_tolerance=3,
    min_intsy=2000,
    scan_nums=None,
)

ms2_params = ScanArrayParameters(
    ms_level=2,
    mz_tolerance=0.05,
    scan_gap_tolerance=3,
    min_intsy=2000,
    scan_nums=None,
)

if __name__ == "__main__":
    test_mzml_import(
        input_filepaths=FILEPATHS,
        sample_names=SAMPLENAMES,
        scan_array_params=(ms1_params, ms2_params),
    )
