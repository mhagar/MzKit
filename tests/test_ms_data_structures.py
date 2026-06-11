from pathlib import Path

import pytest

from core.data_structs import Injection
from core.data_structs.scan_array import ScanArrayParameters
from core.cli.mzml_import import mzml_to_injection


MZML_PATH = Path("tests/WATERS_DDA_STDMIX_R1.mzML")


def test_mzml_to_injection():
    if not MZML_PATH.exists():
        pytest.skip(f"test data not present: {MZML_PATH} (mzML is gitignored)")

    scan_array_params = (
        ScanArrayParameters(
            ms_level=1,
            mz_tolerance=0.01,
            scan_gap_tolerance=3,
            min_intsy=2000,
            scan_nums=None,
        ),
        ScanArrayParameters(
            ms_level=2,
            mz_tolerance=0.01,
            scan_gap_tolerance=3,
            min_intsy=2000,
            scan_nums=None,
        ),
    )

    injection: Injection = mzml_to_injection(
        input_filepath=MZML_PATH,
        scan_array_params=scan_array_params,
        acquisition_mode="dda",
    )

    assert injection
    assert injection.scan_array_ms1 is not None


if __name__ == "__main__":
    test_mzml_to_injection()
