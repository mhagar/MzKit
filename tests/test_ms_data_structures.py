from core.data_structs import Injection, Fingerprint, ScanArray
from core.cli.mzml_import import mzml_to_injection

from pathlib import Path


def test_mzml_to_injection():
    injection: Injection = mzml_to_injection(
        input_filepath=Path("test_files/WATERS_DDA_STDMIX_R1.mzML")
    )

    assert injection


if __name__ == "__main__":
    test_mzml_to_injection()