from typing import TYPE_CHECKING
from pathlib import Path

from core.data_structs import DataRegistry
from core.data_structs.scan_array import ScanArrayParameters
from core.utils.persistence import load_project
from core.cli.generate_ensemble import (
    get_cofeature_ensemble
)

import pytest

if TYPE_CHECKING:
    from core.data_structs import (
        Sample, Injection, FeaturePointer, Ensemble
    )


# Path to TestSoln2 .mzk
TEST_SOLN_2_MZK_PATH = Path(
    '/home/mh/Dropbox/MzKit/tests/test_soln_2.mzk'
)

# Cycloheximide M+H
TARGET_MZ = 282.17024
TARGET_RT = 145


@pytest.fixture
def data_registry() -> DataRegistry:
    if not TEST_SOLN_2_MZK_PATH.exists():
        pytest.skip(
            f"test data not present: {TEST_SOLN_2_MZK_PATH} "
            "(MS data is gitignored; place it locally to run this test)"
        )
    samples, alignments = load_project(
        filepath=TEST_SOLN_2_MZK_PATH
    )

    registry = DataRegistry()
    registry.register_samples(samples)
    for alignment in alignments:
        registry.register_alignment(alignment)

    return registry


@pytest.fixture
def sample(data_registry) -> 'Sample':
    return data_registry.get_all_samples()[0]


@pytest.fixture
def injection(sample) -> 'Injection':
    return sample.injection


@pytest.fixture
def search_ftr_ptr(injection) -> 'FeaturePointer':
    return injection.scan_array_ms1.extract_feature_pointer(
        target_mz=TARGET_MZ,
        mz_window=0.01,
        target_rt=TARGET_RT,
        rt_window=10,
    )


@pytest.fixture
def ensemble(search_ftr_ptr, injection) -> 'Ensemble':
    return get_cofeature_ensemble(
        injection=injection,
        search_ftr_ptr=search_ftr_ptr,
        ms1_corr_threshold=0.9,
        ms2_corr_threshold=0.9,
        min_intsy=4000,
        use_rel_intsy=True,
    )

@pytest.fixture
def input_filepaths() -> list[str]:
    paths = [
        'tests/test_files/hifan_fractions/20250708_L97117_5_1.mzML',
        'tests/test_files/hifan_fractions/20250708_L97117_5_3.mzML',
        'tests/test_files/hifan_fractions/20250708_L97117_5_5.mzML',
        'tests/test_files/hifan_fractions/20250708_L97117_5_7.mzML',
    ]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        pytest.skip(
            f"test data not present (e.g. {missing[0]}); "
            "mzML fixtures are gitignored"
        )
    return paths


@pytest.fixture
def sample_names() -> list[str]:
    return  [
        'L97117_5_1',
        'L97117_5_3',
        'L97117_5_5',
        'L97117_5_7',
    ]


@pytest.fixture
def scan_array_params() -> tuple:
    return (
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
        )
    )




