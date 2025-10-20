from typing import TYPE_CHECKING
from pathlib import Path

from core.data_structs import DataRegistry
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
    samples: list['Sample'] = load_project(
        filepath=TEST_SOLN_2_MZK_PATH
    )

    registry = DataRegistry()
    registry.register_samples(samples)

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




