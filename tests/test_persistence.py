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


def test_formula_assignment_roundtrip(tmp_path):
    """A FormulaAssignment must survive save/load with every score term intact
    (unlike the lossy FormulaCandidate annotation format)."""
    from core.data_structs.formula_assignment import (
        FormulaAssignment, AssignedCandidate,
    )

    reg = DataRegistry()
    assignment = FormulaAssignment(
        source_uuid=999,
        candidates=[
            AssignedCandidate(
                formula_str="C12H15Cl2NO5S", adduct="H",
                error_ppm=1.2, error_da=0.0004, rdbe=5.5,
                mass_loglik=-1.1, iso_loglik=-2.2, chem_logprior=-3.3,
                ms2_loglik=-0.5, log_posterior=-7.0,
            ),
        ],
        chosen_idx=0,
        precursor_mz=356.0126, charge=1, adducts=["H"], ms2_mode="tallest",
        elements="CHNOSClBr", max_counts="C*H*N*O*S2Cl4Br4", min_counts=None,
        halogen_detected=True,
        params={"max_counts": "C*H*N*O*P0S2", "halogen_cap": "Cl4Br4"},
    )
    reg.register_assignment(assignment)

    path = tmp_path / "roundtrip.mzk"
    persistence.save_project(filepath=path, data_registry=reg)
    _, _, assignments = persistence.load_project(filepath=path)

    assert len(assignments) == 1
    loaded = assignments[0]
    assert loaded.uuid == assignment.uuid
    assert loaded.source_uuid == 999
    assert loaded.chosen_idx == 0
    assert loaded.halogen_detected is True
    assert loaded.max_counts == "C*H*N*O*S2Cl4Br4"
    assert loaded.params == assignment.params
    assert loaded.precursor_mz == 356.0126
    assert loaded.adducts == ["H"]

    assert len(loaded.candidates) == 1
    c = loaded.candidates[0]
    assert c.formula_str == "C12H15Cl2NO5S"
    assert c.ms2_loglik == -0.5          # score terms round-trip exactly
    assert c.log_posterior == -7.0
    assert c.chem_logprior == -3.3


def test_pre_1_2_assignment_is_skipped_not_fatal(tmp_path, caplog):
    """Assignments saved before 1.2.0 (element-set provenance fields) aren't
    migrated; loading skips them with a warning and keeps the rest."""
    import json
    import zipfile

    path = tmp_path / "old.mzk"
    persistence.save_project(filepath=path, data_registry=DataRegistry())
    old = {
        "source_uuid": 1, "candidates": [], "uuid": 2, "chosen_idx": None,
        "elements": "CHNOPS", "autodetect_cl_br": True, "error_ppm": 5.0,
    }
    with zipfile.ZipFile(path, mode="a") as zf:
        zf.writestr("assignments/2.json", json.dumps(old))

    with caplog.at_level(logging.WARNING):
        _, _, assignments = persistence.load_project(filepath=path)

    assert assignments == []
    assert "older format version" in caplog.text


def test_alignment_analyte_uuid_roundtrip(tmp_path):
    """Analyte uuids must survive save/load (analyte metadata keys on them)."""
    from core.data_structs.alignment import EnsembleAlignment, AlignedAnalyte

    reg = DataRegistry()
    analyte = AlignedAnalyte(
        ensemble_map={1: 11, 2: 22}, consensus_rt=60.0, consensus_mz=300.1,
    )
    alignment = EnsembleAlignment(sample_uuids=(1, 2), analytes=[analyte])
    reg.register_alignment(alignment)

    path = tmp_path / "roundtrip.mzk"
    persistence.save_project(filepath=path, data_registry=reg)
    _, alignments, _ = persistence.load_project(filepath=path)

    assert len(alignments) == 1
    loaded = alignments[0].analytes[0]
    assert loaded.uuid == analyte.uuid
    assert alignments[0].get_analyte(analyte.uuid) is loaded
    assert loaded.ensemble_map == {1: 11, 2: 22}


def test_load_project(
    data_registry: DataRegistry,
):
    samples, alignments, _assignments = persistence.load_project(
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
    import pytest
    missing = [p for p in mzml_paths if not Path(p).exists()]
    if missing:
        pytest.skip(
            f"test data not present (e.g. {missing[0]}); mzML is gitignored"
        )

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





