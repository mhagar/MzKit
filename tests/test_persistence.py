from core.utils import persistence

from core.data_structs import DataRegistry

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

def test_save_load_project(
    data_registry: DataRegistry,
    tmp_path,
):
    filepath = tmp_path / 'test_project.mzk'
    persistence.save_project(
        filepath=filepath,
        data_registry=data_registry
    )
    samples = persistence.load_project(filepath=filepath).samples

    # Check if loaded and saved samples are indeed the same
    sample_uuids_original = data_registry.get_all_sample_uuids()
    sample_uuids_loaded = [ x.uuid for x in samples ]

    assert sample_uuids_loaded == sample_uuids_original


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
            AssignedCandidate(
                formula_str="C6H12O6", adduct=None, error_ppm=None,
                error_da=None, rdbe=1.0, mass_loglik=None, iso_loglik=None,
                chem_logprior=None, ms2_loglik=None, log_posterior=None,
                manual=True,
            ),
        ],
        chosen_idx=0,
        chosen_by="user",
        precursor_mz=356.0126, charge=1, adducts=["H"], ms2_mode="tallest",
        elements="CHNOSClBr", max_counts="C*H*N*O*S2Cl4Br4", min_counts=None,
        halogen_detected=True,
        params={"max_counts": "C*H*N*O*P0S2", "halogen_cap": "Cl4Br4"},
    )
    reg.register_assignment(assignment)

    path = tmp_path / "roundtrip.mzk"
    persistence.save_project(filepath=path, data_registry=reg)
    assignments = persistence.load_project(filepath=path).assignments

    assert len(assignments) == 1
    loaded = assignments[0]
    assert loaded.uuid == assignment.uuid
    assert loaded.source_uuid == 999
    assert loaded.chosen_idx == 0
    assert loaded.chosen_by == "user" and loaded.user_chosen
    assert loaded.halogen_detected is True
    assert loaded.max_counts == "C*H*N*O*S2Cl4Br4"
    assert loaded.params == assignment.params
    assert loaded.precursor_mz == 356.0126
    assert loaded.adducts == ["H"]

    assert len(loaded.candidates) == 2
    assert loaded.candidates[1].manual and loaded.candidates[1].error_ppm is None
    assert not loaded.candidates[0].manual
    c = loaded.candidates[0]
    assert c.formula_str == "C12H15Cl2NO5S"
    assert c.ms2_loglik == -0.5          # score terms round-trip exactly
    assert c.log_posterior == -7.0
    assert c.chem_logprior == -3.3


def test_analyte_group_roundtrip(tmp_path):
    """AnalyteGroups (incl. overlapping ones) survive save/load field by field."""
    from core.data_structs import AnalyteGroup

    reg = DataRegistry()
    big = AnalyteGroup(
        analyte_uuids=[3, 1, 2], title="Psammaplins", notes="bromotyrosines",
        smiles="C1=CC=CC=C1", created_at=1760000000.0,
        provenance={"alignment_uuid": 42, "params": {"linkage_method": "average"}},
    )
    small = AnalyteGroup(analyte_uuids=[2], notes="overlaps big")
    reg.register_group(big)
    reg.register_group(small)

    path = tmp_path / "groups.mzk"
    persistence.save_project(filepath=path, data_registry=reg)
    groups = persistence.load_project(filepath=path).groups

    assert sorted(groups, key=lambda g: g.created_at) == [big, small]


def test_analyte_annotation_roundtrip(tmp_path):
    from core.data_structs import AnalyteAnnotation

    reg = DataRegistry()
    named = AnalyteAnnotation(
        analyte_uuid=7, name="Psammaplin A", smiles="CCO",
        source="NPAtlas", db_id="NPA1", scores={"explained": "7/10", "cosine": 0.61},
    )
    reg.register_annotation(named)
    reg.register_annotation(AnalyteAnnotation(analyte_uuid=8, name="unknown 1"))

    path = tmp_path / "annotations.mzk"
    persistence.save_project(filepath=path, data_registry=reg)
    loaded = persistence.load_project(filepath=path).annotations

    assert sorted(loaded, key=lambda x: x.analyte_uuid) == [
        named, AnalyteAnnotation(analyte_uuid=8, name="unknown 1"),
    ]


def test_pre_1_4_file_has_no_groups(tmp_path):
    """Files saved before 1.4.0 have no groups/ entries; they load as []."""
    import json
    import zipfile

    path = tmp_path / "old.mzk"
    with zipfile.ZipFile(path, mode="w") as zf:
        zf.writestr(
            "project_metadata.json", json.dumps({"format_version": "1.3.0"}),
        )

    data = persistence.load_project(filepath=path)
    assert data.groups == []
    assert data.matrices == []
    assert data.annotations == []


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
        assignments = persistence.load_project(filepath=path).assignments

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
    alignments = persistence.load_project(filepath=path).alignments

    assert len(alignments) == 1
    loaded = alignments[0].analytes[0]
    assert loaded.uuid == analyte.uuid
    assert alignments[0].get_analyte(analyte.uuid) is loaded
    assert loaded.ensemble_map == {1: 11, 2: 22}
