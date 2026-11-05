from core.utils import persistence
from core.data_structs.ensemble import AdductNetwork

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


def _reloaded(data_registry, sample, ensemble, tmp_path):
    """`ensemble` as it comes back from a save/load of `data_registry`."""
    path = tmp_path / 'reloaded.mzk'
    persistence.save_project(filepath=path, data_registry=data_registry)
    return next(
        s for s in persistence.load_project(filepath=path).samples
        if s.uuid == sample.uuid
    ).injection.ensembles[ensemble.uuid]


def test_formula_assignment_roundtrip(data_registry, sample, ensemble, tmp_path):
    """An ensemble's FormulaAssignment must survive save/load with every score
    term intact."""
    from core.data_structs.formula_assignment import (
        FormulaAssignment, AssignedCandidate,
    )

    assignment = FormulaAssignment(
        source_uuid=ensemble.uuid,
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
        envelope=[0, 2], scan_num=7, M=355.0053,
    )
    network = AdductNetwork(
        M=355.0053, base_envelope=[0, 2], ions={0: '[M+H]+', 3: '[M+Na]+'},
        near_tie_adducts=[('H', 1), (None, 1)],
    )
    data_registry.update_ensemble(
        ensemble.uuid, formula_assignment=assignment, adduct_network=network,
    )

    loaded = _reloaded(data_registry, sample, ensemble, tmp_path).formula_assignment
    assert loaded == assignment
    assert loaded.source_uuid == ensemble.uuid
    assert loaded.chosen_idx == 0
    assert loaded.chosen_by == "user" and loaded.user_chosen
    assert loaded.halogen_detected is True
    assert loaded.max_counts == "C*H*N*O*S2Cl4Br4"
    assert loaded.params == assignment.params
    assert loaded.precursor_mz == 356.0126
    assert loaded.adducts == ["H"]

    assert len(loaded.candidates) == 2
    assert loaded.candidates[1].manual and loaded.candidates[1].error_ppm is None
    assert loaded.envelope == [0, 2] and loaded.scan_num == 7

    loaded_net = _reloaded(data_registry, sample, ensemble, tmp_path).adduct_network
    assert loaded_net == network and loaded_net.base_ion == '[M+H]+'
    assert not loaded.candidates[0].manual
    c = loaded.candidates[0]
    assert c.formula_str == "C12H15Cl2NO5S"
    assert c.ms2_loglik == -0.5          # score terms round-trip exactly
    assert c.log_posterior == -7.0
    assert c.chem_logprior == -3.3


def test_ensemble_group_roundtrip(tmp_path):
    """EnsembleGroups (incl. overlapping ones) survive save/load field by field."""
    from core.data_structs import EnsembleGroup

    reg = DataRegistry()
    big = EnsembleGroup(
        ensemble_uuids=[3, 2**120, 2], title="Psammaplins", notes="bromotyrosines",
        smiles="C1=CC=CC=C1", created_at=1760000000.0,
        provenance={"alignment_uuid": 42, "params": {"linkage_method": "average"}},
    )
    small = EnsembleGroup(ensemble_uuids=[2], notes="overlaps big")
    reg.register_group(big)
    reg.register_group(small)

    path = tmp_path / "groups.mzk"
    persistence.save_project(filepath=path, data_registry=reg)
    groups = persistence.load_project(filepath=path).groups

    assert sorted(groups, key=lambda g: g.created_at) == [big, small]


def test_ensemble_structure_annotation_roundtrip(data_registry, sample, ensemble, tmp_path):
    from core.data_structs.ensemble import EnsembleAnnotation, Hypothesis

    annotation = EnsembleAnnotation(
        name="Psammaplin A", notes="from the sponge",
        hypotheses=(
            Hypothesis("CCO", "ethanol", origin="NPAtlas", db_id="NPA1",
                       scores={"explained": "7/10", "cosine": 0.61}),
            Hypothesis("CCCO"),
        ),
        chosen_idx=1,
    )
    data_registry.update_ensemble(ensemble.uuid, annotation=annotation)

    assert _reloaded(data_registry, sample, ensemble, tmp_path).annotation == annotation

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


def test_ensemble_annotation_roundtrip(data_registry, sample, ensemble, tmp_path):
    from find_mfs import FormulaCandidate
    from molmass import Formula

    ens = ensemble
    a = ens.add_ion_annot([0], 1, FormulaCandidate(
        Formula('C15H23NO4'), 1.2, 0.0003, 5.0, adduct='Na'), label='')
    b = ens.add_ion_annot([0], 1, FormulaCandidate(
        Formula('C15H21NO3'), -0.4, 0.0001, 6.0), label='x', scan_num=3)
    ens.add_ion_pair_annot(a.uuid, b.uuid, 'neutral_loss')
    ens.add_mz_diff_annot(0, 0, 1, 18.0106, formula='H2O')
    ens.add_generic_annot([0], 1, 'hello')
    assert ens.ion_pair_annots[0].formula_diff == 'H2O'
    assert a.isotope_envelope.size  # adduct path still builds an ion formula

    loaded = _reloaded(data_registry, sample, ens, tmp_path)

    for field in ('mz_diffs', 'ion_annots', 'ion_pair_annots', 'generic_annots'):
        assert getattr(loaded, field) == getattr(ens, field)


def test_old_generic_annotation_loads_as_multi_peak():
    a = persistence._generic_annot(
        {'cofeature_idx': 3, 'ms_level': 1, 'text': 'x', 'uuid': 1, 'source': 'user'}
    )
    assert a.cofeature_idxs == [3] and a.text == 'x'
