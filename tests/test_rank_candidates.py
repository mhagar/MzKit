import numpy as np
import pytest

from core.cli.rank_candidates import (
    DbCandidate, RankParams, considered_peaks, explained_peaks, fetch_candidates, ion_name,
    neutral_mass, rank_candidates, rank_params_from_config,
)
from core.utils import glacier
from core.utils.array_types import to_spec_arr
from core.utils.config import load_default_config
from test_compound_db import db  # noqa: F401 (fixture)

needs_model = pytest.mark.skipif(not glacier.model_available(), reason="GLACIER model not in models/glacier/")


def spec(*peaks):
    mz, intsy = zip(*peaks)
    return to_spec_arr(np.array(mz, float), np.array(intsy, float))


def test_default_config_matches_defaults():
    assert rank_params_from_config(load_default_config()) == RankParams()


def test_considered_peaks_drops_precursor_and_small_and_caps():
    params = RankParams(top_n_peaks=2, min_exp_rel=0.05, mz_tolerance=0.01)
    ms2 = spec((100.0, 50), (150.0, 100), (200.0, 3), (250.0, 80), (300.005, 1000))
    # 300.005 is the precursor; 200 is under 5% of the base fragment (150)
    assert considered_peaks(ms2, 300.0, params).tolist() == [1, 3]


def test_explained_ignores_grass_and_tolerance():
    params = RankParams(min_pred_rel=0.01, mz_tolerance=0.01)
    ms2 = spec((100.0, 1), (150.0, 1), (200.0, 1))
    pred = spec((100.005, 0.02), (150.0, 0.005), (200.05, 1.0))
    # 150 is in the grass, 200.05 out of tolerance
    assert explained_peaks(ms2, np.array([0, 1, 2]), pred, params) == [0]


def test_fetch_candidates_dedupes_across_sources(db):  # noqa: F811
    cands, capped = fetch_candidates(['COCONUT', 'NPAtlas', 'AntiMarin'], formula='C2H6O', db_path=db)
    assert not capped
    # KEY1 in COCONUT and NPAtlas collapses; AntiMarin rows have no key, so stay
    assert [(c.source, c.db_id) for c in cands] == [
        ('COCONUT', 'CNP1'), ('COCONUT', 'CNP2'), ('AntiMarin', 'AM1'), ('AntiMarin', 'AM2'),
    ]
    by_mass, _ = fetch_candidates(['NPAtlas'], mass=46.0419, ppm=5, db_path=db)
    assert [c.db_id for c in by_mass] == ['NPA1']
    assert by_mass[0].organism == 'Fungi: Aspergillus'
    by_text, _ = fetch_candidates(['NPAtlas'], text_field='Organism', text='aspergil', db_path=db)
    assert [c.db_id for c in by_text] == ['NPA1']


def test_ion_name_from_find_mfs_adduct():
    assert ion_name('H', None) == ion_name('H', 1) == '[M+H]+'
    assert ion_name('-H', -1) == '[M-H]-'
    assert ion_name('Cl', -1) == '[M+Cl]-'
    assert ion_name('H2', 2) == '[M+H2]2+'   # unknown to GLACIER: falls back
    assert ion_name(None, 1) is None


def test_neutral_mass_falls_back_to_protonated():
    assert neutral_mass(101.0, '[M+Na]+') < neutral_mass(101.0, None) == neutral_mass(101.0, '[M+H]+')


@needs_model
def test_true_structure_ranks_first_among_isomers():
    """A molecule's own prediction, as the 'experimental' MS2, ranks it first."""
    isomers = {  # C9H11NO2
        'phenylalanine': 'N[C@@H](Cc1ccccc1)C(=O)O',
        'benzocaine': 'CCOC(=O)c1ccc(N)cc1',
        'methacetin': 'CC(=O)Nc1ccc(OC)cc1',
    }
    cands = [DbCandidate('test', k, k, s, 'C9H11NO2') for k, s in isomers.items()]
    params = RankParams()
    pred = glacier.load_model().predict([isomers['benzocaine']], np.array([params.collision_energy]),
                                        ['[M+H]+'], [params.instrument])[0]
    mz, intsy, _ = glacier.merge_peaks(pred['spec'])
    precursor = 165.078979 + 1.007276

    result = rank_candidates(to_spec_arr(mz, intsy), precursor, '[M+H]+', cands, params)

    best = result.scored[0]
    assert best.candidate.db_id == 'benzocaine'
    assert best.explained == result.n_considered > 0
    assert best.cosine == pytest.approx(1.0)
    assert best.frag_atoms.shape[0] == len(best.prediction) == len(best.ions) == len(best.h_shifts)
    assert all(ion.endswith('+') for ion in best.ions)
    assert not result.adduct_fallback and not result.skipped


@needs_model
def test_unpredictable_smiles_skipped_not_fatal():
    cands = [DbCandidate('t', 'ok', 'ok', 'CCO', 'C2H6O'), DbCandidate('t', 'bad', 'bad', 'C1CC', 'x')]
    result = rank_candidates(spec((20.0, 1.0)), 47.05, '[M+2H]2+', cands)
    assert [s.candidate.db_id for s in result.scored] == ['ok']
    assert [c.db_id for c in result.skipped] == ['bad']
    assert result.adduct == '[M+H]+' and result.adduct_fallback


def test_label_falls_back_to_id_for_long_names():
    assert DbCandidate('COCONUT', 'CNP1', 'erythromycin', 'C', 'C').label == 'erythromycin'
    assert DbCandidate('COCONUT', 'CNP1', '', 'C', 'C').label == 'CNP1'
    assert DbCandidate('COCONUT', 'CNP1', '17-hydroxy-11-[(2S,3S,4R,5R,6S)-3,4,5-trihydroxy-6-'
                       '(hydroxymethyl)oxan-2-yl]-3,5-dioxa', 'C', 'C').label == 'CNP1'
