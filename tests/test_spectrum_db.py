import numpy as np
import pytest

from core.cli.search_spectra import IDENTITY, SpectralSearchParams, library_hit, search_spectra
from core.utils import ms2deepscore, spectrum_db
from core.utils.array_types import to_spec_arr

PEAKS_A = [(91.054, 30.0), (119.049, 100.0), (147.044, 55.0), (165.055, 20.0)]
PEAKS_B = [(70.065, 100.0), (98.060, 40.0), (126.055, 60.0), (154.050, 10.0)]

# GNPS-cleaned dialect, plus one negative-mode entry (skipped)
GNPS = f"""BEGIN IONS
SPECTRUM_ID=CCMSLIB1
COMPOUND_NAME=alpha
ADDUCT=[M+H]1+
SMILES=CCO
INCHIKEY=LFQSCWFLJHTTHZ-UHFFFAOYSA-N
IONMODE=positive
PRECURSOR_MZ=183.07
{chr(10).join(f'{m} {i} ' for m, i in PEAKS_A)}
END IONS

BEGIN IONS
SPECTRUM_ID=CCMSLIB2
COMPOUND_NAME=negative
IONMODE=negative
PRECURSOR_MZ=181.05
{chr(10).join(f'{m} {i}' for m, i in PEAKS_A)}
END IONS
"""

# MZmine dialect: PEPMASS, NAME, SPECTRUMID, INChIKey in INCHIAUX
MZMINE = f"""BEGIN IONS
SPECTRUMID=7
NAME=beta
INCHIAUX=VURLRACCOCGFDB-UHFFFAOYSA-N
SMILES=CCN
ADDUCT=[M+Na]+
PEPMASS=172.04
CHARGE=1
IONMODE=POSITIVE
{chr(10).join(f'{m} {i}' for m, i in PEAKS_B)}
END IONS
"""


@pytest.fixture
def db(tmp_path):
    (tmp_path / 'GNPS.mgf').write_text(GNPS)
    (tmp_path / 'inhouse.mgf').write_text(MZMINE)
    # An external embedder covering only GNPS
    np.savez(tmp_path / 'GNPS.toy.npz', ids=np.array(['CCMSLIB1']), embeddings=np.array([[1.0, 0.0, 0.0]]))
    path = tmp_path / 'spectra.sqlite'
    assert spectrum_db.build_db(tmp_path, path) == str(path)
    return path


def _spec(peaks):
    mz, intsy = np.array(peaks).T
    return to_spec_arr(mz, intsy)


def test_both_dialects_harmonized(db):
    assert spectrum_db.cache_ok(db)
    assert sorted(spectrum_db.available_libraries(db)) == ['GNPS', 'inhouse']
    rows = spectrum_db.rows([1, 2, 3], db)
    assert len(rows) == 2   # the negative-mode entry was skipped
    by_id = {r[1]: r for r in rows.values()}
    assert by_id['CCMSLIB1'][:8] == (
        'GNPS', 'CCMSLIB1', 'alpha', 'CCO', 'LFQSCWFLJHTTHZ-UHFFFAOYSA-N', '', '[M+H]1+', 183.07,
    )
    beta = by_id['7']
    assert (beta[0], beta[2], beta[4], beta[7]) == ('inhouse', 'beta', 'VURLRACCOCGFDB-UHFFFAOYSA-N', 172.04)
    np.testing.assert_allclose(beta[-1]['mz'], [m for m, _ in PEAKS_B], rtol=1e-6)


def test_identity_search_finds_itself(db):
    result = search_spectra(_spec(PEAKS_A), 183.075, ['GNPS', 'inhouse'], IDENTITY, db_path=db)
    [hit] = result.hits
    assert (hit.spectrum_id, hit.name) == ('CCMSLIB1', 'alpha')
    assert hit.score == pytest.approx(1.0, abs=1e-6)
    assert len(hit.matches) == len(PEAKS_A) - 1   # 165.055 (-H2O) is in the precursor region
    assert hit.delta_mz == pytest.approx(-0.005)
    # Outside the precursor window / other library: nothing
    assert not search_spectra(_spec(PEAKS_A), 190.0, ['GNPS'], IDENTITY, db_path=db).hits
    assert not search_spectra(_spec(PEAKS_A), 183.07, ['inhouse'], IDENTITY, db_path=db).hits


def test_library_hit_rebuilds_an_accepted_hit(db):
    [searched] = search_spectra(_spec(PEAKS_A), 183.075, ['GNPS'], IDENTITY, db_path=db).hits
    hit = library_hit(_spec(PEAKS_A), 183.075, 'GNPS', 'CCMSLIB1', db_path=db)
    assert (hit.spectrum_id, hit.score, hit.matches) == (searched.spectrum_id, searched.score, searched.matches)
    assert library_hit(_spec(PEAKS_A), 183.075, 'GNPS', 'nope', db_path=db) is None


def test_external_embeddings_map_by_id(db):
    assert 'toy' in spectrum_db.embedders(db)
    emb = spectrum_db.load_embeddings('toy', db)
    gnps_row = next(i for i, r in spectrum_db.rows([1, 2], db).items() if r[1] == 'CCMSLIB1')
    np.testing.assert_array_equal(emb[gnps_row - 1], [1, 0, 0])
    assert np.isnan(emb[2 - gnps_row]).all()   # the in-house row isn't covered
    result = search_spectra(
        _spec(PEAKS_B), 172.04, ['GNPS', 'inhouse'], 'toy',
        SpectralSearchParams(top_k=5), query_embedding=np.array([0.5, 0.5, 0.0]), db_path=db,
    )
    assert [h.spectrum_id for h in result.hits] == ['CCMSLIB1']   # uncovered rows aren't hits
    assert result.hits[0].score == pytest.approx(np.sqrt(0.5), abs=1e-3)


@pytest.mark.skipif(not ms2deepscore.model_available(), reason="MS2DeepScore weights not present")
def test_ms2deepscore_analogue_search_finds_itself(db):
    assert spectrum_db.embedders(db)[0] == spectrum_db.MS2DEEPSCORE
    result = search_spectra(
        _spec(PEAKS_B), 172.04, ['GNPS', 'inhouse'], spectrum_db.MS2DEEPSCORE, db_path=db,
    )
    assert result.n_searched == 2
    assert result.hits[0].spectrum_id == '7'
    assert result.hits[0].score == pytest.approx(1.0, abs=1e-2)
