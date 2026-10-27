import json

import pytest

from core.utils import compound_db


@pytest.fixture
def db(tmp_path, monkeypatch):
    coconut = tmp_path / 'coconut.csv'
    coconut.write_text(
        "identifier,name,iupac_name,canonical_smiles,molecular_formula,exact_molecular_weight,"
        "standard_inchi_key,organisms,np_classifier_class,extra\n"
        "CNP1,a,,CCO,C2H6O,46.04186,KEY1,Celosia cristata|Pinus koraiensis|Celosia sp,Sesquiterpenoids,x\n"
        "CNP2,,methoxymethane,COC,C2H6O,46.04186,KEY2,,,x\n"
        "CNP3,c,,C=O,CH2O,30.01056,KEY3,,,x\n"
        "CNP4,d,,C[N+](C)(C)C,C4H12N+,74.09643,KEY4,,,x\n"
    )
    npatlas = tmp_path / 'npatlas.tsv'
    npatlas.write_text(
        "npaid\tcompound_name\tcompound_smiles\tcompound_molecular_formula\tcompound_accurate_mass\tcompound_inchikey"
        "\torigin_type\tgenus\torigin_species\n"
        "NPA1\td\tCCO\tC2H6O\t46.0419\tKEY1\tFungus\tAspergillus\tterreus\n"
    )
    antimarin = tmp_path / 'antimarin.json'
    antimarin.write_text(json.dumps([
        {'coconut_id': str(i), 'exact_mass': 46.0419, 'smiles': 'CCO', 'name': f'am{i}',
         'molecular_formula': 'C2H6O', 'origin_organism_type': 'porifera'}
        for i in (1, 2)
    ]))
    paths = {'COCONUT': coconut, 'NPAtlas': npatlas, 'AntiMarin': antimarin}
    monkeypatch.setattr(compound_db, 'SOURCES', {
        name: (paths[name], *entry[1:]) for name, entry in compound_db.SOURCES.items()
    })
    path = tmp_path / 'compounds.sqlite'
    compound_db.build_db(path)
    return path


def test_formula_search_canonicalises_and_separates_sources(db):
    rows = {r[0]: r for r in compound_db.search_formula('COCONUT', 'OC2H6', db)}
    assert rows.keys() == {'CNP1', 'CNP2'}
    assert rows['CNP2'][1] == 'methoxymethane'  # blank name -> IUPAC name
    assert rows['CNP1'][-4:-1] == ('', 'Celosia|Pinus', 'Celosia cristata|Pinus koraiensis|Celosia sp')
    [npa] = compound_db.search_formula('NPAtlas', 'C2H6O', db)
    assert npa[0] == 'NPA1' and npa[-4:-1] == ('Fungi', 'Aspergillus', 'Aspergillus terreus')
    am = compound_db.search_formula('AntiMarin', 'C2H6O', db)
    assert sorted(r[0] for r in am) == ['AM1', 'AM2'] and {r[-4] for r in am} == {'Marine'}
    assert sorted(compound_db.available_sources(db)) == ['AntiMarin', 'COCONUT', 'NPAtlas']
    assert compound_db.cache_ok(db)


def test_charged_formula_matches(db):
    assert [r[0] for r in compound_db.search_formula('COCONUT', 'C4H12N+', db)] == ['CNP4']


def test_mass_search_window(db):
    hits = compound_db.search_mass('COCONUT', 30.0106, 5, db)
    assert [r[0] for r in hits] == ['CNP3']
    assert abs(hits[0][-1]) < 5
    assert compound_db.search_mass('COCONUT', 30.0110, 5, db) == []  # ~13 ppm off


def test_text_search(db):
    assert [r[0] for r in compound_db.search_text('COCONUT', 'Name', 'METHOXY', db)] == ['CNP2']
    assert [r[0] for r in compound_db.search_text('COCONUT', 'Organism', 'pinus', db)] == ['CNP1']
    assert len(compound_db.search_text('AntiMarin', 'Organism', 'marine', db)) == 2
    assert compound_db.search_text('COCONUT', 'Name', '%', db) == []  # wildcards are literal
