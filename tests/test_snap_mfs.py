"""Tests for core/cli/snap_mfs.py (SNAP-MFs family ranking)."""
import csv

import pytest

from core.cli.snap_mfs import MCES, TANIMOTO, format_families, snap_mfs

_HEADER = ['id', 'identifier', 'canonical_smiles', 'name',
           'molecular_formula', 'np_classifier_class', 'standard_inchi_key']
_ROWS = [
    # A homologous fatty-acid family: one compound per query formula
    ('CNP1', 'CCCCCCCCCCCCCCCC(=O)O', 'palmitic acid', 'C16H32O2', 'Fatty acids'),
    ('CNP2', 'CCCCCCCCCCCCCCCCC(=O)O', 'margaric acid', 'C17H34O2', 'Fatty acids'),
    ('CNP3', 'CCCCCCCCCCCCCCCCCC(=O)O', 'stearic acid', 'C18H36O2', 'Fatty acids'),
    # Same formula as a query, unrelated structure: must not join the family
    ('CNP4', 'CC1(C)CCCC2(C)C1CCC(C)(O)C2CCCC(=O)O', 'decoy', 'C18H36O2', ''),
    # Related structure but formula not queried: never loaded
    ('CNP5', 'CCCCCCCCCCCCCCCCCCC(=O)O', 'nonadecanoic acid', 'C19H38O2', ''),
    ('CNP6', 'not a smiles', 'broken', 'C16H32O2', ''),
]


# MCES rates the decoy J ~0.68 to the acids (the shared chain), so it needs a higher cutoff
@pytest.mark.parametrize('metric, cutoff', [(TANIMOTO, 0.5), (MCES, 0.8)])
def test_homologues_form_top_family(tmp_path, metric, cutoff):
    db = tmp_path / 'coconut.csv'
    with db.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(_HEADER)
        for i, row in enumerate(_ROWS):
            writer.writerow([i, *row, ''])

    formulas = ['C16H32O2', 'C17H34O2', 'O2C18H36', 'C16H32O2']  # dup, non-Hill
    families = snap_mfs(formulas, db, cutoff, metric=metric)
    assert len(families) == 1
    top = families[0]
    assert top.score == 3
    assert {c.identifier for c in top.compounds} == {'CNP1', 'CNP2', 'CNP3'}
    assert top.top_class() == 'Fatty acids'
    assert '*#1  explains 3/3 formulae' in format_families(families, formulas)


def test_npatlas_tsv(tmp_path):
    db = tmp_path / 'npatlas.tsv'
    with db.open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['npaid', 'compound_name', 'compound_molecular_formula',
                         'compound_inchikey', 'compound_smiles', 'origin_type', 'genus'])
        for identifier, smiles, name, formula, _ in _ROWS:
            writer.writerow([identifier, name, formula, '', smiles,
                             'Bacterium', 'Streptomyces'])

    families = snap_mfs(['C16H32O2', 'C17H34O2', 'C18H36O2'], db)
    assert {c.identifier for c in families[0].compounds} == {'CNP1', 'CNP2', 'CNP3'}
    assert families[0].top_class() == 'Bacterium Streptomyces'
