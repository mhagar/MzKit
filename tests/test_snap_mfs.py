"""Tests for core/cli/snap_mfs.py (SNAP-MFs family ranking)."""
import csv
from pathlib import Path

import pytest

from core.cli.snap_mfs import MCES, TANIMOTO, format_families, snap_mfs
from core.utils import compound_db

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




def _build_cache(tmp_path, monkeypatch, source: str, values) -> 'Path':
    """A cache holding only `source`, from rows of {unified column: value}."""
    path, colmap, prep = compound_db.SOURCES[source]
    dump = tmp_path / f'dump{path.suffix}'
    with dump.open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t' if path.suffix == '.tsv' else ',')
        writer.writerow(colmap)
        for row in values:
            writer.writerow([row.get(unified, '') for unified in colmap.values()])
    monkeypatch.setattr(compound_db, 'SOURCES', {source: (dump, colmap, prep)})
    return Path(compound_db.build_db(tmp_path / 'compounds.sqlite'))


# MCES rates the decoy J ~0.68 to the acids (the shared chain), so it needs a higher cutoff
@pytest.mark.parametrize('metric, cutoff', [(TANIMOTO, 0.5), (MCES, 0.8)])
def test_homologues_form_top_family(tmp_path, monkeypatch, metric, cutoff):
    db = _build_cache(tmp_path, monkeypatch, 'COCONUT', [
        dict(db_id=i, smiles=smiles, name=name, formula=formula, np_class=np_class)
        for i, smiles, name, formula, np_class in _ROWS
    ])

    formulas = ['C16H32O2', 'C17H34O2', 'O2C18H36', 'C16H32O2']  # dup, non-Hill
    families = snap_mfs(formulas, 'COCONUT', cutoff, metric=metric, db_path=db)
    assert len(families) == 1
    top = families[0]
    assert top.score == 3
    assert {c.identifier for c in top.compounds} == {'CNP1', 'CNP2', 'CNP3'}
    assert top.top_class() == 'Fatty acids'
    assert '*#1  explains 3/3 formulae' in format_families(families, formulas)


def test_npatlas_organism_class(tmp_path, monkeypatch):
    db = _build_cache(tmp_path, monkeypatch, 'NPAtlas', [
        dict(db_id=i, smiles=smiles, name=name, formula=formula,
             organism_type='Bacterium', npa_genus='Streptomyces', npa_species='sp.')
        for i, smiles, name, formula, _ in _ROWS
    ])

    families = snap_mfs(['C16H32O2', 'C17H34O2', 'C18H36O2'], 'NPAtlas', db_path=db)
    assert {c.identifier for c in families[0].compounds} == {'CNP1', 'CNP2', 'CNP3'}
    assert families[0].top_class() == 'Bacteria Streptomyces'
