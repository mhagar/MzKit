"""
Structure databases (COCONUT, NPAtlas, AntiMarin, ...) cached into one indexed SQLite file,
so formula / mass lookups read only matching rows instead of the whole dump.
"""
import logging
import os
import sqlite3
from pathlib import Path

from core.formula.manual import canonical_formula
from core.utils.config import get_project_root

DB_DIR = Path(get_project_root()) / 'coconut'
DEFAULT_DB = DB_DIR / 'compounds.sqlite'

# Bump when the table layout changes; older caches must be rebuilt
SCHEMA_VERSION = 3

# Unified columns, in table order. `organisms` is pipe-joined full names
# ("Genus species|..."); `genus` is derived from it.
COLUMNS = ('db_id', 'name', 'smiles', 'formula', 'exact_mass', 'inchikey',
           'organism_type', 'genus', 'organisms', 'np_class')


def _coconut(df):
    df['name'] = df['name'].where(df['name'] != '', df['iupac_name'])
    return df


def _npatlas(df):
    df['organisms'] = (df['npa_genus'] + ' ' + df['npa_species']).str.strip()
    return df


def _antimarin(df):
    df['db_id'] = 'AM' + df['db_id']  # its ids are a bare serial number
    return df


# source -> (dump path, {dump column: unified column}, prep(df) -> df).
# .tsv / .json / anything else (CSV) picks the reader. prep runs after the
# rename; unified columns still missing afterwards are left blank.
# Adding a database = adding an entry here.
SOURCES = {
    'COCONUT': (DB_DIR / 'coconut_csv-10-2026.csv', {
        'identifier': 'db_id',
        'name': 'name',
        'iupac_name': 'iupac_name',
        'canonical_smiles': 'smiles',
        'molecular_formula': 'formula',
        'exact_molecular_weight': 'exact_mass',
        'standard_inchi_key': 'inchikey',
        'organisms': 'organisms',
        'np_classifier_class': 'np_class',
    }, _coconut),
    'NPAtlas': (DB_DIR / 'NPAtlas_download_2024_09.tsv', {
        'npaid': 'db_id',
        'compound_name': 'name',
        'compound_smiles': 'smiles',
        'compound_molecular_formula': 'formula',
        'compound_accurate_mass': 'exact_mass',
        'compound_inchikey': 'inchikey',
        'origin_type': 'organism_type',
        'genus': 'npa_genus',
        'origin_species': 'npa_species',
    }, _npatlas),
    'AntiMarin': (DB_DIR / 'antimarin_porifera_download.json', {
        'coconut_id': 'db_id',
        'name': 'name',
        'smiles': 'smiles',
        'molecular_formula': 'formula',
        'exact_mass': 'exact_mass',
        'origin_organism_type': 'organism_type',
    }, _antimarin),
}

# Dump spellings -> one vocabulary (Plant / Marine / Bacteria / Fungi / Other)
ORGANISM_TYPES = {'Fungus': 'Fungi', 'Bacterium': 'Bacteria', 'porifera': 'Marine'}

MAX_RESULTS = 5000  # shortcut: fixed cap, add paging if it bites


def _read_chunks(path: Path, colmap: dict[str, str]):
    """The dump's mapped columns as all-str DataFrames ('' for missing)."""
    import pandas as pd
    if path.suffix.lower() == '.json':
        yield pd.read_json(path, dtype=False)[list(colmap)].astype(str)
        return
    yield from pd.read_csv(
        path, sep='\t' if path.suffix.lower() == '.tsv' else ',',
        usecols=list(colmap), dtype=str, keep_default_na=False, chunksize=50_000,
    )


def _genera(organisms: str) -> str:
    """'Morus alba|Morus indica|Pinus sp' -> 'Morus|Pinus'"""
    return '|'.join(dict.fromkeys(o.split()[0] for o in organisms.split('|') if o.strip()))


def build_db(
        db_path: Path = DEFAULT_DB,
        progress_callback=None,  # injected by ProcessRunner; unused here
        cancel_event=None,       # injected by ProcessRunner; unused here
) -> str:
    """
    (Re)build the cache from every dump in SOURCES that exists. Written to a
    temp file and renamed at the end, so a half-built cache is never used.
    Returns the path to the built database.
    """
    import pandas as pd
    db_path = Path(db_path)
    tmp = db_path.with_suffix('.tmp')
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.execute(
            "CREATE TABLE compounds (source TEXT, db_id TEXT, name TEXT, "
            "smiles TEXT, formula TEXT, exact_mass REAL, inchikey TEXT, "
            "organism_type TEXT, genus TEXT, organisms TEXT, np_class TEXT)"
        )
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        canonical: dict[str, str] = {}  # dump formula -> Hill form, shared across chunks
        for source, (path, colmap, prep) in SOURCES.items():
            path = Path(path)
            if not path.exists():
                logging.warning(f"{source}: {path} not found, skipped")
                continue
            n = 0
            for chunk in _read_chunks(path, colmap):
                chunk = prep(chunk.rename(columns=colmap)).reindex(columns=COLUMNS, fill_value='')
                for f in set(chunk['formula']) - canonical.keys():
                    try:
                        canonical[f] = canonical_formula(f)
                    except ValueError:
                        canonical[f] = f  # kept as-is; just won't match a formula query
                chunk['formula'] = chunk['formula'].map(canonical)
                chunk['genus'] = chunk['organisms'].map(_genera)
                chunk['organism_type'] = chunk['organism_type'].replace(ORGANISM_TYPES)
                chunk['exact_mass'] = pd.to_numeric(chunk['exact_mass'], errors='coerce')
                chunk.insert(0, 'source', source)
                con.executemany(
                    f"INSERT INTO compounds VALUES ({','.join('?' * (len(COLUMNS) + 1))})",
                    chunk.itertuples(index=False, name=None),
                )
                n += len(chunk)
            logging.info(f"{source}: {n} compounds")
        con.execute("CREATE INDEX ix_formula ON compounds(source, formula)")
        con.execute("CREATE INDEX ix_mass ON compounds(source, exact_mass)")
        con.commit()
    finally:
        con.close()
    os.replace(tmp, db_path)
    return str(db_path)


def cache_ok(db_path: Path = DEFAULT_DB) -> bool:
    """The cache exists and was built with the current schema."""
    if not Path(db_path).exists():
        return False
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        return con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def available_sources(db_path: Path = DEFAULT_DB) -> list[str]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        return [r[0] for r in con.execute("SELECT DISTINCT source FROM compounds")]


def search_formula(source: str, formula: str, db_path: Path = DEFAULT_DB) -> list[tuple]:
    """Rows of COLUMNS. :raises ValueError: if `formula` doesn't parse."""
    formula = canonical_formula(formula)
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        return con.execute(
            f"SELECT {','.join(COLUMNS)} FROM compounds "
            "WHERE source=? AND formula=? LIMIT ?",
            (source, formula, MAX_RESULTS),
        ).fetchall()


def search_mass(
        source: str, mass: float, ppm: float, db_path: Path = DEFAULT_DB,
) -> list[tuple]:
    """Rows of COLUMNS + (delta ppm,) for neutral `mass` +/- `ppm`, closest first."""
    tol = mass * ppm * 1e-6
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        return con.execute(
            f"SELECT {','.join(COLUMNS)}, (exact_mass - ?1) / ?1 * 1e6 AS dppm "
            "FROM compounds WHERE source=?2 AND exact_mass BETWEEN ?3 AND ?4 "
            "ORDER BY abs(dppm) LIMIT ?5",
            (mass, source, mass - tol, mass + tol, MAX_RESULTS),
        ).fetchall()


# Text search fields: GUI label -> SQL expression searched
TEXT_FIELDS = {
    'Name': "name",
    'Organism': "organisms || '|' || organism_type",
    'ID': "db_id",
    'NP class': "np_class",
}


def search_text(
        source: str, field: str, text: str, db_path: Path = DEFAULT_DB,
) -> list[tuple]:
    """Rows of COLUMNS whose TEXT_FIELDS[field] contains `text` (case-insensitive)."""
    pattern = '%' + text.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_') + '%'
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        return con.execute(
            f"SELECT {','.join(COLUMNS)} FROM compounds "
            f"WHERE source=? AND {TEXT_FIELDS[field]} LIKE ? ESCAPE '\\' LIMIT ?",
            (source, pattern, MAX_RESULTS),
        ).fetchall()
