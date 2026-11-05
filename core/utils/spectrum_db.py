"""
Spectral libraries (GNPS, in-house, ...) cached into one indexed SQLite file,
plus a row-aligned embedding matrix per embedder for analogue search.

Every `<library>.mgf` in DB_DIR is one library (named by its stem), read by
matchms with metadata harmonization, which maps the MGF dialects onto one set
of keys (PEPMASS -> precursor_mz, NAME -> compound_name, ...).

Embeddings sit next to the cache as `<db stem>.<embedder>.npy` (float16, row
i = rowid i + 1):
- 'ms2deepscore': computed at build time if the model is available.
- External embedders: `<library>.<embedder>.npz` in DB_DIR, with `ids` (the
  library's spectrum ids, see `_spectrum_id`) and `embeddings` (n x d).
Rows are unit vectors (so a dot product is the cosine); rows an embedder
doesn't cover, or couldn't embed, are NaN and never score.
"""
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from core.utils import ms2deepscore
from core.utils.array_types import to_spec_arr
from core.utils.config import get_project_root

DB_DIR = Path(get_project_root()) / 'spectra'
DEFAULT_DB = DB_DIR / 'spectra.sqlite'

# Bump when the table layout changes; older caches must be rebuilt
SCHEMA_VERSION = 1

MS2DEEPSCORE = 'ms2deepscore'

# Table columns after rowid, in order
COLUMNS = ('library', 'spectrum_id', 'name', 'smiles', 'inchikey', 'formula',
           'adduct', 'precursor_mz', 'peaks')

_INSERT_BATCH = 5000
_EMBED_BATCH = 4096


def _spectrum_id(spectrum, library: str, ordinal: int) -> str:
    """The id embedding files key on: SPECTRUM_ID / SPECTRUMID, else '<library>:<1-based ordinal>'."""
    return str(spectrum.get('spectrum_id') or f"{library}:{ordinal}")


def _row(spectrum, library: str, ordinal: int) -> Optional[tuple]:
    """A table row for a harmonized matchms Spectrum; None if it's unusable or negative mode."""
    # MzKit has no negative-mode support yet, so those entries could never match
    if str(spectrum.get('ionmode') or '').lower() == 'negative':
        return None
    precursor = spectrum.get('precursor_mz')
    keep = spectrum.peaks.intensities > 0   # a few GNPS entries have negative intensities
    mz, intsy = spectrum.peaks.mz[keep], spectrum.peaks.intensities[keep]
    if not precursor or not len(mz):
        return None
    get = lambda k: str(spectrum.get(k) or '')
    return (
        library, _spectrum_id(spectrum, library, ordinal), get('compound_name'), get('smiles'),
        get('inchikey') or get('inchi_aux'),   # MZmine exports put the InChIKey in INCHIAUX
        get('formula'), get('adduct'), float(precursor),
        np.column_stack([mz, intsy]).astype(np.float32).tobytes(),
    )


def _embedding_path(db_path: Path, embedder: str, suffix: str = '.npy') -> Path:
    return db_path.with_name(f"{db_path.stem}.{embedder}{suffix}")


def build_db(
        mgf_dir: Path = DB_DIR,
        db_path: Path = DEFAULT_DB,
        progress_callback=None,
        cancel_event=None,
) -> Optional[str]:
    """
    (Re)build the cache (and embeddings) from every .mgf in `mgf_dir`.
    Written to temp files and renamed at the end, so a half-built cache is
    never used. Returns the database path, or None if cancelled.
    """
    from matchms.importing import load_from_mgf
    mgf_dir, db_path = Path(mgf_dir), Path(db_path)
    report = progress_callback or (lambda *_: None)
    cancelled = lambda: cancel_event is not None and cancel_event.is_set()
    mgfs = sorted(mgf_dir.glob('*.mgf'))
    if not mgfs:
        raise FileNotFoundError(f"No .mgf libraries in {mgf_dir}")

    tmp = db_path.with_suffix('.tmp')
    tmp.unlink(missing_ok=True)
    tmp_embeddings: dict[str, Path] = {}
    ids: list[tuple[str, str]] = []   # (library, spectrum_id) per row
    built = False
    con = sqlite3.connect(tmp)
    try:
        con.execute(
            "CREATE TABLE spectra (library TEXT, spectrum_id TEXT, name TEXT, smiles TEXT, "
            "inchikey TEXT, formula TEXT, adduct TEXT, precursor_mz REAL, peaks BLOB)"
        )
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        insert = f"INSERT INTO spectra VALUES ({','.join('?' * len(COLUMNS))})"
        for i, mgf in enumerate(mgfs):
            library, batch, n = mgf.stem, [], 0
            # pyteomics reads paths as ASCII; GNPS names aren't
            with open(mgf, encoding='utf-8', errors='replace') as f:
                for ordinal, spectrum in enumerate(load_from_mgf(f), 1):
                    if (row := _row(spectrum, library, ordinal)) is not None:
                        batch.append(row)
                    if len(batch) == _INSERT_BATCH:
                        if cancelled():
                            return None
                        con.executemany(insert, batch)
                        ids.extend(r[:2] for r in batch)
                        n += len(batch)
                        batch = []
                        report(100 * i / len(mgfs), f"{library}: read {n} spectra")
            con.executemany(insert, batch)
            ids.extend(r[:2] for r in batch)
            logging.info(f"{library}: {n + len(batch)} spectra")
        con.execute("CREATE INDEX ix_precursor ON spectra(library, precursor_mz)")
        con.commit()
        last_rowid = con.execute("SELECT max(rowid) FROM spectra").fetchone()[0] or 0
        assert last_rowid == len(ids), "rowids must be contiguous: embedding row i is rowid i + 1"

        if ms2deepscore.model_available():
            path = tmp_embeddings[MS2DEEPSCORE] = _embedding_path(db_path, MS2DEEPSCORE, '.npy.tmp')
            if not _embed_ms2deepscore(con, path, len(ids), report, cancelled):
                return None
        else:
            logging.warning(f"{ms2deepscore.missing_model_message()}: no MS2DeepScore embeddings")
        for embedder, path in _external_embeddings(mgf_dir, db_path, ids).items():
            tmp_embeddings[embedder] = path
        built = True
    finally:
        con.close()
        if not built:   # cancelled or failed
            tmp.unlink(missing_ok=True)
            for path in tmp_embeddings.values():
                path.unlink(missing_ok=True)

    for stale in db_path.parent.glob(f"{db_path.stem}.*.npy"):
        stale.unlink()
    os.replace(tmp, db_path)
    for embedder, path in tmp_embeddings.items():
        os.replace(path, _embedding_path(db_path, embedder))
    return str(db_path)


def _embed_ms2deepscore(con, path: Path, n: int, report, cancelled) -> bool:
    """Embed every row (positive mode) into `path`; False if cancelled."""
    model = ms2deepscore.load_model()
    out = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=(n, model.settings["embedding_dim"]))
    rows = con.execute("SELECT precursor_mz, peaks FROM spectra ORDER BY rowid")
    done = 0
    while batch := rows.fetchmany(_EMBED_BATCH):
        if cancelled():
            del out
            return False
        peaks = [np.frombuffer(blob, np.float32).reshape(-1, 2) for _, blob in batch]
        out[done:done + len(batch)] = model.embed(
            (p[:, 0], p[:, 1], {"precursor_mz": mz, "ionmode": "positive"})
            for (mz, _), p in zip(batch, peaks)
        )
        done += len(batch)
        report(100 * done / n, f"MS2DeepScore: embedded {done}/{n} spectra")
    _normalize_rows(out)
    return True


def _external_embeddings(mgf_dir: Path, db_path: Path, ids: list[tuple[str, str]]) -> dict[str, Path]:
    """Each `<library>.<embedder>.npz` in `mgf_dir`, mapped onto rows: embedder -> its temp file."""
    row_of = {key: i for i, key in enumerate(ids)}
    libraries = {lib for lib, _ in ids}
    out: dict[str, np.ndarray] = {}
    for npz in sorted(mgf_dir.glob('*.npz')):
        library, _, embedder = npz.stem.rpartition('.')
        if library not in libraries or embedder == MS2DEEPSCORE:
            logging.warning(f"{npz.name}: not '<library>.<embedder>.npz' for a known library, skipped")
            continue
        file_ids, embeddings = read_embeddings_npz(npz)
        if embedder not in out:
            out[embedder] = np.lib.format.open_memmap(
                _embedding_path(db_path, embedder, '.npy.tmp'), mode='w+', dtype=np.float16, shape=(len(ids), embeddings.shape[1]),
            )
        elif out[embedder].shape[1] != embeddings.shape[1]:
            raise ValueError(f"{npz.name}: {embeddings.shape[1]}-d, but {embedder} is {out[embedder].shape[1]}-d")
        rows = np.array([row_of.get((library, i), -1) for i in file_ids])
        found = rows >= 0
        out[embedder][rows[found]] = embeddings[found]
        logging.info(f"{npz.name}: {found.sum()}/{len(rows)} ids matched")
    for m in out.values():
        _normalize_rows(m)
    return {embedder: _embedding_path(db_path, embedder, '.npy.tmp') for embedder in out}


def _normalize_rows(m: np.ndarray, chunk: int = 65536):
    """Scale each row of memmap `m` to unit length in place; zero / non-finite rows become NaN."""
    for start in range(0, len(m), chunk):
        block = np.asarray(m[start:start + chunk], np.float32)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            m[start:start + chunk] = np.where(np.isfinite(norms) & (norms > 0), block / norms, np.nan)
    m.flush()


def read_embeddings_npz(path: Path) -> tuple[list[str], np.ndarray]:
    """(ids, n x d embeddings) from an .npz with `ids` and `embeddings`. Raises ValueError if malformed."""
    with np.load(path, allow_pickle=False) as f:
        if 'ids' not in f or 'embeddings' not in f:
            raise ValueError(f"{Path(path).name}: needs 'ids' and 'embeddings' arrays")
        ids, embeddings = [str(i) for i in f['ids']], np.asarray(f['embeddings'], np.float32)
    if embeddings.ndim != 2 or len(ids) != len(embeddings):
        raise ValueError(f"{Path(path).name}: {len(ids)} ids for embeddings of shape {embeddings.shape}")
    return ids, embeddings


def cache_ok(db_path: Path = DEFAULT_DB) -> bool:
    """The cache exists and was built with the current schema."""
    if not Path(db_path).exists():
        return False
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        return con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def available_libraries(db_path: Path = DEFAULT_DB) -> list[str]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        return [r[0] for r in con.execute("SELECT DISTINCT library FROM spectra")]


def embedders(db_path: Path = DEFAULT_DB) -> list[str]:
    """Embedders with a matrix for this cache, MS2DeepScore first."""
    db_path = Path(db_path)
    names = [p.name[len(db_path.stem) + 1:-len('.npy')] for p in db_path.parent.glob(f"{db_path.stem}.*.npy")]
    return sorted(names, key=lambda n: (n != MS2DEEPSCORE, n))


def load_embeddings(embedder: str, db_path: Path = DEFAULT_DB) -> np.ndarray:
    """The (n rows x d) float16 unit-row matrix (NaN: no embedding), memory-mapped; row i is rowid i + 1."""
    return np.load(_embedding_path(Path(db_path), embedder), mmap_mode='r')


def _in(libraries) -> str:
    return f"library IN ({','.join('?' * len(libraries))})"


def window(libraries: list[str], mz: float, tol: float, db_path: Path = DEFAULT_DB) -> list[int]:
    """Rowids in `libraries` with precursor m/z within `mz` +/- `tol` (Da)."""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        return [r[0] for r in con.execute(
            f"SELECT rowid FROM spectra WHERE {_in(libraries)} AND precursor_mz BETWEEN ? AND ?",
            (*libraries, mz - tol, mz + tol),
        )]


def rowids(libraries: list[str], db_path: Path = DEFAULT_DB) -> np.ndarray:
    """All rowids in `libraries`, ascending."""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        return np.array([r[0] for r in con.execute(
            f"SELECT rowid FROM spectra WHERE {_in(libraries)} ORDER BY rowid", libraries,
        )], dtype=np.int64)


def find_rowid(library: str, spectrum_id: str, db_path: Path = DEFAULT_DB) -> Optional[int]:
    """The rowid of `library`'s spectrum `spectrum_id` (None: not in the index)."""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        row = con.execute(
            "SELECT rowid FROM spectra WHERE library = ? AND spectrum_id = ? LIMIT 1", (library, spectrum_id),
        ).fetchone()
    return None if row is None else row[0]


def rows(ids: list[int], db_path: Path = DEFAULT_DB) -> dict[int, tuple]:
    """rowid -> (library, spectrum_id, name, smiles, inchikey, formula, adduct, precursor_mz, SpectrumArray)."""
    out = {}
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        for start in range(0, len(ids), 900):   # SQLite's bound-parameter limit
            chunk = [int(i) for i in ids[start:start + 900]]
            for rowid, *row, blob in con.execute(
                f"SELECT rowid, {','.join(COLUMNS)} FROM spectra WHERE rowid IN ({','.join('?' * len(chunk))})",
                chunk,
            ):
                peaks = np.frombuffer(blob, np.float32).reshape(-1, 2)
                out[rowid] = (*row, to_spec_arr(peaks[:, 0], peaks[:, 1]))
    return out
