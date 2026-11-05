"""
Search the spectral libraries (core/utils/spectrum_db.py) for one analyte's MS2.

Two modes:
 - 'identity': library entries whose precursor is within `precursor_tolerance`
   Da, scored by CosineGreedy (with its matched peak pairs for the mirror);
   at least `min_matched_peaks` matched, best first.
 - an embedder ('ms2deepscore', or an imported one): analogues anywhere in the
   ticked libraries, ranked by embedding cosine. MS2DeepScore approximates
   structural (ECFP4 Tanimoto) similarity, so the embedding score is the
   ranking; modified cosine is only computed for the top-k, as an extra
   column.

Positive mode only. Parameters: the `[spectral_search]` config section.
"""
from dataclasses import dataclass, field
from typing import NamedTuple, Optional, TYPE_CHECKING

import numpy as np
from matchms.similarity import CosineGreedy, ModifiedCosine

from core.cli.align_ensembles import _cosine_with_matches, _spectrum_from
from core.utils import ms2deepscore, spectrum_db

if TYPE_CHECKING:
    from configparser import ConfigParser
    from core.utils.array_types import SpectrumArray

SECTION = 'spectral_search'
IDENTITY = 'identity'

_CHUNK = 65536   # embedding rows scored at a time


class SpectralSearchParams(NamedTuple):
    """See the `[spectral_search]` section of default_config.ini."""
    precursor_tolerance: float = 0.02
    mz_tolerance: float = 0.01
    min_matched_peaks: int = 3
    max_hits: int = 100
    top_k: int = 50


def params_from_config(config: 'ConfigParser') -> SpectralSearchParams:
    """SpectralSearchParams from `[spectral_search]`, falling back to the defaults per key."""
    d = SpectralSearchParams()
    return SpectralSearchParams(
        precursor_tolerance=config.getfloat(SECTION, 'precursor_tolerance', fallback=d.precursor_tolerance),
        mz_tolerance=config.getfloat(SECTION, 'mz_tolerance', fallback=d.mz_tolerance),
        min_matched_peaks=config.getint(SECTION, 'min_matched_peaks', fallback=d.min_matched_peaks),
        max_hits=config.getint(SECTION, 'max_hits', fallback=d.max_hits),
        top_k=config.getint(SECTION, 'top_k', fallback=d.top_k),
    )


@dataclass
class LibraryHit:
    rowid: int
    library: str
    spectrum_id: str
    name: str
    smiles: str
    inchikey: str
    adduct: str
    precursor_mz: float
    peaks: 'SpectrumArray' = field(repr=False)
    score: float                      # cosine (identity) or embedding cosine (analogue)
    mod_cosine: float                 # display only
    n_matched: int                    # modified cosine's matched peaks
    delta_mz: float                   # library - query precursor
    matches: list[tuple[int, int]] = field(default_factory=list, repr=False)   # (query, library) peak idx


@dataclass
class SearchResult:
    hits: list[LibraryHit] = field(repr=False)   # best first (repr: ProcessController logs results)
    mode: str                                     # IDENTITY or the embedder
    n_searched: int
    params: SpectralSearchParams


def search_spectra(
    ms2: 'SpectrumArray',
    precursor_mz: float,
    libraries: list[str],
    mode: str = IDENTITY,
    params: SpectralSearchParams = SpectralSearchParams(),
    query_embedding: Optional[np.ndarray] = None,
    db_path=spectrum_db.DEFAULT_DB,
    progress_callback=None,
    cancel_event=None,
) -> Optional[SearchResult]:
    """
    Hits for `ms2` in `libraries`, best first; None if cancelled.
    `query_embedding` is required for imported embedders (MS2DeepScore embeds `ms2` itself).
    """
    cosine = CosineGreedy(tolerance=params.mz_tolerance)
    if mode == IDENTITY:
        ids = spectrum_db.window(libraries, precursor_mz, params.precursor_tolerance, db_path)
        rows = spectrum_db.rows(ids, db_path)
        scored = []
        for rowid, row in rows.items():
            score, matches = _cosine_with_matches(cosine, ms2, row[-1], precursor_mz, row[-2])
            if len(matches) >= params.min_matched_peaks:
                scored.append((score, rowid, matches))
        scored.sort(key=lambda s: -s[0])
        scored = scored[:params.max_hits]
        n_searched = len(ids)
    else:
        if query_embedding is None:
            if mode != spectrum_db.MS2DEEPSCORE:
                raise ValueError(f"No query embedding for {mode}")
            query_embedding = ms2deepscore.load_model().embed(
                [(ms2['mz'], ms2['intsy'], {"precursor_mz": float(precursor_mz), "ionmode": "positive"})]
            )[0]
        emb = spectrum_db.load_embeddings(mode, db_path)
        scores, n_searched = _embedding_scores(
            emb, np.asarray(query_embedding, np.float32),
            spectrum_db.rowids(libraries, db_path), cancel_event,
        )
        if scores is None:
            return None
        k = min(params.top_k, n_searched)
        top = np.argpartition(-scores, k - 1)[:k] if k else np.array([], int)
        # Exact (float32) scores for the top k, which the float16 pass only shortlisted
        query = np.asarray(query_embedding, np.float32)
        scores[top] = np.asarray(emb[top], np.float32) @ (query / (np.linalg.norm(query) or 1.0))
        top = top[np.argsort(-scores[top])]
        rows = spectrum_db.rows([int(i) + 1 for i in top], db_path)
        scored = [
            (float(scores[i]), int(i) + 1,
             _cosine_with_matches(cosine, ms2, rows[int(i) + 1][-1], precursor_mz, rows[int(i) + 1][-2])[1])
            for i in top
        ]

    hits = _hits(ms2, precursor_mz, scored, rows, params)
    if progress_callback is not None:
        progress_callback(100.0, f"{len(hits)} hits of {n_searched} searched")
    return SearchResult(hits, mode, n_searched, params)


def library_hit(
    ms2: 'SpectrumArray',
    precursor_mz: float,
    library: str,
    spectrum_id: str,
    params: SpectralSearchParams = SpectralSearchParams(),
    db_path=spectrum_db.DEFAULT_DB,
) -> Optional[LibraryHit]:
    """
    One library spectrum as a LibraryHit against `ms2` (identity cosine), e.g.
    to mirror an accepted hypothesis again; None if it isn't in the index.
    """
    rowid = spectrum_db.find_rowid(library, spectrum_id, db_path)
    if rowid is None:
        return None
    rows = spectrum_db.rows([rowid], db_path)
    score, matches = _cosine_with_matches(
        CosineGreedy(tolerance=params.mz_tolerance), ms2, rows[rowid][-1], precursor_mz, rows[rowid][-2],
    )
    return _hits(ms2, precursor_mz, [(score, rowid, matches)], rows, params)[0]


def _hits(ms2, precursor_mz, scored, rows, params) -> list[LibraryHit]:
    """LibraryHits for (score, rowid, matches) of `rows`, with modified cosine for display."""
    modified = ModifiedCosine(tolerance=params.mz_tolerance)
    query = _spectrum_from(ms2['mz'], ms2['intsy'], precursor_mz)
    query.set('precursor_mz', float(precursor_mz))
    hits = []
    for score, rowid, matches in scored:
        library, spectrum_id, name, smiles, inchikey, _formula, adduct, lib_mz, peaks = rows[rowid]
        lib = _spectrum_from(peaks['mz'], peaks['intsy'], lib_mz)
        lib.set('precursor_mz', lib_mz)
        mod = modified.pair(query, lib)
        hits.append(LibraryHit(
            rowid, library, spectrum_id, name, smiles, inchikey, adduct, lib_mz, peaks,
            score, float(mod['score']), int(mod['matches']), lib_mz - float(precursor_mz), matches,
        ))
    return hits


def _embedding_scores(embeddings, query, rowids, cancel_event) -> tuple[Optional[np.ndarray], int]:
    """
    Cosine of `query` with every (unit, float16) embedding row; rows outside
    `rowids`, and NaN (unembedded) rows, score -inf. Scored in float16:
    converting the matrix to float32 costs ~10x the product. Chunked so the
    memmap is read once.
    """
    query = (query / (np.linalg.norm(query) or 1.0)).astype(np.float16)
    scores = np.empty(len(embeddings), np.float32)
    for start in range(0, len(embeddings), _CHUNK):
        if cancel_event is not None and cancel_event.is_set():
            return None, 0
        scores[start:start + _CHUNK] = embeddings[start:start + _CHUNK] @ query
    keep = np.zeros(len(embeddings), bool)
    keep[rowids - 1] = True
    scores[~keep | np.isnan(scores)] = -np.inf
    return scores, int(np.isfinite(scores).sum())
