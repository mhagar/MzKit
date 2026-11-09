"""
Search the spectral libraries (core/utils/spectrum_db.py) for one analyte's MS2.

Two modes:
 - 'identity': library entries whose precursor is within `precursor_tol_da`
   Da (or, given a `formula`, whose formula is that one), scored by
   CosineGreedy (with its matched peak pairs for the mirror); at least
   `min_matched_peaks` matched, best first.
 - an embedder ('ms2deepscore', or an imported one): analogues anywhere in the
   ticked libraries, ranked by embedding cosine. MS2DeepScore approximates
   structural (ECFP4 Tanimoto) similarity, so the embedding score is the
   ranking; modified cosine is only computed for the top-k, as an extra
   column.

Every cosine (and the modified cosine) scores only peaks above
`min_rel_intensity` of the base peak, outside the precursor region.

With `glacier_params`, each hit with a structure also gets its GLACIER
prediction scored against its library spectrum (`LibraryHit.glacier`), for
hovering library peaks onto the fragments that would explain them.

Positive mode only. Parameters: the `[spectral_search]` config section.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple, Optional, TYPE_CHECKING

import numpy as np
from matchms.similarity import CosineGreedy, ModifiedCosine

from core.cli.align_ensembles import _cosine_with_matches, _spectrum_from
from core.utils import ms2deepscore, spectrum_db

if TYPE_CHECKING:
    from configparser import ConfigParser
    from core.cli.rank_candidates import RankParams, ScoredCandidate
    from core.utils.array_types import SpectrumArray

SECTION = 'spectral_search'
IDENTITY = 'identity'

_CHUNK = 65536   # embedding rows scored at a time


class SpectralSearchParams(NamedTuple):
    """See the `[spectral_search]` section of default_config.ini."""
    precursor_tol_da: float = 0.02
    mz_tolerance: float = 0.01
    min_rel_intensity: float = 0.01
    min_matched_peaks: int = 3
    max_hits: int = 100
    top_k: int = 50


def params_from_config(config: 'ConfigParser') -> SpectralSearchParams:
    """SpectralSearchParams from `[spectral_search]`, falling back to the defaults per key."""
    d = SpectralSearchParams()
    return SpectralSearchParams(
        precursor_tol_da=config.getfloat(SECTION, 'precursor_tol_da', fallback=d.precursor_tol_da),
        mz_tolerance=config.getfloat(SECTION, 'mz_tolerance', fallback=d.mz_tolerance),
        min_rel_intensity=config.getfloat(SECTION, 'min_rel_intensity', fallback=d.min_rel_intensity),
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
    params: SpectralSearchParams = field(default=SpectralSearchParams(), repr=False)   # as scored
    # GLACIER's prediction for the hit's structure, scored against `peaks` (None: none made)
    glacier: Optional['ScoredCandidate'] = field(default=None, repr=False)

    @property
    def label(self) -> str:
        """The name, or the spectrum ID if there's none or it's too long to read (like DbCandidate.label)."""
        from core.cli.rank_candidates import MAX_NAME_LEN
        return self.name if self.name and len(self.name) <= MAX_NAME_LEN else self.spectrum_id


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
    glacier_params: Optional['RankParams'] = None,
    formula: Optional[str] = None,
) -> Optional[SearchResult]:
    """
    Hits for `ms2` in `libraries`, best first; None if cancelled.
    `query_embedding` is required for imported embedders (MS2DeepScore embeds `ms2` itself).
    With `glacier_params`, the hits get GLACIER predictions (if the model is there).

    `formula` (identity only) searches the entries with that formula instead of
    the precursor window.

    :raises ValueError: `formula` doesn't parse, or isn't an identity search.
    """
    if formula is not None and mode != IDENTITY:
        raise ValueError("Formula queries are identity searches only")
    cosine = CosineGreedy(tolerance=params.mz_tolerance)
    min_rel = params.min_rel_intensity
    if mode == IDENTITY:
        ids = (spectrum_db.by_formula(libraries, formula, db_path) if formula is not None
               else spectrum_db.window(libraries, precursor_mz, params.precursor_tol_da, db_path))
        rows = spectrum_db.rows(ids, db_path)
        scored = []
        for rowid, row in rows.items():
            score, matches = _cosine_with_matches(cosine, ms2, row[-1], precursor_mz, row[-2], min_rel)
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
             _cosine_with_matches(cosine, ms2, rows[int(i) + 1][-1], precursor_mz, rows[int(i) + 1][-2],
                                  min_rel)[1])
            for i in top
        ]

    hits = _hits(ms2, precursor_mz, scored, rows, params)
    if glacier_params is not None and not predict_hits(hits, glacier_params, cancel_event):
        return None
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
    glacier_params: Optional['RankParams'] = None,
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
        params.min_rel_intensity,
    )
    hits = _hits(ms2, precursor_mz, [(score, rowid, matches)], rows, params)
    if glacier_params is not None:
        predict_hits(hits, glacier_params)
    return hits[0]


def predict_hits(hits: list[LibraryHit], params: 'RankParams', cancel_event=None) -> bool:
    """
    Set each hit's `glacier`: its structure predicted as its own adduct
    ([M+H]+ if GLACIER doesn't know it) and scored against its library
    spectrum. Quietly does nothing without the GLACIER model. False if cancelled.
    """
    from core.cli.rank_candidates import DbCandidate, _predict, glacier_adduct, scored_prediction
    from core.utils import glacier

    if not glacier.model_available():
        return True
    model = glacier.load_model()
    by_adduct = defaultdict(list)
    for h in hits:
        if h.smiles:
            by_adduct[glacier_adduct(h.adduct or None, model.meta['ion2onehot_pos'])[0]].append(h)
    step = max(params.batch_size, 1)
    for adduct, group in by_adduct.items():
        for start in range(0, len(group), step):
            if cancel_event is not None and cancel_event.is_set():
                return False
            batch = group[start:start + step]
            for h, pred in zip(batch, _predict(model, [h.smiles for h in batch], adduct, params)):
                if pred is not None:
                    h.glacier = scored_prediction(
                        model, DbCandidate(h.library, h.spectrum_id, h.name, h.smiles, ""), pred,
                        h.peaks, h.precursor_mz, adduct, np.array([], int), params,
                    )
    return True


def _hits(ms2, precursor_mz, scored, rows, params) -> list[LibraryHit]:
    """LibraryHits for (score, rowid, matches) of `rows`, with modified cosine for display."""
    modified = ModifiedCosine(tolerance=params.mz_tolerance)
    query = _spectrum_from(ms2['mz'], ms2['intsy'], precursor_mz, params.min_rel_intensity)
    query.set('precursor_mz', float(precursor_mz))
    hits = []
    for score, rowid, matches in scored:
        library, spectrum_id, name, smiles, inchikey, _formula, adduct, lib_mz, peaks = rows[rowid]
        lib = _spectrum_from(peaks['mz'], peaks['intsy'], lib_mz, params.min_rel_intensity)
        lib.set('precursor_mz', lib_mz)
        mod = modified.pair(query, lib)
        hits.append(LibraryHit(
            rowid, library, spectrum_id, name, smiles, inchikey, adduct, lib_mz, peaks,
            score, float(mod['score']), int(mod['matches']), lib_mz - float(precursor_mz), matches, params,
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
