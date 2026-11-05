"""
Rank database structure candidates for one analyte by how well their GLACIER
predicted MS2 (core/utils/glacier.py) explains its experimental MS2.

Two scores, kept separate:
 - explained k/N: of the N most intense experimental fragments (precursor
   excluded), how many have a predicted peak within tolerance that rises
   above GLACIER's grass. Predicted intensity beyond that is ignored, since
   GLACIER's intensities are unreliable. The primary sort key.
 - cosine (CosineGreedy): the tiebreaker, with its matched peak pairs.

Candidates come from the compound cache (core/utils/compound_db.py).
Parameters: the `[glacier]` config section.
"""
import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional, TYPE_CHECKING

import numpy as np

from core.utils import compound_db
from core.utils.array_types import to_spec_arr
from core.utils.spectra import informative_mask

if TYPE_CHECKING:
    from configparser import ConfigParser
    from core.utils.array_types import SpectrumArray

logger = logging.getLogger(__name__)

SECTION = 'glacier'


class RankParams(NamedTuple):
    """See the `[glacier]` section of default_config.ini."""
    collision_energy: float = 30.0
    instrument: str = 'Orbitrap'
    mz_tolerance: float = 0.01
    top_n_peaks: int = 10
    min_exp_rel: float = 0.05
    min_pred_rel: float = 0.01
    mass_ppm: float = 5.0
    batch_size: int = 32


def rank_params_from_config(config: 'ConfigParser') -> RankParams:
    """RankParams from `[glacier]`, falling back to the defaults per key."""
    d = RankParams()
    return RankParams(
        collision_energy=config.getfloat(SECTION, 'collision_energy', fallback=d.collision_energy),
        instrument=config.get(SECTION, 'instrument', fallback=d.instrument),
        mz_tolerance=config.getfloat(SECTION, 'mz_tolerance', fallback=d.mz_tolerance),
        top_n_peaks=config.getint(SECTION, 'top_n_peaks', fallback=d.top_n_peaks),
        min_exp_rel=config.getfloat(SECTION, 'min_exp_rel', fallback=d.min_exp_rel),
        min_pred_rel=config.getfloat(SECTION, 'min_pred_rel', fallback=d.min_pred_rel),
        mass_ppm=config.getfloat(SECTION, 'mass_ppm', fallback=d.mass_ppm),
        batch_size=config.getint(SECTION, 'batch_size', fallback=d.batch_size),
    )


# Longer names (systematic ones, which databases often use as the name) read
# no better than a SMILES: the candidate goes by its database ID instead
MAX_NAME_LEN = 40


@dataclass(frozen=True)
class DbCandidate:
    source: str
    db_id: str
    name: str
    smiles: str
    formula: str
    organism_type: str = ""
    genus: str = ""         # pipe-joined
    organisms: str = ""     # pipe-joined
    np_class: str = ""

    @property
    def label(self) -> str:
        """The name, or the database ID if there's none or it's too long to read."""
        return self.name if self.name and len(self.name) <= MAX_NAME_LEN else self.db_id

    @property
    def organism(self) -> str:
        """
        e.g. "Bacteria: Streptomyces; Nocardia" ("" if none known).
        """
        genus = self.genus.replace('|', '; ')
        return f"{self.organism_type}: {genus}" if self.organism_type and genus else self.organism_type or genus


@dataclass
class ScoredCandidate:
    candidate: DbCandidate
    explained: int                    # k of RankResult.n_considered
    cosine: float
    prediction: 'SpectrumArray'       # merged, base peak = 1
    frag_atoms: np.ndarray            # (n predicted peaks, n heavy atoms) bool
    matches: list[tuple[int, int]] = field(default_factory=list)   # (exp, pred) idx
    explained_idx: list[int] = field(default_factory=list)         # exp peaks explained
    # Per predicted peak: its ion formula (glacier.ion_formula, e.g. "C7H7O+") and
    # the H shift GLACIER applied to its fragment
    ions: list[str] = field(default_factory=list)
    h_shifts: list[int] = field(default_factory=list)


@dataclass
class RankResult:
    scored: list[ScoredCandidate] = field(repr=False)   # best first (repr: ProcessController logs results)
    n_considered: int                 # N of explained k/N
    adduct: str                       # as predicted
    adduct_fallback: bool             # the analyte's adduct isn't one GLACIER knows
    params: RankParams
    skipped: list[DbCandidate] = field(default_factory=list, repr=False)  # GLACIER couldn't predict


def fetch_candidates(
    sources: list[str],
    formula: Optional[str] = None,
    mass: Optional[float] = None,
    ppm: float = RankParams().mass_ppm,
    db_path=compound_db.DEFAULT_DB,
    text_field: Optional[str] = None,
    text: Optional[str] = None,
) -> tuple[list[DbCandidate], bool]:
    """
    Compounds matching `formula`, else `text` in `text_field` (a
    compound_db.TEXT_FIELDS label, e.g. 'Name'), else neutral `mass` +/- `ppm`,
    in each of `sources`, merged in source order, one per InChIKey-14
    (stereoisomers and cross-database duplicates collapse onto the first seen).

    :return: (candidates, whether any source hit compound_db.MAX_RESULTS).
    :raises ValueError: if `formula` doesn't parse.
    """
    candidates: list[DbCandidate] = []
    seen: set = set()
    capped = False
    for source in sources:
        if formula is not None:
            rows = compound_db.search_formula(source, formula, db_path)
        elif text_field is not None:
            rows = compound_db.search_text(source, text_field, text, db_path)
        else:
            rows = compound_db.search_mass(source, mass, ppm, db_path)
        capped |= len(rows) >= compound_db.MAX_RESULTS
        for row in rows:
            db_id, name, smiles, formula_, _mass, inchikey, org_type, genus, organisms, np_class = row[:10]
            key = inchikey[:14] if inchikey else (source, db_id)
            if key in seen or not smiles:
                continue
            seen.add(key)
            candidates.append(DbCandidate(
                source, db_id, name, smiles, formula_,
                org_type or "", genus or "", organisms or "", np_class or "",
            ))
    return candidates, capped


def ion_name(adduct: Optional[str], charge: Optional[int]) -> Optional[str]:
    """find-mfs's adduct ("H", "-H", "Na") and charge (None: +1) -> "[M+H]+" etc."""
    if not adduct:
        return None
    z = charge or 1
    sign = '+' if z > 0 else '-'
    n = '' if abs(z) == 1 else str(abs(z))
    return f"[M{'' if adduct.startswith('-') else '+'}{adduct}]{n}{sign}"


def neutral_mass(mz: float, adduct: Optional[str]) -> float:
    """Neutral mass for an observed m/z; [M+H]+ when `adduct` is unknown."""
    from core.cli.adduct_table import adducts_for_polarity
    table = {a.name: a for a in adducts_for_polarity(None)}
    return table.get(adduct, table['[M+H]+']).neutral_mass(mz)


def glacier_adduct(adduct: Optional[str], known) -> tuple[str, bool]:
    """(the adduct to predict with, whether it's a fallback for `adduct`)."""
    if adduct in known:
        return adduct, False
    return ('[M-H]-' if adduct and adduct.endswith('-') else '[M+H]+'), True


def considered_peaks(
    ms2: 'SpectrumArray',
    precursor_mz: float,
    params: RankParams,
) -> np.ndarray:
    """
    Indices into `ms2` of the peaks explained-k/N is scored over: the most
    intense fragments in the informative window (below the precursor
    region), relative to the base fragment there.
    """
    idx = np.flatnonzero(informative_mask(ms2['mz'], precursor_mz))
    if not idx.size:
        return idx
    idx = idx[ms2['intsy'][idx] >= params.min_exp_rel * ms2['intsy'][idx].max()]
    return idx[np.argsort(-ms2['intsy'][idx], kind='stable')][:params.top_n_peaks]


def explained_peaks(
    ms2: 'SpectrumArray',
    considered: np.ndarray,
    prediction: 'SpectrumArray',
    params: RankParams,
) -> list[int]:
    """The `considered` experimental peaks a predicted peak above the grass explains."""
    if not prediction.size:
        return []
    pred = prediction['mz'][prediction['intsy'] >= params.min_pred_rel * prediction['intsy'].max()]
    return [
        int(i) for i in considered
        if pred.size and np.abs(pred - ms2['mz'][i]).min() <= params.mz_tolerance
    ]


def _predict(model, smiles: list[str], adduct: str, params: RankParams) -> list[Optional[dict]]:
    """One prediction per SMILES; None where GLACIER fails (a bad molecule
    fails its whole batch, so the batch is retried one at a time)."""
    n = len(smiles)
    args = (np.full(n, params.collision_energy), [adduct] * n, [params.instrument] * n)
    try:
        return model.predict(smiles, *args)
    except Exception:
        out = []
        for s in smiles:
            try:
                out.append(model.predict([s], *(a[:1] for a in args))[0])
            except Exception as e:
                logger.info(f"GLACIER skipped {s}: {e}")
                out.append(None)
        return out


def rank_candidates(
    ms2: 'SpectrumArray',
    precursor_mz: float,
    adduct: Optional[str],
    candidates: list[DbCandidate],
    params: RankParams = RankParams(),
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Optional[RankResult]:
    """
    Predict each candidate's MS2 and score it against `ms2`; best first by
    (explained, cosine). None if cancelled.
    """
    from matchms.similarity import CosineGreedy

    from core.cli.align_ensembles import _cosine_with_matches
    from core.utils import glacier

    model = glacier.load_model()
    used_adduct, fallback = glacier_adduct(adduct, model.meta['ion2onehot_pos'])
    considered = considered_peaks(ms2, precursor_mz, params)
    cosine = CosineGreedy(tolerance=params.mz_tolerance, mz_power=0.0, intensity_power=1.0)

    scored: list[ScoredCandidate] = []
    skipped: list[DbCandidate] = []
    step = max(params.batch_size, 1)
    for start in range(0, len(candidates), step):
        if cancel_event is not None and cancel_event.is_set():
            return None
        batch = candidates[start:start + step]
        for cand, pred in zip(batch, _predict(model, [c.smiles for c in batch], used_adduct, params)):
            if pred is None:
                skipped.append(cand)
                continue
            mz, intsy, src = glacier.merge_peaks(pred['spec'])
            prediction = to_spec_arr(mz, intsy)
            # Both sides share the precursor: GLACIER's intact-ion peak must not score
            score, matches = _cosine_with_matches(cosine, ms2, prediction, precursor_mz, precursor_mz)
            explained = explained_peaks(ms2, considered, prediction, params)
            scored.append(ScoredCandidate(
                cand, len(explained), score, prediction, pred['frag'][src],
                matches, explained,
                ions=[glacier.ion_formula(pred['form'][r], pred['h_shift'][r], used_adduct,
                                          pred['radical'][r], model.elements) for r in src],
                h_shifts=pred['h_shift'][src].tolist(),
            ))
        if progress_callback is not None:
            done = min(start + step, len(candidates))
            progress_callback(100 * done / len(candidates), f"GLACIER {done}/{len(candidates)}")

    scored.sort(key=lambda s: (s.explained, s.cosine), reverse=True)
    return RankResult(scored, len(considered), used_adduct, fallback, params, skipped)
