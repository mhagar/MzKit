"""
Cluster an alignment's analytes by an externally computed MS2 similarity
(i.e. MS2DeepScore, SIMBA, ...) instead of modified cosine.

Round trip:
1. The consensus .mgf from `export_table` has one MS2 entry per analyte,
   tagged with a stable analyte id (`ANALYTE_ID_TAG`). Analyte UUIDs
   survive filtering, so a matrix scored on a parent alignment also
   applies to its filtered children.

2. The user scores the MGF however they like and writes a square
   similarity (or distance) matrix as CSV/TSV, labelled by those ids in
   the header row and first column - i.e.
   `pd.DataFrame(m, index=ids, columns=ids).to_csv(path)`.

3. `read_similarity_csv` loads it (`RawMatrix`);
   `transform` maps it onto an alignment's analytes under `ImportParams`;
   `cluster_imported` clusters the result
            (`cluster_analytes.cluster_from_similarity`).

Transform, in order (see `transform`):
    match ids -> NaN = unrelated -> clip to [clip_min, clip_max] and
    rescale to [0, 1] (inverted for distances) -> symmetrize (mean, or
    min with a bidirectional threshold) -> floor -> diagonal = 1.

The floor matters: learned scores almost never reach exactly 0, and
clustering treats only 0 as "unrelated" (see cluster_analytes).

TODO: imported matrices only exist for the session (held by the GUI).
TODO: Implement persistence (a `SimilarityMatrix` in DataRegistry, saved in .mzk) so
they survive a project reload.
"""
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Optional, TYPE_CHECKING
import uuid

import numpy as np

from core.cli.cluster_analytes import AnalyteClustering, cluster_from_similarity

if TYPE_CHECKING:
    from core.data_structs.alignment import AlignedAnalyte, EnsembleAlignment


ANALYTE_ID_TAG = 'MZKIT_ANALYTE'

HISTOGRAM_BINS = 40


# --- Reading ------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class RawMatrix:
    """
    A labelled square matrix as read from file, untransformed.
    Hashes by identity; `uid` identifies it in cache keys.
    """
    name: str
    ids: tuple[str, ...]
    values: np.ndarray
    path: Optional[Path] = None
    uid: int = field(default_factory=lambda: uuid.uuid4().int)


def read_similarity_csv(path: Path) -> RawMatrix:
    """
    Read a square, labelled matrix (header row and first column hold the
    same ids, in any order). Tab-separated unless the file is .csv.
    Empty cells / 'nan' read as NaN.

    Raises ValueError on a malformed matrix.
    """
    path = Path(path)
    delimiter = ',' if path.suffix.lower() == '.csv' else '\t'
    with path.open(newline='') as f:
        rows = [row for row in csv.reader(f, delimiter=delimiter) if row]
    if len(rows) < 2:
        raise ValueError(f"{path.name}: needs a header row and data rows")

    col_ids = [c.strip() for c in rows[0][1:]]
    row_ids = [r[0].strip() for r in rows[1:]]
    n = len(row_ids)
    if len(col_ids) != n:
        raise ValueError(
            f"{path.name}: not square ({n} rows, {len(col_ids)} columns)"
        )
    if len(set(row_ids)) != n:
        raise ValueError(f"{path.name}: duplicate row ids")
    if set(row_ids) != set(col_ids):
        raise ValueError(f"{path.name}: row and column ids differ")

    values = np.empty((n, n))
    for i, row in enumerate(rows[1:]):
        cells = row[1:]
        if len(cells) != n:
            raise ValueError(
                f"{path.name}: row {i + 1} has {len(cells)} values, "
                f"expected {n}"
            )
        try:
            values[i] = [float(c) if c.strip() else np.nan for c in cells]
        except ValueError as e:
            raise ValueError(f"{path.name}: row {i + 1}: {e}") from None

    # Reorder columns to match rows
    col_of = {c: j for j, c in enumerate(col_ids)}
    values = values[:, [col_of[r] for r in row_ids]]

    return RawMatrix(
        name=path.stem, ids=tuple(row_ids), values=values, path=path,
    )


def write_similarity_csv(
    path: Path, ids: list[str], values: np.ndarray,
) -> None:
    """Inverse of `read_similarity_csv`: tab-separated unless .csv."""
    path = Path(path)
    delimiter = ',' if path.suffix.lower() == '.csv' else '\t'
    with path.open('w', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow([''] + ids)
        for id_, row in zip(ids, values):
            writer.writerow([id_] + [f"{v:.6g}" for v in row])


# --- Transform ----------------------------------------------------------------

class ImportParams(NamedTuple):
    """
    is_distance: values are distances (larger = less similar).
    clip_min / clip_max: values are clipped to this range, then rescaled
        to [0, 1]. For distances, clip_max is the "fully unrelated"
        distance.
    bidirectional: keep a pair only if both directions score at least
        `bidirectional_threshold` (after rescaling), taking the smaller;
        otherwise the two directions are averaged.
    floor: (symmetrized) scores below this become 0, i.e. unrelated.
    """
    is_distance: bool = False
    clip_min: float = 0.0
    clip_max: float = 1.0
    bidirectional: bool = False
    bidirectional_threshold: float = 0.7
    floor: float = 0.6


@dataclass(frozen=True)
class ImportReport:
    """
    n_ids: ids in the matrix. n_matched: of those, analytes in the
    alignment. n_unknown: ids not in the alignment. n_uncovered:
    alignment analytes missing from the matrix.
    n_nan / n_clipped: matched off-diagonal cells that were NaN / clipped.
    max_asymmetry: largest |A->B - B->A| after rescaling.
    n_pairs / n_pairs_kept: matched pairs, and those scoring > 0 in the end.
    histogram / bin_edges: symmetrized pair scores before the floor.
    """
    n_ids: int
    n_matched: int
    n_unknown: int
    n_uncovered: int
    n_nan: int
    n_clipped: int
    max_asymmetry: float
    n_pairs: int
    n_pairs_kept: int
    histogram: np.ndarray
    bin_edges: np.ndarray

    def format_text(self) -> str:
        lines = [
            f"Matched {self.n_matched} of {self.n_ids} matrix ids to analytes",
        ]
        if self.n_unknown:
            lines.append(f"{self.n_unknown} ids not in this alignment (ignored)")
        if self.n_uncovered:
            lines.append(
                f"{self.n_uncovered} analytes not in the matrix (not clustered)"
            )
        if self.n_nan:
            lines.append(f"{self.n_nan} missing values (treated as unrelated)")
        if self.n_clipped:
            lines.append(f"{self.n_clipped} values clipped")
        if self.max_asymmetry > 1e-6:
            lines.append(f"Max asymmetry {self.max_asymmetry:.3f}")
        lines.append(
            f"{self.n_pairs_kept} of {self.n_pairs} pairs related after floor"
        )
        return '\n'.join(lines)


class TransformResult(NamedTuple):
    analytes: list['AlignedAnalyte']
    similarity: np.ndarray
    report: ImportReport


def transform(
    raw: RawMatrix,
    alignment: 'EnsembleAlignment',
    params: ImportParams = ImportParams(),
) -> TransformResult:
    """
    Map `raw` onto `alignment`'s analytes as a similarity in [0, 1]
    (0 = unrelated, diagonal 1), for the matched analytes (in alignment
    order).

    Raises ValueError if clip_max <= clip_min.
    """
    if params.clip_max <= params.clip_min:
        raise ValueError("Clip max must be greater than clip min")

    index_of = {id_: i for i, id_ in enumerate(raw.ids)}
    analytes, idx = [], []
    for analyte in alignment.analytes:
        i = index_of.get(str(analyte.uuid))
        if i is not None:
            analytes.append(analyte)
            idx.append(i)
    n = len(idx)

    values = raw.values[np.ix_(idx, idx)].astype(float)
    off_diag = ~np.eye(n, dtype=bool)

    nan = np.isnan(values)
    n_nan = int(np.count_nonzero(nan & off_diag))
    values[nan] = params.clip_max if params.is_distance else params.clip_min

    clipped = (values < params.clip_min) | (values > params.clip_max)
    n_clipped = int(np.count_nonzero(clipped & off_diag))
    values = np.clip(values, params.clip_min, params.clip_max)
    values = (values - params.clip_min) / (params.clip_max - params.clip_min)
    if params.is_distance:
        values = 1.0 - values

    max_asymmetry = float(np.abs(values - values.T).max()) if n else 0.0
    if params.bidirectional:
        sim = np.minimum(values, values.T)
        sim[sim < params.bidirectional_threshold] = 0.0
    else:
        sim = 0.5 * (values + values.T)

    upper = np.triu_indices(n, k=1)
    histogram, bin_edges = np.histogram(
        sim[upper], bins=HISTOGRAM_BINS, range=(0.0, 1.0),
    )

    sim[sim < params.floor] = 0.0
    np.fill_diagonal(sim, 1.0)

    report = ImportReport(
        n_ids=len(raw.ids),
        n_matched=n,
        n_unknown=len(raw.ids) - n,
        n_uncovered=len(alignment.analytes) - n,
        n_nan=n_nan,
        n_clipped=n_clipped,
        max_asymmetry=max_asymmetry,
        n_pairs=len(upper[0]),
        n_pairs_kept=int(np.count_nonzero(sim[upper] > 0)),
        histogram=histogram,
        bin_edges=bin_edges,
    )
    return TransformResult(analytes, sim, report)


# --- Clustering ---------------------------------------------------------------

class ImportedClusterSpec(NamedTuple):
    """Cache key / params for a clustering by an imported matrix."""
    matrix_uid: int
    matrix_name: str
    params: ImportParams
    linkage_method: str = 'average'

    @property
    def source_name(self) -> str:
        return self.matrix_name


def cluster_imported(
    alignment: 'EnsembleAlignment',
    raw: RawMatrix,
    spec: ImportedClusterSpec,
    progress_callback=None,
    cancel_event=None,
) -> Optional[AnalyteClustering]:
    """
    Cluster `alignment`'s analytes by `raw` transformed under
    `spec.params`. ProcessController-compatible.
    """
    analytes, similarity, report = transform(raw, alignment, spec.params)
    return cluster_from_similarity(
        analytes, similarity, spec.linkage_method,
        n_dropped=report.n_uncovered,
        params=spec,
        dropped_reason=f"not in {raw.name}",
    )
