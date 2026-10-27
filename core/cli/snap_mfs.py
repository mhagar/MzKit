"""
SNAP-MFs: infer compound families behind molecular formulae asserted to be related.

This is a port of snapmf's formula-list mode (~/Dropbox/snapmf,
``network_from_formula_list``), without Cytoscape/GraphML support.

The analysis proceeds as follows:

1. Query the database for each formula. A formula may have zero or many hits.
2. Link compounds from different formulae when their similarity is at least
   the cutoff. Two similarity metrics are available:

   - ``TANIMOTO``: Morgan fingerprints (radius 2, counts) with Tanimoto
     similarity.
   - ``MCES``: normalized MCES, ``J = e_c^2 / (e_a * e_b)``, from
     ``core.utils.mces``. This is slow because it solves one ILP per pair.

3. Drop connected families containing fewer than ``min_size`` compounds.
4. Rank families by the number of query formulae they explain. Families tied
   for the highest score are the top candidates.

The database is a source in the compound cache (``core/utils/compound_db.py``).
The "class" is COCONUT's NP Classifier class; for other sources it is the
producing organism, e.g. "Bacteria Streptomyces" (NPAtlas) or "Marine"
(AntiMarin).

TODO: incorporate GLACIER?
"""

import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from core.formula.manual import canonical_formula
from core.utils.compound_db import DEFAULT_DB

TANIMOTO_CUTOFF = 0.5   # comparable to snapmf's Dice 0.66
MIN_FAMILY_SIZE = 3     # snapmf's min_atlas_size
MCES_TIME_LIMIT = 10.0  # s per pair; undecided pairs aren't linked

TANIMOTO = 'tanimoto'
MCES = 'mces'
METRICS = {
    TANIMOTO: 'Tanimoto (ECFP4)',
    MCES: 'Normalized MCES (J)',
}


@dataclass(frozen=True)
class DbCompound:
    """A database compound used to construct candidate families."""

    identifier: str
    name: str
    smiles: str
    formula: str
    np_class: str  # NP Classifier class (COCONUT) or organism (others)


@dataclass(frozen=True)
class Family:
    """A connected family of compounds and the query formulae it explains."""

    compounds: list[DbCompound]
    formulas: frozenset[str]

    @property
    def score(self) -> int:
        """Return the number of distinct query formulae explained."""
        return len(self.formulas)

    def top_class(self) -> str:
        """Return the most common member class, or an empty string if none."""
        classes = Counter(c.np_class for c in self.compounds if c.np_class)
        return classes.most_common(1)[0][0] if classes else ''


def load_matches(
    source: str,
    formulas: set[str],
    log: Callable[[str], None] = lambda _: None,
    db_path: Path = DEFAULT_DB,
) -> list[DbCompound]:
    """Load cached compounds from ``source`` that match ``formulas``.

    Results retain database row order. Only the first row for each InChIKey-14
    is kept, so stereoisomers count as the same compound. Rows without an
    InChIKey remain distinct. For non-COCONUT sources, the organism type and
    genus are stored in ``np_class`` instead of an NP Classifier class.

    :param source: Compound-cache source to query.
    :param formulas: Canonical molecular formulae in Hill order.
    :param log: Callback for progress messages.
    :param db_path: Path to the compound cache.
    :return: Matching, connectivity-deduplicated compounds.
    """
    formula_list = list(formulas)
    placeholders = ','.join('?' * len(formula_list))

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        rows = con.execute(
            "SELECT db_id, name, smiles, formula, inchikey, np_class, "
            "organism_type, genus "
            "FROM compounds "
            f"WHERE source = ? AND formula IN ({placeholders}) "
            "ORDER BY rowid",
            (source, *formula_list),
        ).fetchall()

    compounds: list[DbCompound] = []
    seen: set[str] = set()

    for db_id, name, smiles, formula, inchikey, np_class, org_type, genus in rows:
        # Without a key, use the unique database ID so rows are never merged.
        key14 = inchikey[:14] if inchikey else db_id
        if key14 in seen:
            continue

        seen.add(key14)

        if source != 'COCONUT':
            np_class = f"{org_type} {genus}".strip()
        compounds.append(DbCompound(db_id, name, smiles, formula, np_class))

    log(
        f"Flattened stereoisomers: {len(rows)} entries -> {len(compounds)} "
        f"distinct InChIKey-14"
    )

    return compounds


def snap_mfs(
    formulas: list[str],
    source: str,
    cutoff: float = TANIMOTO_CUTOFF,
    min_size: int = MIN_FAMILY_SIZE,
    metric: str = TANIMOTO,
    log: Callable[[str], None] | None = None,
    db_path: Path = DEFAULT_DB,
) -> list[Family]:
    """
    Find candidate compound families for neutral molecular formulae.

    Compounds are linked only across different formulae. Connected components
    become families, and families smaller than ``min_size`` are discarded.

    :param formulas: Neutral query formulae; duplicates and non-Hill ordering
        are canonicalized before searching.
    :param source: Compound-cache source to query.
    :param cutoff: Minimum similarity required to link two compounds.
    :param min_size: Minimum number of compounds in a retained family.
    :param metric: Similarity metric, either ``TANIMOTO`` or ``MCES``.
    :param log: Callback invoked with progress and summary messages.
    :param db_path: Path to the compound cache.
    :return: Families sorted by score, then compound count, best first.
    :raises ValueError: If a formula does not parse.
    """
    log = log or (lambda _: None)
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    # Canonicalize the query and load all matching structures.
    query = {canonical_formula(f) for f in formulas}
    
    log(
        f"Query: {len(formulas)} formulae -> {len(query)} distinct: "
        f"{', '.join(sorted(query))}"
    )
    log(f"Reading {source} from {db_path} ...")

    RDLogger.DisableLog('rdApp.*')  # Databases have a few unparseable SMILES.
    try:
        compounds = []
        mols = []
        unparsed = Counter()
        for c in load_matches(source, query, log, db_path):
            mol = Chem.MolFromSmiles(c.smiles)
            if mol is None:
                unparsed[c.formula] += 1
            else:
                Chem.RemoveStereochemistry(mol)  # Flat, as matched.
                compounds.append(
                    replace(c, smiles=Chem.MolToSmiles(mol))
                )
                mols.append(mol)
    finally:
        RDLogger.EnableLog('rdApp.*')

    # Report hits before computing pairwise similarities.
    hits = Counter(c.formula for c in compounds)
    log("Database hits per formula:")
    for formula in sorted(query):
        n_unparsed = unparsed[formula]
        note = (
            f" ({n_unparsed} unparseable SMILES skipped)"
            if n_unparsed
            else ""
        )
        drop = (
            "  <- DROPPED: no database compounds"
            if not hits[formula]
            else ""
        )
        log(f"    {formula:<14} {hits[formula]:>6}{note}{drop}")

    if not compounds:
        return []

    # Build an undirected similarity graph using cross-formula links only.
    group = np.array([c.formula for c in compounds])
    if metric == MCES:
        rows, cols = _mces_links(mols, group, cutoff, log)
    else:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        fps = [generator.GetSparseCountFingerprint(m) for m in mols]
        rows = []
        cols = []
        for i, fp in enumerate(fps):
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps))
            linked = np.flatnonzero(
                (sims >= cutoff) & (group != group[i])
            )
            rows.extend([i] * len(linked))
            cols.extend(linked)

    n = len(compounds)
    adjacency = coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n, n),
    )
    n_components, labels = connected_components(adjacency, directed=False)
    log(
        f"{n} compounds, {len(rows) // 2} links at "
        f"{METRICS[metric]} >= {cutoff} (between different formulae) -> "
        f"{n_components} connected families"
    )

    # Convert connected components to families, then filter and rank them.
    families = []
    dropped = []
    for label in range(n_components):
        members = [compounds[i] for i in np.flatnonzero(labels == label)]
        family = Family(members, frozenset(c.formula for c in members))
        if len(members) >= min_size:
            families.append(family)
        else:
            dropped.append(family)

    families.sort(
        key=lambda f: (f.score, len(f.compounds)),
        reverse=True,
    )
    _log_dropped(log, dropped, min_size)

    explained = set().union(*(f.formulas for f in families))
    log(f"Kept {len(families)} families of >= {min_size} compounds")
    if unexplained := sorted(set(hits) - explained):
        log(f"Formulae in no kept family: {', '.join(unexplained)}")

    return families


def _mces_links(
    mols,
    group: np.ndarray,
    cutoff: float,
    log: Callable[[str], None],
) -> tuple[list[int], list[int]]:
    """Return both directions of each cross-formula link with J >= cutoff.

    Only exactly solved pairs can become links; timed-out pairs are skipped.
    """
    import time

    from core.utils.mces import Mol, mces_j

    graphs = [Mol(m) for m in mols]
    n = len(graphs)
    n_pairs = sum(
        int((group[i + 1:] != group[i]).sum())
        for i in range(n)
    )
    log(
        f"Normalized MCES on {n_pairs} cross-formula pairs "
        f"(J >= {cutoff}, {MCES_TIME_LIMIT:g} s limit per pair) ..."
    )

    rows: list[int] = []
    cols: list[int] = []
    status = Counter()
    t0 = last = time.perf_counter()
    done = 0

    for i in range(n):
        for j in range(i + 1, n):
            if group[i] == group[j]:
                continue

            _, J, _, st, _ = mces_j(
                graphs[i], graphs[j], cutoff, MCES_TIME_LIMIT
            )
            status[st] += 1
            done += 1
            if st == 'exact' and J >= cutoff:
                rows += [i, j]
                cols += [j, i]

        if time.perf_counter() - last > 10:
            last = time.perf_counter()
            log(
                f"    {done}/{n_pairs} pairs, {len(rows) // 2} links, "
                f"{last - t0:.0f} s elapsed"
            )

    log(
        f"    done in {time.perf_counter() - t0:.0f} s: "
        f"{status['exact']} solved (J >= cutoff), "
        f"{status['below']} proven below, "
        f"{status['timeout']} timed out (not linked)"
    )
    return rows, cols


def _log_dropped(
    log: Callable[[str], None],
    dropped: list[Family],
    min_size: int,
) -> None:
    """Log unlinked compounds by formula and small families in full."""
    singletons = Counter(
        f.compounds[0].formula
        for f in dropped
        if len(f.compounds) == 1
    )
    small = [f for f in dropped if len(f.compounds) > 1]

    log(f"Dropped {len(dropped)} families of < {min_size} compounds:")
    log(
        f"    {sum(singletons.values())} unlinked compounds "
        f"(similar to nothing from another formula): "
        + ", ".join(
            f"{formula} x{n}"
            for formula, n in sorted(singletons.items())
        )
    )

    for family in sorted(small, key=lambda f: (-f.score, -len(f.compounds))):
        log(
            f"    {len(family.compounds)} compounds, "
            f"{family.score} formulae [{family.top_class() or '?'}]: "
            + "; ".join(
                f"{c.formula} {c.name or c.identifier}"
                for c in family.compounds
            )
        )


def format_families(
    families: list[Family],
    formulas: list[str],
    max_families: int = 10,
    max_compounds: int = 25,
) -> str:
    """Format a plain-text report, marking tied best families with ``*``.

    :param families: Ranked families returned by :func:`snap_mfs`.
    :param formulas: Original query formulae, used for the total query count.
    :param max_families: Maximum number of families to include.
    :param max_compounds: Maximum number of compounds to show per family.
    :return: The formatted report.
    """
    n_query = len({canonical_formula(f) for f in formulas})
    if not families:
        return "SNAP-MFs: no family kept (all below the minimum size)."

    best = families[0].score
    lines = [
        (
            f"SNAP-MFs: {len(families)} families for {n_query} formulae "
            f"(showing {min(max_families, len(families))})"
        )
    ]

    for rank, family in enumerate(families[:max_families], 1):
        mark = '*' if family.score == best else ' '
        lines.append(
            f"\n{mark}#{rank}  explains {family.score}/{n_query} formulae, "
            f"{len(family.compounds)} compounds  "
            f"[{family.top_class() or '?'}]"
            f"\n    formulae: {', '.join(sorted(family.formulas))}"
        )

        sorted_compounds = sorted(
            family.compounds,
            key=lambda c: c.formula,
        )
        for c in sorted_compounds[:max_compounds]:
            lines.append(
                f"    {c.formula:<14} {c.identifier:<16} "
                f"{c.name or '(unnamed)'}  {c.smiles}"
            )

        if len(family.compounds) > max_compounds:
            lines.append(
                f"    ... {len(family.compounds) - max_compounds} more"
            )

    return '\n'.join(lines)
