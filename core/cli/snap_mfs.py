"""
SNAP-MFs: guess the compound family behind a set of molecular formulae the
user asserts are related. A port of snapmf's formula-list mode
(~/Dropbox/snapmf, `network_from_formula_list`), minus Cytoscape/GraphML.

1. Every query formula pulls every database compound with that formula;
   the formula is the compound's "group".
2. Two compounds are joined when their similarity is >= `cutoff` and they
   came from different formulae. `metric`: TANIMOTO, Morgan (radius 2,
   counts) Tanimoto (snapmf uses Dice >= 0.66, i.e. Tanimoto ~0.49:
   T = D / (2 - D)); or MCES, normalized MCES J = e_c^2 / (e_a e_b)
   (`core.utils.mces`; slow - one ILP per candidate pair).
3. Connected families smaller than `min_size` compounds are dropped.
4. Families are ranked by how many query formulae they explain; those
   tied for the most are the top candidates.

Database: a COCONUT CSV export (columns identifier, canonical_smiles, name,
molecular_formula, np_classifier_class, standard_inchi_key), or an NPAtlas
.tsv download (mapped onto those; the "class" is the producing organism,
e.g. "Bacterium Streptomyces").
"""
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from core.formula.manual import canonical_formula

TANIMOTO_CUTOFF = 0.5   # ~ snapmf's Dice 0.66
MIN_FAMILY_SIZE = 3     # snapmf's min_atlas_size
MCES_TIME_LIMIT = 10.0  # s per pair; undecided pairs aren't linked

TANIMOTO = 'tanimoto'
MCES = 'mces'
METRICS = {TANIMOTO: 'Tanimoto (ECFP4)', MCES: 'Normalized MCES (J)'}

_COLUMNS = [
    'identifier', 'canonical_smiles', 'name', 'molecular_formula',
    'np_classifier_class', 'standard_inchi_key',
]
_NPATLAS_COLUMNS = {
    'npaid': 'identifier', 'compound_smiles': 'canonical_smiles',
    'compound_name': 'name', 'compound_molecular_formula': 'molecular_formula',
    'compound_inchikey': 'standard_inchi_key',
}


@dataclass(frozen=True)
class DbCompound:
    identifier: str
    name: str
    smiles: str
    formula: str
    np_class: str   # NP Classifier class (COCONUT) or organism (NPAtlas)


@dataclass(frozen=True)
class Family:
    compounds: list[DbCompound]
    formulas: frozenset[str]    # the query formulae it explains

    @property
    def score(self) -> int:
        return len(self.formulas)

    def top_class(self) -> str:
        """Most common class (or organism, NPAtlas) among members ('' if none)."""
        classes = Counter(c.np_class for c in self.compounds if c.np_class)
        return classes.most_common(1)[0][0] if classes else ''


def load_matches(
    db_path: Path,
    formulas: set[str],
    log: Callable[[str], None] = lambda _: None,
) -> list[DbCompound]:
    """
    Database compounds whose formula is in `formulas` (Hill order), one per
    InChIKey-14 (connectivity): stereoisomers count as the same molecule.
    The first listed is kept. A .tsv is read as an NPAtlas download.
    """
    import pandas as pd
    if Path(db_path).suffix.lower() == '.tsv':
        df = pd.read_csv(
            db_path, sep='\t', dtype=str, keep_default_na=False,
            usecols=[*_NPATLAS_COLUMNS, 'origin_type', 'genus'],
        ).rename(columns=_NPATLAS_COLUMNS)
        df['np_classifier_class'] = (df['origin_type'] + ' ' + df['genus']).str.strip()
    else:
        df = pd.read_csv(
            db_path, usecols=_COLUMNS, dtype=str, keep_default_na=False,
        )
    df = df[df['molecular_formula'].isin(formulas)]
    # No key -> unique by identifier, never merged
    key14 = df['standard_inchi_key'].str[:14].where(
        df['standard_inchi_key'] != '', df['identifier'],
    )
    n_before = len(df)
    df = df[~key14.duplicated()]
    log(f"Flattened stereoisomers: {n_before} entries -> {len(df)} "
        f"distinct InChIKey-14")
    return [
        DbCompound(r.identifier, r.name, r.canonical_smiles,
                   r.molecular_formula, r.np_classifier_class)
        for r in df.itertuples(index=False)
    ]


def snap_mfs(
    formulas: list[str],
    db_path: Path,
    cutoff: float = TANIMOTO_CUTOFF,
    min_size: int = MIN_FAMILY_SIZE,
    metric: str = TANIMOTO,
    log: Optional[Callable[[str], None]] = None,
) -> list[Family]:
    """
    Candidate families for `formulas` (neutral), best first.
    :param log: called with a line per step (hits, links, dropped families).
    :raises ValueError: if a formula doesn't parse.
    """
    log = log or (lambda _: None)
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    query = {canonical_formula(f) for f in formulas}
    log(f"Query: {len(formulas)} formulae -> {len(query)} distinct: "
        f"{', '.join(sorted(query))}")
    log(f"Reading {db_path} ...")
    RDLogger.DisableLog('rdApp.*')  # databases have a few unparseable SMILES
    try:
        compounds, mols, unparsed = [], [], Counter()
        for c in load_matches(Path(db_path), query, log):
            mol = Chem.MolFromSmiles(c.smiles)
            if mol is None:
                unparsed[c.formula] += 1
            else:
                Chem.RemoveStereochemistry(mol)  # flat, as matched
                compounds.append(replace(c, smiles=Chem.MolToSmiles(mol)))
                mols.append(mol)
    finally:
        RDLogger.EnableLog('rdApp.*')

    hits = Counter(c.formula for c in compounds)
    log("Database hits per formula:")
    for formula in sorted(query):
        note = f" ({unparsed[formula]} unparseable SMILES skipped)" if unparsed[formula] else ""
        drop = "  <- DROPPED: no database compounds" if not hits[formula] else ""
        log(f"    {formula:<14} {hits[formula]:>6}{note}{drop}")
    if not compounds:
        return []

    group = np.array([c.formula for c in compounds])
    if metric == MCES:
        rows, cols = _mces_links(mols, group, cutoff, log)
    else:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        fps = [generator.GetSparseCountFingerprint(m) for m in mols]
        rows, cols = [], []
        for i, fp in enumerate(fps):
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps))
            j = np.flatnonzero((sims >= cutoff) & (group != group[i]))
            rows.extend([i] * len(j))
            cols.extend(j)
    n = len(compounds)
    adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_components, labels = connected_components(adjacency, directed=False)
    log(f"{n} compounds, {len(rows) // 2} links at {METRICS[metric]} >= {cutoff} "
        f"(between different formulae) -> {n_components} connected families")

    families, dropped = [], []
    for label in range(n_components):
        members = [compounds[i] for i in np.flatnonzero(labels == label)]
        family = Family(members, frozenset(c.formula for c in members))
        (families if len(members) >= min_size else dropped).append(family)
    families.sort(key=lambda f: (f.score, len(f.compounds)), reverse=True)
    _log_dropped(log, dropped, min_size)

    explained = set().union(*(f.formulas for f in families))
    log(f"Kept {len(families)} families of >= {min_size} compounds")
    if unexplained := sorted(set(hits) - explained):
        log(f"Formulae in no kept family: {', '.join(unexplained)}")
    return families


def _mces_links(mols, group: np.ndarray, cutoff: float, log) -> tuple[list, list]:
    """Both directions of every cross-formula pair with J >= cutoff."""
    import time
    from core.utils.mces import Mol, mces_j

    graphs = [Mol(m) for m in mols]
    n = len(graphs)
    n_pairs = sum(int((group[i + 1:] != group[i]).sum()) for i in range(n))
    log(f"Normalized MCES on {n_pairs} cross-formula pairs "
        f"(J >= {cutoff}, {MCES_TIME_LIMIT:g} s limit per pair) ...")
    rows, cols, status = [], [], Counter()
    t0 = last = time.perf_counter()
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            if group[i] == group[j]:
                continue
            _, J, _, st, _ = mces_j(graphs[i], graphs[j], cutoff, MCES_TIME_LIMIT)
            status[st] += 1
            done += 1
            if st == 'exact' and J >= cutoff:
                rows += [i, j]
                cols += [j, i]
        if time.perf_counter() - last > 10:
            last = time.perf_counter()
            log(f"    {done}/{n_pairs} pairs, {len(rows) // 2} links, "
                f"{last - t0:.0f} s elapsed")
    log(f"    done in {time.perf_counter() - t0:.0f} s: {status['exact']} solved "
        f"(J >= cutoff), {status['below']} proven below, "
        f"{status['timeout']} timed out (not linked)")
    return rows, cols


def _log_dropped(log: Callable[[str], None], dropped: list[Family], min_size: int):
    """Unlinked compounds as counts per formula; small families in full."""
    singletons = Counter(f.compounds[0].formula for f in dropped if len(f.compounds) == 1)
    small = [f for f in dropped if len(f.compounds) > 1]
    log(f"Dropped {len(dropped)} families of < {min_size} compounds:")
    log(f"    {sum(singletons.values())} unlinked compounds (similar to nothing "
        f"from another formula): "
        + ", ".join(f"{formula} x{n}" for formula, n in sorted(singletons.items())))
    for family in sorted(small, key=lambda f: (-f.score, -len(f.compounds))):
        log(f"    {len(family.compounds)} compounds, "
            f"{family.score} formulae [{family.top_class() or '?'}]: "
            + "; ".join(f"{c.formula} {c.name or c.identifier}" for c in family.compounds))


def format_families(
    families: list[Family],
    formulas: list[str],
    max_families: int = 10,
    max_compounds: int = 25,
) -> str:
    """Plain-text report: top candidates (tied best score) marked '*'."""
    n_query = len({canonical_formula(f) for f in formulas})
    if not families:
        return "SNAP-MFs: no family kept (all below the minimum size)."
    best = families[0].score
    lines = [
        f"SNAP-MFs: {len(families)} families for {n_query} formulae "
        f"(showing {min(max_families, len(families))})"
    ]
    for rank, family in enumerate(families[:max_families], 1):
        mark = '*' if family.score == best else ' '
        lines.append(
            f"\n{mark}#{rank}  explains {family.score}/{n_query} formulae, "
            f"{len(family.compounds)} compounds  [{family.top_class() or '?'}]"
            f"\n    formulae: {', '.join(sorted(family.formulas))}"
        )
        for c in sorted(family.compounds, key=lambda c: c.formula)[:max_compounds]:
            lines.append(
                f"    {c.formula:<14} {c.identifier:<16} "
                f"{c.name or '(unnamed)'}  {c.smiles}"
            )
        if len(family.compounds) > max_compounds:
            lines.append(f"    ... {len(family.compounds) - max_compounds} more")
    return '\n'.join(lines)
