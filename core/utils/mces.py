"""
Myopic MCES and bond-Johnson J between two molecules (for SNAP-MFs' "normalized MCES" metric).

    d   myopic MCES distance: the bond weight left unmatched by the best common edge subgraph
        (bond weights 1 / 1.5 / 2 / 3, so d lies on the 0.5 grid)
    J   = e_c^2 / (e_1 e_2),  e_c = (e_1 + e_2 - d) / 2      (e = heavy-atom bond counts)
        1 for the same graph, 0 for nothing in common; SIMBA's normalised MCES

"Myopic": only distances up to a threshold are solved exactly. The threshold is set per pair from a
J floor (mces_threshold_for), so the answer is censored:

    exact    J >= j_min, and J is exact
    below    J <  j_min, proven; J_max is the best ceiling we have on J (<= j_min)
    timeout  undecided within the time limit; J_max is a ceiling from the lower bound (>= j_min)

The model and semantics are myopic_mces 1.2.2's (Kretschmer et al., AlBi-HHU/myopic-mces), without
networkx or PuLP: the lower bounds b0, b1, b2 are tried cheapest first, then 1.2.2's ILP is built with
numpy and solved by HiGHS. Ported from the user's j_bank.py; the only change is solving through
scipy.optimize.milp (SciPy's bundled HiGHS) instead of highspy.
"""

import math
import time

import numpy as np
from rdkit import Chem
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

PRESOLVE_MAX_COLS = 1500            # presolve costs more than it saves on smaller ILPs


def bond_johnson(d: float, e1: int, e2: int) -> float:
    """J from a myopic MCES distance over e1 and e2 heavy-atom bonds."""
    ec = max((e1 + e2 - d) / 2.0, 0.0)
    return ec * ec / (e1 * e2) if e1 and e2 else float("nan")


def mces_threshold_for(j_min: float, e1: int, e2: int) -> float:
    """The distance above which J < j_min (MCES's threshold for a J floor)."""
    return e1 + e2 - 2.0 * math.sqrt(j_min * e1 * e2)


class Mol:
    """Heavy-atom graph: element per atom, bonds (u, v, weight), weighted degrees, fragment count."""

    __slots__ = ("smiles", "elem", "u", "v", "w", "n", "m", "wdeg", "frags")

    def __init__(self, mol: str | Chem.Mol):
        if not isinstance(mol, Chem.Mol):
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                raise ValueError("unparsable SMILES")
        self.smiles = Chem.MolToSmiles(mol)
        self.elem = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
        b = np.array([(x.GetBeginAtomIdx(), x.GetEndAtomIdx(), x.GetBondTypeAsDouble()) for x in mol.GetBonds()],
                     dtype=np.float64).reshape(-1, 3)
        self.u, self.v, self.w = b[:, 0].astype(np.int64), b[:, 1].astype(np.int64), b[:, 2].copy()
        self.n, self.m = len(self.elem), len(b)
        self.wdeg = np.bincount(self.u, self.w, self.n) + np.bincount(self.v, self.w, self.n)
        self.frags = len(Chem.GetMolFrags(mol))


def _as_mol(x) -> Mol:
    return x if isinstance(x, Mol) else Mol(x)


def bound0(m1: Mol, m2: Mol) -> float:
    """Atoms in excess per element are unmatched, and so are >= that many bonds (of weight >= 1), less one per
    fragment that could be left unmatched whole: all but one per molecule, so valid only when d < W1 + W2 (the
    sums of bond weights), which any threshold below W1 + W2 implies."""
    if min(m1.w.min(initial=1.0), m2.w.min(initial=1.0)) < 1:
        return 0.0
    e = np.union1d(m1.elem, m2.elem)
    c1 = np.searchsorted(e, m1.elem)
    c2 = np.searchsorted(e, m2.elem)
    diff = np.bincount(c1, minlength=len(e)) - np.bincount(c2, minlength=len(e))
    return float(max(diff[diff > 0].sum() - (m1.frags - 1), 0) + max(-diff[diff < 0].sum() - (m2.frags - 1), 0))


def bound1(m1: Mol, m2: Mol) -> float:
    """1.2.2's filter1: weighted degrees matched in sorted order within each element."""
    tot = 0.0
    for t in np.union1d(m1.elem, m2.elem):
        a = np.sort(m1.wdeg[m1.elem == t])[::-1]
        b = np.sort(m2.wdeg[m2.elem == t])[::-1]
        k = min(len(a), len(b))
        tot += np.abs(a[:k] - b[:k]).sum() + a[k:].sum() + b[k:].sum()
    return tot / 2


def _features(m1: Mol, m2: Mol) -> tuple[np.ndarray, np.ndarray]:
    """Per atom, the incident bond weights per neighbour element, sorted descending and zero-padded."""
    elems = {e: i for i, e in enumerate(np.union1d(m1.elem, m2.elem).tolist())}
    lists, width = [], 1
    for M in (m1, m2):
        L = [{} for _ in range(M.n)]
        el = M.elem.tolist()
        for a, b, w in zip(M.u.tolist(), M.v.tolist(), M.w.tolist()):
            L[a].setdefault(el[b], []).append(w)
            L[b].setdefault(el[a], []).append(w)
        width = max([width] + [len(ws) for d in L for ws in d.values()])
        lists.append(L)
    out = []
    for M, L in zip((m1, m2), lists):
        F = np.zeros((M.n, len(elems), width))
        for i, d in enumerate(L):
            for e, ws in d.items():
                F[i, elems[e], :len(ws)] = sorted(ws, reverse=True)
        out.append(F.reshape(M.n, -1))
    return out[0], out[1]


def bound2(m1: Mol, m2: Mol) -> float:
    """1.2.2's filter2: per element, a min-cost assignment of atoms with cost ½·L1 of their neighbourhood
    features (unmatched atoms cost half their weighted degree)."""
    F1, F2 = _features(m1, m2)
    res = 0.0
    for t in np.union1d(m1.elem, m2.elem):
        i1, i2 = np.flatnonzero(m1.elem == t), np.flatnonzero(m2.elem == t)
        n1, n2 = len(i1), len(i2)
        if not n1 or not n2:
            res += (m1.wdeg[i1].sum() + m2.wdeg[i2].sum()) / 2
            continue
        n = max(n1, n2)
        C = np.empty((n, n))
        C[:n1, :n2] = cdist(F1[i1], F2[i2], "cityblock") / 2
        if n1 < n2:
            C[n1:, :] = m2.wdeg[i2][None, :] / 2
        elif n2 < n1:
            C[:, n2:] = m1.wdeg[i1][:, None] / 2
        r, c = linear_sum_assignment(C)
        res += C[r, c].sum()
    return float(res)


def lower_bound(m1: Mol, m2: Mol, threshold: float) -> tuple[float, str]:
    """The first of b0, b1, b2 above threshold, else b2 -> (bound, name)."""
    use_b0 = threshold < m1.w.sum() + m2.w.sum()
    for name, f in (("b0", bound0), ("b1", bound1), ("b2", bound2))[0 if use_b0 else 1:]:
        b = f(m1, m2)
        if b > threshold:
            return b, name
    return b, "b2"


def build_ilp(m1: Mol, m2: Mol, threshold: float | None):
    """1.2.2's ILP -> (cost, rows, cols, vals, row_lo, row_hi, n_col, n_row); all columns binary.

    Columns: y (same-element atom pairs), c (compatible bond pairs, cost |w1 - w2|), then G1's and G2's unmatched-
    bond indicators (cost w). Rows of constraint (5) without a c term never bind and are dropped. A threshold
    adds the row cost . x <= threshold."""
    t1, t2 = m1.elem, m2.elem
    Yi, Yj = np.nonzero(t1[:, None] == t2[None, :])
    ny = len(Yi)
    ycol = np.full((m1.n, m2.n), -1, dtype=np.int64)
    ycol[Yi, Yj] = np.arange(ny)
    a1u, a1v, a2u, a2v = t1[m1.u], t1[m1.v], t2[m2.u], t2[m2.v]
    comp = ((a1u[:, None] == a2u[None, :]) & (a1v[:, None] == a2v[None, :])) | \
           ((a1v[:, None] == a2u[None, :]) & (a1u[:, None] == a2v[None, :]))
    E1, E2 = np.nonzero(comp)
    nc = len(E1)
    ccol = ny + np.arange(nc)
    c1col = ny + nc + np.arange(m1.m)
    c2col = ny + nc + m1.m + np.arange(m2.m)
    ncol = ny + nc + m1.m + m2.m
    cost = np.zeros(ncol)
    cost[ccol] = np.abs(m1.w[E1] - m2.w[E2])
    cost[c1col] = m1.w
    cost[c2col] = m2.w
    R, C, V, lo, hi = [], [], [], [], []
    r0 = 0
    for idx, nrow in ((Yi, m1.n), (Yj, m2.n)):                       # each atom mapped at most once
        R.append(r0 + idx); C.append(np.arange(ny)); V.append(np.ones(ny))
        lo.append(np.full(nrow, -np.inf)); hi.append(np.ones(nrow)); r0 += nrow
    for E, cXcol, mX in ((E1, c1col, m1.m), (E2, c2col, m2.m)):       # each bond matched once or unmatched
        R += [r0 + E, r0 + np.arange(mX)]; C += [ccol, cXcol]; V += [np.ones(nc), np.ones(mX)]
        lo.append(np.ones(mX)); hi.append(np.ones(mX)); r0 += mX
    for ends, other, bu, bv, yc in (((m1.u[E1], m1.v[E1]), E2, m2.u, m2.v, ycol),
                                    ((m2.u[E2], m2.v[E2]), E1, m1.u, m1.v, ycol.T)):
        nb = len(bu)                                                  # bond pairs at atom i need i mapped onto
        keys = np.concatenate([ends[0] * nb + other, ends[1] * nb + other])   # an end of the other bond
        uk, inv = np.unique(keys, return_inverse=True)
        R.append(r0 + inv); C.append(np.concatenate([ccol, ccol])); V.append(np.ones(2 * nc))
        node, eb = uk // nb, uk % nb
        for b in (bu, bv):
            yy = yc[node, b[eb]]
            ok = yy >= 0
            R.append(r0 + np.flatnonzero(ok)); C.append(yy[ok]); V.append(-np.ones(int(ok.sum())))
        lo.append(np.full(len(uk), -np.inf)); hi.append(np.zeros(len(uk))); r0 += len(uk)
    if threshold is not None:
        nz = np.flatnonzero(cost)
        R.append(np.full(len(nz), r0)); C.append(nz); V.append(cost[nz])
        lo.append(np.array([-np.inf])); hi.append(np.array([float(threshold)])); r0 += 1
    return (cost, np.concatenate(R), np.concatenate(C), np.concatenate(V), np.concatenate(lo), np.concatenate(hi),
            ncol, r0)


def solve_highs(problem, time_limit: float | None = None) -> tuple[str, float]:
    """Solve build_ilp's problem with HiGHS (via scipy.optimize.milp) -> (status, objective).

    status: "optimal"; "infeasible"; "feasible" (stopped at the time limit with an incumbent; the objective is
    an upper bound); "timeout" (no incumbent, objective nan); "error"."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_matrix

    cost, R, C, V, lo, hi, ncol, nrow = problem
    A = csc_matrix((V, (R, C)), shape=(nrow, ncol))
    A.sum_duplicates()
    options = {"disp": False, "presolve": ncol > PRESOLVE_MAX_COLS}
    if time_limit:
        options["time_limit"] = float(time_limit)
    res = milp(cost, integrality=np.ones(ncol), bounds=Bounds(0, 1),
               constraints=LinearConstraint(A, lo, hi), options=options)
    if res.status == 0:
        return "optimal", float(res.fun)
    if res.status == 2:
        return "infeasible", math.nan
    if res.status == 1:
        return ("feasible", float(res.fun)) if res.x is not None else ("timeout", math.nan)
    return "error", math.nan


def _grid(x: float) -> float:
    return round(2 * x) / 2


def mces_j(m1, m2, j_min: float, time_limit: float | None = 10.0) -> tuple[float, float, float, str, float]:
    """Censored J -> (d, J, J_max, status, seconds); see the module docstring for the statuses.

    Only "exact" carries d and J (J_max = J). Ceilings when censored: the size ratio min(e)/max(e)
    (e_c <= the smaller bond count), J at a proven distance lower bound, or j_min itself when the ILP
    is infeasible below the threshold."""
    t0 = time.perf_counter()
    m1, m2 = _as_mol(m1), _as_mol(m2)
    if m1.smiles == m2.smiles:
        return 0.0, 1.0, 1.0, "exact", 0.0
    e1, e2 = m1.m, m2.m
    nan = math.nan
    if not e1 or not e2:
        return nan, nan, 0.0, "below", time.perf_counter() - t0
    ratio = min(e1, e2) / max(e1, e2)
    if ratio < j_min:
        return nan, nan, ratio, "below", time.perf_counter() - t0
    thr = math.floor(2 * mces_threshold_for(j_min, e1, e2) + 1e-9) / 2
    b, _ = lower_bound(m1, m2, thr)
    ceiling = min(ratio, bond_johnson(b, e1, e2))
    if b > thr:
        return nan, nan, ceiling, "below", time.perf_counter() - t0
    problem = build_ilp(m1, m2, thr)
    st, obj = solve_highs(problem, time_limit)
    dt = time.perf_counter() - t0
    if st == "optimal":
        J = bond_johnson(_grid(obj), e1, e2)
        return _grid(obj), J, J, "exact", dt
    if st == "infeasible":
        return nan, nan, min(ceiling, j_min), "below", dt
    return nan, nan, ceiling, "timeout", dt
