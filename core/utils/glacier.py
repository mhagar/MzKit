"""
GLACIER MS/MS spectrum prediction (Coley group ms-pred) without torch or dgl.

The three neural blocks (encoder, breakpoint decoder, intensity head) run as
ONNX graphs in onnxruntime; featurization and the fragment bookkeeping between
them are ported here in NumPy + RDKit. The graphs and `meta.json` (ms-pred
constants: element order, adduct masses, RDKit enum codes) live in
`<project root>/models/glacier/` and are written by
`tools/glacier/export_onnx.py`, which also checks this module against torch.

Mirrors `ms_pred.glacier.joint_model.JointModel.predict_mol`: one batch per
call, same padding, same outputs.
"""
from functools import lru_cache
import json
import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from core.utils.config import get_project_root

MODEL_DIR = get_project_root() / "models" / "glacier"
BLOCKS = ("encoder", "breakpoints", "intensity")
WEIGHTS_URL = ("https://www.dropbox.com/scl/fo/ta99j0mp1w7qzek5zvy3w/"
               "AFCLQNO8W2EP7ZDMOj9nxWI?rlkey=563zxtyvvhwfu1uyz6n10n2ui")  # MassSpecGym-trained


def model_available() -> bool:
    return all((MODEL_DIR / f"{b}.onnx").exists() for b in BLOCKS) and (MODEL_DIR / "meta.json").exists()


def missing_model_message() -> str:
    return (
        f"GLACIER model not found in {MODEL_DIR}.\n"
        f"Download best.ckpt from {WEIGHTS_URL} and convert it with "
        f"`uv run --script tools/glacier/export_onnx.py --ckpt best.ckpt --ms-pred <ms-pred checkout>`."
    )


@lru_cache(maxsize=1)
def load_model() -> 'GlacierONNX':
    if not model_available():
        raise FileNotFoundError(missing_model_message())
    return GlacierONNX(MODEL_DIR)


class GlacierONNX:
    def __init__(self, model_dir):
        import onnxruntime as ort  # deferred: only GLACIER needs it

        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.sessions = {b: ort.InferenceSession(str(model_dir / f"{b}.onnx"), providers=["CPUExecutionProvider"])
                         for b in BLOCKS}
        hp = self.meta["hparams"]
        self.max_broken_bonds = hp["max_broken_bonds"]
        self.n_shifts = 2 * self.max_broken_bonds + 1
        self.rows_per_frag = 2 * self.n_shifts  # (adduct, electron) mass terms x H shifts
        self.mass = self.meta["element_to_mass"]
        self.elements = self.meta["valid_elements"]

    def _run(self, block, **feeds):
        sess = self.sessions[block]
        return sess.run(None, {i.name: feeds[i.name] for i in sess.get_inputs()})

    def predict(self, smiles, collision_energies, adducts, instruments) -> list[dict]:
        """
        Predict spectra for a batch. Collision energy may be NaN (unknown).
        Returns one dict per molecule with
          "spec": (rows, 2) [m/z, intensity], one row per fragment x adduct-electron term x H shift
                  (intensity 0 for impossible H shifts), unmerged and unnormalized, like ms-pred;
          "frag": (rows, n_atoms) bool, the heavy atoms of each row's fragment.
        """
        meta = self.meta
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        for s, m in zip(smiles, mols):
            if m is None:
                raise ValueError(f"Invalid SMILES: {s}")
        graphs = [featurize(m, meta) for m in mols]
        n_atoms = np.array([g["x"].shape[0] for g in graphs])
        B, N = len(mols), int(n_atoms.max())

        x = np.zeros((B, N, graphs[0]["x"].shape[1]), np.float32)
        attn_bias = np.full((B, N + 1, N + 1), -99999, np.float32)
        spatial_pos = np.zeros((B, N, N), np.int64)
        degree = np.zeros((B, N), np.int64)
        edge_input = np.zeros((B, N, N, *graphs[0]["edge_input"].shape[2:]), np.float32)
        adj = np.zeros((B, N, N), np.float32)
        masses = np.zeros((B, N))
        atom_hs = np.zeros((B, N))
        atom_form = np.zeros((B, N, len(self.elements)), np.float32)
        for i, (g, m) in enumerate(zip(graphs, mols)):
            n = n_atoms[i]
            x[i, :n], attn_bias[i, :n + 1, :n + 1] = g["x"], g["attn_bias"]
            spatial_pos[i, :n, :n], degree[i, :n], edge_input[i, :n, :n] = g["spatial_pos"], g["degree"], g["edge_input"]
            adj[i, :n, :n] = Chem.GetAdjacencyMatrix(m, useBO=True)
            # FragmentEngine bookkeeping (Kekulize doesn't change H counts, so the input mol is used as-is)
            for a in m.GetAtoms():
                j, sym, hs = a.GetIdx(), a.GetSymbol(), a.GetTotalNumHs()
                if a.GetIsotope():
                    raise ValueError(f"Isotope-labelled atoms are not supported: {Chem.MolToSmiles(m)}")
                atom_hs[i, j] = hs
                masses[i, j] = self.mass[sym] + hs * self.mass["H"]
                atom_form[i, j] = formula_to_dense(f"{sym}H{hs}", self.elements)
        root_form = np.stack([formula_to_dense(neutral_formula(m), self.elements) for m in mols]).astype(np.float32)

        root_tokens, node_emb = self._run(
            "encoder", x=x, attn_bias=attn_bias, spatial_pos=spatial_pos, degree=degree, edge_input=edge_input,
            adducts=np.array([meta["ion2onehot_pos"][a] for a in adducts], np.float32),
            collision_engs=np.asarray(collision_energies, np.float32),
            instruments=np.array([meta["instrument2onehot_pos"][s] for s in instruments], np.float32),
        )
        node_mask = np.arange(N)[None, :] >= n_atoms[:, None]
        scores, k = self._run("breakpoints", root_tokens=root_tokens, node_embeddings=node_emb,
                              node_mask=node_mask, root_form_vecs=root_form)
        # top-k atoms per slot; ties go to the lower atom index (ONNX TopK convention)
        rank = np.argsort(np.argsort(-scores, axis=-1, kind="stable"), axis=-1, kind="stable")
        frags, counts = breakpoints_to_fragments(rank < k[..., None], adj, n_atoms)

        # Per-fragment inputs to the intensity head
        F = counts.max()
        fmask = np.zeros((B, F, N), bool)   # fragment atoms
        pad = np.ones((B, F), bool)
        for i in range(B):
            fmask[i, :counts[i]] = frags[i]
            pad[i, :counts[i]] = False
        frag_form = np.einsum("bfn,bne->bfe", fmask.astype(np.float32), atom_form)
        num_broken = np.einsum("bfn,bnm,bfm->bf", fmask.astype(np.float32), adj, (~fmask).astype(np.float32))
        memory_mask = ~(fmask | pad[..., None])  # ms-pred pads fragment masks with True
        (inten,) = self._run("intensity", root_tokens=root_tokens, node_embeddings=node_emb,
                             frag_form_vecs=frag_form, root_form_vecs=root_form,
                             num_broken=num_broken, frag_mask=pad, memory_mask=memory_mask)

        # m/z of every fragment +/- H shifts, for the adduct mass and the electron term
        frag_hs = fmask @ atom_hs[:, :, None]
        max_remove = np.minimum(frag_hs[..., 0], self.max_broken_bonds)
        max_add = np.minimum(atom_hs.sum(1)[:, None] - frag_hs[..., 0], self.max_broken_bonds)
        h_shift = np.arange(-self.max_broken_bonds, self.max_broken_bonds + 1) * self.mass["H"]
        adduct_shift = np.array([[meta["ion2mass"][a], -meta["electron_mass"] if a.endswith("+") else meta["electron_mass"]]
                                 for a in adducts])
        net = (fmask * masses[:, None, :]).sum(-1)
        mz = net[:, :, None, None] + h_shift[None, None, None, :] + adduct_shift[:, None, :, None]
        mz = np.where(mz > 0, mz, 0)
        centre = self.max_broken_bonds
        h = np.arange(self.n_shifts)
        valid = (h <= max_add[..., None] + centre) & (h >= centre - max_remove[..., None]) & ~pad[..., None]
        inten = np.where(np.repeat(valid[:, :, None, :], 2, axis=2).reshape(B, F, -1), inten, 0)

        out = []
        for i in range(B):
            rows = counts[i] * self.rows_per_frag
            out.append({
                "spec": np.stack([mz[i].reshape(-1)[:rows], inten[i].reshape(-1)[:rows]], -1),
                "frag": np.repeat(frags[i][:, :n_atoms[i]], self.rows_per_frag, axis=0),
            })
        return out


def max_diff_up_to_symmetry(mol, pred: dict, ref: dict, rows_per_frag: int) -> float:
    """
    Largest |spec difference| between two predict() outputs for `mol`, pairing fragments
    by their atoms' symmetry classes (inf if the fragment sets differ). Symmetric atoms
    (e.g. an NMe2's methyls) get breakpoint scores equal up to float noise, so torch and
    onnxruntime can legitimately break such ties differently. For parity checks.
    """
    if pred["spec"].shape != ref["spec"].shape:
        return float("inf")
    ranks = np.array(Chem.CanonicalRankAtoms(mol, breakTies=False))
    blocks = []
    for out in (pred, ref):
        frags = out["frag"][::rows_per_frag]
        keys = [tuple(sorted(ranks[f])) for f in frags]
        order = sorted(range(len(keys)), key=keys.__getitem__)
        blocks.append(([keys[i] for i in order], out["spec"].reshape(len(keys), rows_per_frag, 2)[order]))
    (keys_a, spec_a), (keys_b, spec_b) = blocks
    return float(np.abs(spec_a - spec_b).max()) if keys_a == keys_b else float("inf")


def breakpoints_to_fragments(breaks, adj, n_atoms):
    """
    Connected components left after deleting each slot's breakpoint atoms.
    breaks: (B, S, N) bool. Returns per-molecule (n_frags, N) bool arrays (unique,
    lexicographically sorted like torch.unique) and their counts.
    """
    B, S, N = breaks.shape
    keep = ~breaks
    valid = np.arange(N)[None, :] < n_atoms[:, None]
    reach = (adj[:, None] > 0) & keep[..., :, None] & keep[..., None, :]
    for k in range(N):  # transitive closure; an atom only reaches itself via a neighbour
        reach |= reach[..., :, k:k + 1] & reach[..., k:k + 1, :]
    reach &= (valid[:, :, None] & valid[:, None, :])[:, None]
    frags = []
    for i in range(B):
        rows = np.unique(reach[i].reshape(-1, N).astype(np.int64), axis=0)
        rows = rows[rows.any(1)].astype(bool)
        if len(rows) == 0:  # nothing left: the whole molecule is the one fragment
            rows = valid[i][None]
        frags.append(rows)
    return frags, np.array([len(f) for f in frags])


def neutral_formula(mol) -> str:
    return re.match(r"[^+\-]*", CalcMolFormula(mol)).group(0)


def formula_to_dense(formula: str, elements) -> np.ndarray:
    vec = np.zeros(len(elements))
    for sym, num in re.findall(r"([A-Z][a-z]*)([0-9]*)", formula):
        if sym not in elements:
            raise ValueError(f"GLACIER does not support element {sym}")
        vec[elements.index(sym)] += int(num) if num else 1
    return vec


def random_walk_pe(adj_bo: np.ndarray, k: int) -> np.ndarray:
    """Landing probabilities of 1..k-step bond-order-weighted random walks (ms-pred random_walk_pe)."""
    rw = (adj_bo / (adj_bo.sum(1, keepdims=True) + 1e-30)).astype(np.float32)
    out, pe = rw, []
    for _ in range(k):
        pe.append(np.diag(out).copy())
        out = out @ rw
    return np.stack(pe, -1)


def _one_hot(value, choices, other=True):
    feats = [float(value == c) for c in choices]
    return feats + [float(value not in choices)] if other else feats


def featurize(mol, meta) -> dict:
    """ms_pred.glacier.dataset.TreeProcessor.create_graphormer_input (heavy atoms only)."""
    hp, c = meta["hparams"], meta["featurizer"]
    mol = Chem.RemoveHs(mol)
    n = mol.GetNumAtoms()
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        pass
    sssr = Chem.GetSymmSSSR(mol)
    atom_rings = np.zeros(n, int)
    bond_rings = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): 0 for b in mol.GetBonds()}
    for ring in sssr:
        ring = list(ring)
        atom_rings[ring] += 1
        for a1, a2 in zip(ring, ring[1:] + ring[:1]):
            key = tuple(sorted((a1, a2)))
            if key in bond_rings:
                bond_rings[key] += 1

    ring_sizes, max_deg, max_chg, max_h = c["ring_sizes"], c["max_degree"], c["max_abs_charge"], c["max_hydrogens"]
    x, degrees = [], []
    for atom in mol.GetAtoms():
        nh = atom.GetTotalNumHs()
        deg = atom.GetTotalDegree() - nh
        chg = atom.GetFormalCharge()
        f = _one_hot(atom.GetAtomicNum(), c["atom_nums"])
        f += [float(deg == d) for d in range(max_deg + 1)] + [float(deg > max_deg)]
        f += [float(chg == q) for q in range(-max_chg, max_chg + 1)] + [float(abs(chg) > max_chg)]
        f += _one_hot(int(atom.GetHybridization()), c["hybridization"])
        f.append(float(atom.GetIsAromatic()))
        f += [float(nh == h) for h in range(max_h + 1)] + [float(nh > max_h)]
        f += [float(not atom.IsInRing()), float(atom.IsInRing())]
        for r in ring_sizes:
            f += [float(atom.IsInRingSize(r)), float(not atom.IsInRingSize(r))]
        fused = atom_rings[atom.GetIdx()] >= 2
        f += [float(not fused), float(fused)]
        f += _one_hot(int(atom.GetChiralTag()), c["chirality"], other=False)
        if hp["embed_elem_group"]:
            f += c["element_to_group"].get(atom.GetSymbol(), [0.0] * c["element_group_dim"])
        try:
            q = float(atom.GetProp("_GasteigerCharge"))
            q = 0.0 if not np.isfinite(q) else q
        except Exception:
            q = 0.0
        f.append(q)
        x.append(f)
        degrees.append(deg)
    x = np.array(x, np.float32)
    if hp["pe_embed_k"] > 0:
        x = np.concatenate([x, random_walk_pe(Chem.GetAdjacencyMatrix(mol, useBO=True), hp["pe_embed_k"])], -1)

    # BFS shortest paths, keeping the first-found parent (argmax over candidates) like ms-pred
    adj = Chem.GetAdjacencyMatrix(mol).astype(bool)
    dist = np.full((n, n), 32767, np.int64)
    parents = np.full((n, n), -1, np.int64)
    for src in range(n):
        visited = np.zeros(n, bool)
        frontier = np.zeros(n, bool)
        frontier[src] = visited[src] = True
        dist[src, src], parents[src, src], depth = 0, src, 0
        while True:
            nxt = adj[frontier].any(0) & ~visited
            if not nxt.any():
                break
            depth += 1
            dist[src, nxt] = depth
            parents[src, nxt] = (adj[:, nxt] & frontier[:, None]).argmax(0)
            visited |= nxt
            frontier = nxt
    spatial_max = c["spatial_pos_max"]
    dist = np.clip(dist, 0, spatial_max - 1)

    bond_feats = np.zeros((n, n, c["edge_feats"]), np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = _one_hot(int(bond.GetBondType()), c["bond_types"])
        f += [float(not bond.GetIsConjugated()), float(bond.GetIsConjugated())]
        f += [float(not bond.IsInRing()), float(bond.IsInRing())]
        for r in ring_sizes:
            f += [float(bond.IsInRingSize(r)), float(not bond.IsInRingSize(r))]
        fused = bond_rings.get(tuple(sorted((i, j))), 0) >= 2
        f += [float(not fused), float(fused)]
        f += _one_hot(int(bond.GetStereo()), c["bond_stereo"])
        bond_feats[i, j] = bond_feats[j, i] = f

    max_hop = hp["multi_hop_max_dist"]
    edge_input = np.zeros((n, n, max_hop, c["edge_feats"]), np.float32)
    for src in range(n):
        for tgt in range(n):
            if tgt == src or parents[src, tgt] == -1:
                continue
            path = [tgt]
            while path[-1] != src:
                path.append(int(parents[src, path[-1]]))
            path.reverse()
            for hop in range(min(len(path) - 1, max_hop)):
                edge_input[src, tgt, hop] = bond_feats[path[hop], path[hop + 1]]

    attn_bias = np.zeros((n + 1, n + 1), np.float32)
    attn_bias[1:, 1:] = np.where(dist >= spatial_max, -99999, 0.0)
    return {"x": x, "attn_bias": attn_bias, "spatial_pos": dist, "degree": np.array(degrees),
            "edge_input": edge_input}
