# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#     "torch==2.6.0",
#     "numpy",
#     "rdkit>=2025.9.3,<2026",  # MzKit's pin, so the fixture matches the runtime's featurization
#     "pytorch-lightning",
#     "einops",
#     "linsatnet",
#     "onnx",
#     "onnxruntime",
#     "pandas",
#     "platformdirs",
#     "tqdm",
#     "scipy",
#     "pyyaml",
#     "packaging",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""
Dev tool (not shipped): convert a GLACIER checkpoint (ms-pred) into the ONNX
graphs + meta.json read by core/utils/glacier.py, check the port end to end
against ms-pred's own predict_mol, and write the pytest parity fixture.

    git clone https://github.com/coleygroup/ms-pred   # tested at commit noted in meta.json
    uv run --script tools/glacier/export_onnx.py --ckpt best.ckpt --ms-pred ../ms-pred

ms-pred's training-only imports (dgl, torch_scatter, pygmtools, plotting) are
stubbed out; at inference dgl is only used for the random-walk positional
encoding, replaced here by an equivalent torch function.
"""
import argparse
import importlib.abc
import importlib.machinery
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUBBED = ("dgl", "torch_scatter", "torch_sparse", "pygmtools", "h5py", "multiprocess",
           "matplotlib", "seaborn", "cairosvg")

# Fixture batch: varied size, adducts, instruments, and an unknown collision energy
SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "C[C@H]1CC[C@@]23CCC(=O)[C@H]2[C@@]1(C)[C@@H](C[C@](C)(C=C)[C@@H](O)[C@@H]3C)OC(=O)CO",  # pleuromutilin
    "COc1cc(C=CC(=O)O)ccc1O",  # ferulic acid
    "CC[C@@H]1[C@@]([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)O[C@H]2C[C@@]"
    "([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)O)C)C)O)(C)O",  # erythromycin
    "OC(=O)c1ccccc1O",  # salicylic acid
]
ADDUCTS = ["[M+H]+", "[M+Na]+", "[M+H]+", "[M-H]-", "[M+H]+", "[M-H]-"]
CES = [30.0, 40.0, float("nan"), 20.0, 35.0, 25.0]
INSTRUMENTS = ["Orbitrap", "QTOF", "Orbitrap", "Unknown", "QTOF", "IT-FT"]


class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in STUBBED:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        m = mock.MagicMock(name=spec.name)
        m.__path__, m.__spec__ = [], spec
        return m

    def exec_module(self, module):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="GLACIER Lightning checkpoint (best.ckpt)")
    parser.add_argument("--ms-pred", type=Path, required=True, help="ms-pred checkout")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "glacier")
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests" / "glacier_parity.npz")
    args = parser.parse_args()

    sys.meta_path.insert(0, _Stub())
    sys.path[:0] = [str(args.ms_pred / "src"), str(ROOT)]
    import torch
    import torch.nn as nn
    from rdkit import Chem
    import ms_pred.common as common
    from LinSATNet import linsat_layer
    from ms_pred.glacier.joint_model import JointModel
    from core.utils import glacier

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    hp = ck["hyper_parameters"]
    model = JointModel(**hp)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    H, NHEAD, TAU = model.hidden_size, model.nhead, model.linsat_tau

    def get_pe_for_tensor(mol, node_features):
        """TreeProcessor.get_pe_for_tensor + nn_utils.random_walk_pe without dgl/torch_scatter."""
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True)
        src, dst = np.nonzero(adj)
        row, col = torch.from_numpy(src), torch.from_numpy(dst)
        value = torch.tensor(adj[(src, dst)], dtype=torch.float32)
        n = adj.shape[0]
        value = value / (torch.zeros(n).index_add_(0, row, value)[row] + 1e-30)
        a = torch.zeros(n, n)
        a[row, col] = value
        out, pe = a, [torch.diagonal(a)]
        for _ in range(model.pe_embed_k - 1):
            out = out @ a
            pe.append(torch.diagonal(out))
        return torch.stack(pe, -1)
    model.tree_processor.get_pe_for_tensor = get_pe_for_tensor

    # Reference run, recording each block's real inputs/outputs
    cap = {}
    def record(name):
        fn = getattr(model, name)
        def wrapped(*a, **kw):
            out = fn(*a, **kw)
            cap[name] = (a, kw, out)
            return out
        setattr(model, name, wrapped)
        return fn
    molecular_embedding = record("molecular_embedding")
    record("breakpoint_forward")
    record("inten_calculation")
    ref = model.predict_mol(SMILES, CES, ADDUCTS, device="cpu", instrument=INSTRUMENTS)

    def embed(emb, t):
        """IntFeaturizer.forward as a plain lookup (the original's masked scatter doesn't export)."""
        table = torch.cat([emb.int_to_feat_matrix, emb._extra_embeddings], 0)
        return table[t.clamp(max=emb.MAX_COUNT_INT).long()].reshape(*t.shape[:-1], -1)

    def linsat_sum_k(logit, k):
        """linsat_layer(logit, E=ones, f=k, max_iter=1) in closed form: one constraint, one Sinkhorn step."""
        n = torch.ones_like(logit).sum(-1, keepdim=True)  # atom count as a tensor, so it stays dynamic in ONNX
        lx = logit / TAU
        log_sum = torch.logsumexp(torch.stack((lx, torch.zeros_like(lx)), 0), 0)
        l0, l1 = lx - log_sum, -log_sum
        b0, b1 = torch.full_like(n, k), n - k
        eps = torch.where(b1 == 0, 1e-7, 0.0)  # linsat's guard for a zero marginal (k == n)
        b0, b1 = b0 + eps, b1 + eps
        log_a = torch.log(1 / n)
        t0 = l0 - torch.logsumexp(l0 + log_a, -1, keepdim=True) + torch.log(b0 / (b0 + b1))
        t1 = l1 - torch.logsumexp(l1 + log_a, -1, keepdim=True) + torch.log(b1 / (b0 + b1))
        return torch.exp(t0 - torch.logsumexp(torch.stack((t0, t1), 0), 0))

    class Encoder(nn.Module):
        def forward(self, x, attn_bias, attn_edge_type, spatial_pos, degree, edge_input,
                    adducts, collision_engs, instruments):
            g = dict(x=x, attn_bias=attn_bias, attn_edge_type=attn_edge_type,
                     spatial_pos=spatial_pos, degree=degree, edge_input=edge_input)
            o = molecular_embedding(adducts, collision_engs, graphormer_input=g, instruments=instruments)
            return o["root_tokens"], o["node_embeddings"]

    class Breakpoints(nn.Module):
        """breakpoint_forward (last decoder layer) + node_ranking + the cardinality gather of breakpoint_inference."""
        def forward(self, root_tokens, node_embeddings, node_mask, root_form_vecs):
            m = model
            memory_mask = torch.cat([torch.zeros_like(node_mask[:, :1]), node_mask], 1)
            root = m.formula_mapper(torch.cat((root_tokens, embed(m.embedder, root_form_vecs)[:, None, :]), -1))
            slots = m.fragment_decoder.slots.unsqueeze(0).expand(root.shape[0], -1, -1)
            memory = torch.cat([root, node_embeddings], 1)
            for layer in m.fragment_decoder.decoder_layers:
                slots = layer(tgt=slots, memory=memory, memory_key_padding_mask=memory_mask)
            logit = torch.einsum("bij,bkj->bik", slots, node_embeddings).masked_fill(node_mask[:, None, :], -1e9)
            k = m.frag_card_mapper(slots).argmax(-1)
            per_k = torch.stack([torch.zeros_like(logit)] + [linsat_sum_k(logit, kk) for kk in (1, 2, 3)], -1)
            return torch.gather(per_k, -1, k[..., None, None].expand(*logit.shape, 1))[..., 0], k

    class Intensity(nn.Module):
        """inten_calculation's network; the mass / H bookkeeping is in core/utils/glacier.py."""
        def forward(self, root_tokens, node_embeddings, frag_form_vecs, root_form_vecs,
                    num_broken, frag_mask, memory_mask):
            m = model
            b, f = frag_form_vecs.shape[:2]
            broken = nn.functional.one_hot(num_broken.clamp(max=m.max_broken_bonds).long(), m.max_broken_bonds + 1)
            tok = m.token_mapper(torch.cat([root_tokens.expand(b, f, H), embed(m.embedder, frag_form_vecs),
                                            embed(m.embedder, root_form_vecs[:, None, :] - frag_form_vecs),
                                            broken.float()], -1))
            h = m.inten_decoder(tgt=tok, memory=node_embeddings, tgt_key_padding_mask=frag_mask,
                                memory_mask=torch.repeat_interleave(memory_mask, NHEAD, dim=0))
            h = m.inten_encoder(h, src_key_padding_mask=frag_mask)
            return torch.sigmoid(m.output_map(h))

    # Example inputs for tracing, from the reference run
    (e_args, e_kw, e_out) = cap["molecular_embedding"]
    g = e_kw["graphormer_input"]
    root_tokens, node_emb = e_out["root_tokens"], e_out["node_embeddings"]
    num_atoms = cap["breakpoint_forward"][0][2]
    node_mask = torch.arange(node_emb.shape[1])[None, :] >= num_atoms[:, None]
    root_form = cap["breakpoint_forward"][0][3].float()
    i_args = cap["inten_calculation"][0]
    frag_targs, nft = i_args[2], i_args[3]
    F = int(nft.max())
    fpad = torch.zeros(len(nft), F, node_emb.shape[1], dtype=torch.bool)
    for i, rows in enumerate(torch.split(frag_targs.bool(), nft.tolist())):
        fpad[i, :len(rows)] = rows
    pad = torch.arange(F)[None, :] >= nft[:, None]
    adj = i_args[9]
    fmask_f = fpad.float()
    afv = torch.zeros(len(nft), node_emb.shape[1], root_form.shape[1])
    for i, rows in enumerate(torch.split(i_args[5], num_atoms.tolist())):
        afv[i, :len(rows)] = rows
    inten_inputs = (root_tokens, node_emb, torch.einsum("bfn,bne->bfe", fmask_f, afv), root_form,
                    torch.einsum("bfn,bnm,bfm->bf", fmask_f, adj, 1 - fmask_f), pad, ~(fpad | pad[..., None]))

    B, N, Fd = "batch", "atoms", "frags"
    blocks = {
        "encoder": (Encoder(), (g["x"], g["attn_bias"], g["attn_edge_type"], g["spatial_pos"], g["degree"],
                                g["edge_input"], e_args[0], e_args[1], e_kw["instruments"]),
                    ["x", "attn_bias", "attn_edge_type", "spatial_pos", "degree", "edge_input",
                     "adducts", "collision_engs", "instruments"], ["root_tokens", "node_embeddings"],
                    {"x": {0: B, 1: N}, "attn_bias": {0: B, 1: "atoms_plus_1", 2: "atoms_plus_1"},
                     "attn_edge_type": {0: B, 1: N, 2: N}, "spatial_pos": {0: B, 1: N, 2: N},
                     "degree": {0: B, 1: N}, "edge_input": {0: B, 1: N, 2: N},
                     "adducts": {0: B}, "collision_engs": {0: B}, "instruments": {0: B},
                     "root_tokens": {0: B}, "node_embeddings": {0: B, 1: N}}),
        "breakpoints": (Breakpoints(), (root_tokens, node_emb, node_mask, root_form),
                        ["root_tokens", "node_embeddings", "node_mask", "root_form_vecs"], ["scores", "k"],
                        {"root_tokens": {0: B}, "node_embeddings": {0: B, 1: N}, "node_mask": {0: B, 1: N},
                         "root_form_vecs": {0: B}, "scores": {0: B, 2: N}, "k": {0: B}}),
        "intensity": (Intensity(), inten_inputs,
                      ["root_tokens", "node_embeddings", "frag_form_vecs", "root_form_vecs", "num_broken",
                       "frag_mask", "memory_mask"], ["inten"],
                      {"root_tokens": {0: B}, "node_embeddings": {0: B, 1: N}, "frag_form_vecs": {0: B, 1: Fd},
                       "root_form_vecs": {0: B}, "num_broken": {0: B, 1: Fd}, "frag_mask": {0: B, 1: Fd},
                       "memory_mask": {0: B, 1: Fd, 2: N}, "inten": {0: B, 1: Fd}}),
    }

    # Closed-form LinSAT against the real layer, on the reference logits
    bp = cap["breakpoint_forward"][2]
    logit = bp["frag_logits"][-1].masked_fill(node_mask[:, None, :], -1e9)
    for kk in (1, 2, 3):
        real = linsat_layer(logit.reshape(-1, logit.shape[-1]), E=torch.ones(1, logit.shape[-1]),
                            f=torch.full((1,), float(kk)), no_warning=True, max_iter=1, tau=TAU)
        err = (real.reshape(logit.shape) - linsat_sum_k(logit, kk)).abs().max().item()
        print(f"LinSAT closed form, k={kk}: max |diff| = {err:.2e}")
        assert err < 1e-5

    torch.backends.mha.set_fastpath_enabled(False)  # fused attention has no ONNX export
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (mod, inputs, in_names, out_names, axes) in blocks.items():
        mod.eval()
        torch.onnx.export(mod, inputs, args.out / f"{name}.onnx", input_names=in_names, output_names=out_names,
                          dynamic_axes=axes, opset_version=17, dynamo=False)
        print(f"Wrote {args.out / f'{name}.onnx'} ({(args.out / f'{name}.onnx').stat().st_size / 1e6:.0f} MB)")

    enum_ints = lambda xs: [int(x) for x in xs]
    commit = subprocess.run(["git", "-C", str(args.ms_pred), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "source": {"ms_pred_commit": commit, "checkpoint": args.ckpt.name},
        "hparams": {k: hp[k] for k in ("max_broken_bonds", "multi_hop_max_dist", "pe_embed_k", "embed_elem_group")},
        "valid_elements": list(common.VALID_ELEMENTS),
        "element_to_mass": {k: float(v) for k, v in common.ELEMENT_TO_MASS.items()},
        "electron_mass": common.ELECTRON_MASS,
        "ion2mass": {k: float(v) for k, v in common.ion2mass.items()},
        "ion2onehot_pos": common.ion2onehot_pos,
        "instrument2onehot_pos": common.instrument2onehot_pos,
        "featurizer": {
            "atom_nums": list(common.VALID_ATOM_NUM),
            "max_degree": common.MAX_COMMON_DEGREE,
            "max_abs_charge": common.MAX_ABS_FORMAL_CHARGE,
            "hybridization": enum_ints(common.COMMON_HYBRIDIZATION),
            "max_hydrogens": common.COMMON_MAX_HYDROGEN_COUNTS,
            "ring_sizes": list(common.COMMON_RING_SIZES),
            "chirality": enum_ints(common.COMMON_CHIRALITY),
            "element_to_group": {k: v.tolist() for k, v in common.element_to_group.items()},
            "element_group_dim": common.ELEMENT_GROUP_DIM,
            "bond_types": enum_ints(common.COMMON_BOND_TYPES),
            "bond_stereo": enum_ints(common.VALID_BOND_STEREO),
            "edge_feats": model.tree_processor.get_edge_feats(),
            "spatial_pos_max": 1024,
        },
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"Wrote {args.out / 'meta.json'}")

    # End-to-end parity: the port vs ms-pred's predict_mol
    port = glacier.GlacierONNX(args.out)
    mine = port.predict(SMILES, CES, ADDUCTS, INSTRUMENTS)
    for i, s in enumerate(SMILES):
        feats = glacier.featurize(Chem.MolFromSmiles(s), meta)
        n = feats["x"].shape[0]
        x_err = np.abs(feats["x"] - g["x"][i, :n].numpy()).max()
        e_ok = np.array_equal(feats["edge_input"], g["edge_input"][i, :n, :n].numpy())
        ref_i = {"spec": ref["spec"][i].numpy(), "frag": ref["frag"][i][:, :n].numpy().astype(bool)}
        exact = np.array_equal(ref_i["frag"], mine[i]["frag"]) if ref_i["frag"].shape == mine[i]["frag"].shape else False
        spec_err = glacier.max_diff_up_to_symmetry(Chem.MolFromSmiles(s), mine[i], ref_i, port.rows_per_frag)
        print(f"{i} n={n:3d} |x diff|={x_err:.1e} edges_equal={e_ok} rows={len(ref_i['spec']):5d} "
              f"frags_identical={exact} max|spec diff| (up to symmetry)={spec_err:.1e}")
        assert x_err < 1e-5 and e_ok and spec_err < 1e-4, f"port disagrees with torch on {s}"
    print("PARITY OK")

    np.savez_compressed(
        args.fixture,
        smiles=np.array(SMILES), adducts=np.array(ADDUCTS), collision_energies=np.array(CES),
        instruments=np.array(INSTRUMENTS),
        offsets=np.cumsum([0] + [len(s) for s in ref["spec"]]),
        spec=np.concatenate([s.numpy() for s in ref["spec"]]),
        frag=np.concatenate([np.pad(f.numpy().astype(bool), ((0, 0), (0, node_emb.shape[1] - f.shape[1])))
                             for f in ref["frag"]]),
        num_atoms=num_atoms.numpy(),
    )
    print(f"Wrote {args.fixture}")


if __name__ == "__main__":
    main()
