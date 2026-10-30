"""
Tests for the onnxruntime GLACIER port (core/utils/glacier.py) against ms-pred's
torch predict_mol outputs in glacier_parity.npz (written by
tools/glacier/export_onnx.py). Skipped when models/glacier/ is missing.
"""
import re
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from core.utils import glacier

FIXTURE = Path(__file__).parent / "glacier_parity.npz"
needs_model = pytest.mark.skipif(not glacier.model_available(), reason="GLACIER model not in models/glacier/")


@needs_model
def test_predictions_match_torch():
    with np.load(FIXTURE) as f:
        smiles = [str(s) for s in f["smiles"]]
        model = glacier.load_model()
        preds = model.predict(smiles, f["collision_energies"], [str(a) for a in f["adducts"]],
                              [str(i) for i in f["instruments"]])
        o = f["offsets"]
        for i, (s, pred) in enumerate(zip(smiles, preds)):
            ref = {"spec": f["spec"][o[i]:o[i + 1]], "frag": f["frag"][o[i]:o[i + 1], :f["num_atoms"][i]]}
            err = glacier.max_diff_up_to_symmetry(Chem.MolFromSmiles(s), pred, ref, model.rows_per_frag)
            assert err < 1e-4, f"{s}: max |spec diff| {err}"


@needs_model
def test_predicts_shapes_other_than_the_export_batch():
    # The blocks were traced on the fixture batch; a lone, smaller molecule must still run
    (pred,) = glacier.load_model().predict(["OC(=O)c1ccccc1"], np.array([30.0]), ["[M+H]+"], ["Orbitrap"])
    assert len(pred["spec"]) and pred["spec"][:, 1].max() > 0


def test_fragments_are_components_left_after_breaks():
    # Butane C0-C1-C2-C3; deleting C1 leaves {C0} (dropped: a lone atom reaches nothing) and {C2, C3}
    adj = np.zeros((1, 4, 4), np.float32)
    for a, b in ((0, 1), (1, 2), (2, 3)):
        adj[0, a, b] = adj[0, b, a] = 1
    breaks = np.zeros((1, 2, 4), bool)
    breaks[0, 0, 1] = True
    breaks[0, 1, 3] = True  # deleting C3 leaves {C0, C1, C2}
    frags, counts = glacier.breakpoints_to_fragments(breaks, adj, np.array([4]))
    assert counts.tolist() == [2]
    assert frags[0].tolist() == [[False, False, True, True], [True, True, True, False]]


def test_missing_model_message(monkeypatch, tmp_path):
    monkeypatch.setattr(glacier, "MODEL_DIR", tmp_path)
    glacier.load_model.cache_clear()
    assert not glacier.model_available()
    with pytest.raises(FileNotFoundError, match="export_onnx.py"):
        glacier.load_model()
    glacier.load_model.cache_clear()


def test_ion_formula_hill_order_and_adducts():
    el = ['C', 'N', 'O', 'H', 'Na']
    form = np.array([7, 0, 1, 7, 0])   # a C7H7O fragment, with its own Hs
    assert glacier.ion_formula(form, 0, '[M+H]+', False, el) == 'C7H8O+'
    assert glacier.ion_formula(form, -2, '[M+H]+', True, el) == 'C7H5O•+'
    assert glacier.ion_formula(form, 1, '[M+Na]+', False, el) == 'C7H8NaO+'
    assert glacier.ion_formula(form, 0, '[M+H-H2O]+', False, el) == 'C7H6+'
    assert glacier.ion_formula(np.array([0, 1, 0, 3, 0]), 0, '[M+H]+', False, el) == 'H4N+'


@needs_model
@pytest.mark.parametrize('adduct', ['[M+H]+', '[M+Na]+', '[M-H]-'])
def test_ion_formula_mass_matches_predicted_mz(adduct):
    """
    Every predicted row's ion formula weighs what GLACIER put at its m/z. (GLACIER
    can push H below zero, e.g. C2H-1- for [M-H]-: the formula reports it as is.)
    """
    model = glacier.load_model()
    [pred] = model.predict(['CCOC(=O)c1ccc(N)cc1'], np.array([30.0]), [adduct], ['Orbitrap'])
    mass, e = model.meta['element_to_mass'], model.meta['electron_mass']
    rows = np.flatnonzero(pred['spec'][:, 1] > 0)
    for r in rows:
        ion = glacier.ion_formula(pred['form'][r], pred['h_shift'][r], adduct, pred['radical'][r], model.elements)
        formula, charge = ion.rstrip('•+-'), ion[-1]
        neutral = sum(mass[s] * int(n or 1) for s, n in re.findall(r"([A-Z][a-z]*)(-?[0-9]*)", formula))
        assert neutral - e * (1 if charge == '+' else -1) == pytest.approx(pred['spec'][r, 0], abs=1e-4), ion
