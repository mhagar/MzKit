import json

import numpy as np
import pytest

from core.labeling import peak_model as pm
from core.labeling.peak_model import evaluation, inference, training
from core.labeling.schema import LabelFile, LaneLabel, insert_span


def synthetic_lane(rng, n=2000):
    """UPLC-like lane: narrow peaks (5-8 scans) on a zero baseline with dropouts,
    some touching pairs, 3-point blips (unlabeled = other), and a streaking
    plateau. Returns (intsy, spans)."""
    y = np.zeros(n)
    spans = []
    t = np.arange(n)
    # plateau: long noisy smear
    p0 = int(rng.integers(100, n - 700))
    p1 = p0 + int(rng.integers(250, 600))
    y[p0:p1] += 10 ** rng.uniform(3.5, 5) * (1 + 0.3 * rng.standard_normal(p1 - p0))
    spans.append((p0, p1, "plateau"))
    free = [(0, p0 - 20), (p1 + 20, n)]
    for lo, hi in free:
        c = lo + 30
        while c < hi - 40:
            amp = 10 ** rng.uniform(3.5, 6.5)
            sig = rng.uniform(1.0, 1.8)
            if rng.random() < 0.25:  # 3-point blip: unlabeled
                y[c - 1:c + 2] += amp * np.array([0.5, 1, 0.5]) * 0.05
            elif rng.random() < 0.3:  # touching pair
                c2 = c + int(4 * sig)
                y += amp * np.exp(-0.5 * ((t - c) / sig) ** 2)
                y += 0.7 * amp * np.exp(-0.5 * ((t - c2) / sig) ** 2)
                v = c + int(np.argmin(y[c:c2]))
                spans += [(c - int(3 * sig), v, "peak"), (v, c2 + int(3 * sig) + 1, "peak")]
            else:
                y += amp * np.exp(-0.5 * ((t - c) / sig) ** 2)
                spans.append((c - int(3 * sig), c + int(3 * sig) + 1, "peak"))
            c += int(rng.integers(30, 120))
    y[y < 500] = 0  # centroided: below threshold reads as exactly zero
    y[rng.random(n) < 0.05] = 0  # dropouts
    return y, sorted(spans)


def test_insert_span_overwrites_and_splits():
    spans = [(0, 10, "plateau")]
    spans = insert_span(spans, 3, 6, "peak")
    assert spans == [(0, 3, "plateau"), (3, 6, "peak"), (6, 10, "plateau")]
    assert insert_span(spans, 0, 10, "peak") == [(0, 10, "peak")]


def test_targets_mark_touching_peak_edges():
    cls, bnd = pm.lane_targets(20, [(2, 6, "peak"), (6, 9, "peak"), (12, 18, "plateau")])
    assert cls[2:9].tolist() == [1] * 7 and cls[12:18].tolist() == [2] * 6
    assert np.flatnonzero(bnd).tolist() == [5, 6]
    _, spans = pm.decode(np.stack([
        np.where(cls == 0, 5, 0), np.where(cls == 1, 5, 0),
        np.where(cls == 2, 5, 0), np.where(bnd > 0, 5, -5),
    ]).astype(float))
    assert spans == [(2, 6, "peak"), (6, 9, "peak"), (12, 18, "plateau")]


def test_label_file_roundtrip(tmp_path):
    lbl = LaneLabel(mz=100.0, lane_idx=7, rt_values=[0.0, 1.0], intsy_values=[0.0, 5.0],
                    spans=[(0, 2, "peak")], sample_uuid=2 ** 100)
    path = tmp_path / "labels.json"
    LabelFile(extraction_params={"parent_threshold": 1e4}, lanes=[lbl]).to_json(path)
    assert json.loads(path.read_text())["schema_version"] == 4
    back = LabelFile.from_json(path)
    assert back.lanes[0].spans == [(0, 2, "peak")] and back.has_lane(2 ** 100, 7)

    example, = pm.load_examples([path])
    intsy_values, spans, rt_values = example
    np.testing.assert_array_equal(intsy_values, [0.0, 5.0])
    np.testing.assert_array_equal(rt_values, [0.0, 1.0])
    assert spans == [(0, 2, "peak")]


def test_numpy_inference_matches_torch():
    torch = pytest.importorskip("torch")
    net = training._torch_net()
    model = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    y, _ = synthetic_lane(np.random.default_rng(0), n=1200)
    with torch.no_grad():
        ref = net(torch.from_numpy(pm.lane_features(y)[None]))[0].numpy()
    np.testing.assert_allclose(pm.forward(model, y), ref, rtol=1e-3, atol=1e-4)


def test_model_learns_synthetic_lanes():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    train = [synthetic_lane(rng) for _ in range(30)]
    test = [synthetic_lane(rng) for _ in range(6)]
    model = pm.train(train, steps=600)
    matches = evaluation.MatchCounts()
    for y, spans in test:
        _, pred = pm.segment_lane(model, y)
        matches += evaluation._peak_match_counts(
            [(s, e) for s, e, k in pred if k == "peak"],
            [(s, e) for s, e, k in spans if k == "peak"],
        )
    precision = matches.true_positives / matches.n_predicted
    recall = matches.true_positives / matches.n_expected
    assert precision > 0.8 and recall > 0.8, matches


def test_review_predictions_scores_out_of_fold_and_sorts(monkeypatch):
    lanes = [LaneLabel(mz=1.0, lane_idx=i, rt_values=[0.0] * 50,
                       intsy_values=[0.0] * 10 + [5e4] * 5 + [0.0] * 35,
                       spans=[(10, 15, "peak")], sample_uuid=1) for i in range(10)]
    seen = []
    monkeypatch.setattr(
        evaluation,
        "train",
        lambda ex, **kw: seen.append(len(ex)) or len(ex),
    )

    def fake_segment(model, intsy, rt=None):
        # lanes whose index is odd: model misses the peak and invents another
        bad = int(intsy[49]) == 1
        spans = [(30, 35, "peak")] if bad else [(10, 15, "peak")]
        return np.zeros(len(intsy), int), spans
    for i in range(1, 10, 2):
        lanes[i].intsy_values[49] = 1.0
        lanes[i].intsy_values[30:35] = [5e4] * 5
    monkeypatch.setattr(evaluation, "segment_lane", fake_segment)
    out = pm.review_predictions(lanes, parent_threshold=1e4, n_folds=5)
    assert seen == [8] * 5  # each fold trains without its own 2 lanes
    assert sorted(o["key"][1] for o in out) == [1, 3, 5, 7, 9]
    assert all(o["fp"] == 1 and o["fn"] == 1 for o in out)


def test_segment_lane_trims_spans_to_signal(monkeypatch):
    y = np.zeros(30); y[5:8] = 1e5; y[10:12] = 1e4  # gap at 8-9 stays inside
    monkeypatch.setattr(inference, "forward", lambda m, intsy, rt=None: None)
    monkeypatch.setattr(inference, "decode", lambda logits: (
        None, [(0, 15, "plateau"), (20, 25, "peak")]))
    assert pm.segment_lane({}, y)[1] == [(5, 12, "plateau")]


def test_run_fraction_channel():
    y = np.ones(5)
    features = pm.lane_features(y, rt=np.array([10.0, 11.0, 13.0, 16.0, 20.0]))
    np.testing.assert_allclose(features[-1], [0, 0.1, 0.3, 0.6, 1.0])


def test_segmenter_crops_match_full_lane():
    """near() (batched crops) must give exactly the full-lane answer, including
    near lane edges and for spans wider than the exact zone."""
    import scipy.sparse as sp
    rng = np.random.default_rng(3)
    model = {k: v.astype(np.float32) for k, v in _random_model(rng).items()}
    # make the wide block read as one plateau: bias the plateau logit up
    model["out.bias"] = model["out.bias"] + np.array([0, 50, 0, -50], np.float32)  # wide PEAK
    lanes = [synthetic_lane(rng, n=1500)[0] for _ in range(4)]
    lanes[1][600:900] = 5e4  # one very wide span (> exact zone)
    lanes[1][750 + pm.EXACT] = 0  # dropout exactly at the exact-zone edge of scan 750

    class SA:  # minimal ScanArray stand-in
        intsy_arr = sp.csr_matrix(np.array(lanes))
        rt_arr = np.linspace(0, 600, 1500)
    seg = pm.LaneSegmenter(model, SA())
    qs = [(l, s) for l in range(4) for s in (0, 3, 100, 299, 700, 750, 1200, 1499)]
    near = seg.near([l for l, _ in qs], [s for _, s in qs])
    ref = pm.LaneSegmenter(model, SA())
    for (l, s), got in zip(qs, near):
        cls, spans = ref.lane(l)
        want = inference.containing_span(spans, s)
        assert got[0] == int(cls[s]), (l, s)
        if want is not None and want[2] == "peak":
            assert got[1] == want, (l, s)  # peak extents exact
        elif want is not None:  # other spans: same class, may be clipped
            assert got[1][2] == want[2] and want[0] <= got[1][0] <= s < got[1][1] <= want[1]


def _random_model(rng):
    C, K = pm.CHANNELS, pm.KERNEL
    m = {"inp.weight": rng.normal(0, 1, (C, pm.N_IN, 1)), "inp.bias": rng.normal(0, .1, C),
         "out.weight": rng.normal(0, .5, (4, C, 1)), "out.bias": rng.normal(0, .1, 4)}
    for i in range(len(pm.DILATIONS)):
        m[f"blocks.{i}.weight"] = rng.normal(0, 0.3 / np.sqrt(C * K), (C, C, K))
        m[f"blocks.{i}.bias"] = rng.normal(0, .05, C)
    return m


def test_unclaimed_run_stops_at_earlier_claims():
    from core.cli.auto_extract_ensembles import LaneMaxSeedProvider
    prov = LaneMaxSeedProvider.__new__(LaneMaxSeedProvider)
    prov.assigned_ranges = {7: [(10, 20), (40, 50)]}
    assert prov.unclaimed_run(7, 0, 100, 30) == (20, 40)
    assert prov.unclaimed_run(7, 25, 35, 30) == (25, 35)
    assert prov.unclaimed_run(7, 0, 100, 15) == (15, 15)  # scan itself claimed
    assert prov.unclaimed_run(3, 0, 100, 15) == (0, 100)  # no claims on lane
