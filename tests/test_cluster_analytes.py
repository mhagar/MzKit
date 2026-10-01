"""
Tests for MS2 modified-cosine clustering of AlignedAnalytes
(core/cli/cluster_analytes.py), on synthetic spectra.
"""
from types import SimpleNamespace

import numpy as np

from core.cli.cluster_analytes import ClusterParams, cluster_analytes
from core.data_structs.alignment import AlignedAnalyte, EnsembleAlignment
from core.utils.array_types import to_spec_arr

SAMPLE = 1

# Fragment patterns; B is A shifted by +14 (CH2) on precursor + one fragment
FRAGS_A = [(80.0, 30.0), (95.0, 50.0), (120.0, 100.0), (150.0, 40.0)]
FRAGS_B = [(80.0, 30.0), (95.0, 50.0), (134.0, 100.0), (150.0, 40.0)]
FRAGS_C = [(61.0, 100.0), (73.0, 60.0), (89.0, 20.0), (201.0, 70.0)]


def _spec(peaks):
    mz, intsy = zip(*peaks)
    return to_spec_arr(
        mz_arr=np.asarray(mz, dtype=float),
        intsy_arr=np.asarray(intsy, dtype=float),
    )


def _build(entries):
    """
    entries: (consensus_rt, precursor_mz, ms2 peaks | None) per analyte.
    Returns (alignment, samples, analytes).
    """
    ensembles, analytes = {}, []
    for i, (rt, precursor, frags) in enumerate(entries):
        ens_uuid = 1000 + i
        ensembles[ens_uuid] = SimpleNamespace(
            uuid=ens_uuid,
            base_intsy=1e6,
            composite_spectrum=SimpleNamespace(
                ms1=_spec([(precursor, 1.0)]),
                ms2=_spec(frags) if frags else None,
                precursor_mz=precursor,
            ),
        )
        analytes.append(AlignedAnalyte(
            ensemble_map={SAMPLE: ens_uuid}, consensus_rt=rt,
        ))
    sample = SimpleNamespace(
        uuid=SAMPLE, injection=SimpleNamespace(ensembles=ensembles),
    )
    alignment = EnsembleAlignment(sample_uuids=(SAMPLE,), analytes=analytes)
    return alignment, [sample], analytes


def test_related_analytes_cluster_and_no_ms2_dropped():
    alignment, samples, analytes = _build([
        (300.0, 200.0, FRAGS_A),
        (100.0, 250.0, FRAGS_C),
        (500.0, 214.0, FRAGS_B),     # analogue of the first
        (50.0, 180.0, None),         # no MS2
    ])
    result = cluster_analytes(alignment, samples, ClusterParams())

    assert result.n_dropped_no_ms2 == 1
    assert analytes[3].uuid not in result.analyte_uuids
    assert len(result.analyte_uuids) == 3

    slot = {u: i for i, u in enumerate(result.analyte_uuids)}
    a, b, c = (slot[analytes[i].uuid] for i in (0, 2, 1))
    assert abs(a - b) == 1                     # adjacent leaves
    assert result.similarity[a, b] > 0.9       # shift-matched
    assert result.similarity[a, c] == 0.0

    # One real merge, covering exactly the two analogues
    assert len(result.nodes) == 1
    node = result.nodes[0]
    assert (node['lo'], node['hi']) == (min(a, b), max(a, b))
    assert node['height'] < 0.1
    assert {node['x_left'], node['x_right']} == {a, b}


def test_unrelated_analytes_form_rt_sorted_comb():
    rng = np.random.default_rng(0)
    entries = []
    for i, rt in enumerate(rng.permutation(np.arange(10) * 30.0)):
        # Disjoint fragments per analyte -> zero similarity everywhere
        base = 100.0 + 50.0 * i
        frags = [(base + k * 7.1, 10.0 + k) for k in range(4)]
        entries.append((float(rt), 500.0 + i, frags))
    alignment, samples, analytes = _build(entries)

    result = cluster_analytes(alignment, samples, ClusterParams())
    rt_of = {a.uuid: a.consensus_rt for a in analytes}
    rts = [rt_of[u] for u in result.analyte_uuids]
    assert rts == sorted(rts)
    assert len(result.nodes) == 0              # no merges below 1.0


def test_min_matched_peaks_zeroes_weak_matches():
    alignment, samples, _ = _build([
        (100.0, 200.0, FRAGS_A),
        (200.0, 214.0, FRAGS_B),
    ])
    strict = cluster_analytes(
        alignment, samples, ClusterParams(min_matched_peaks=10),
    )
    assert strict.similarity[0, 1] == 0.0
    assert len(strict.nodes) == 0


def test_cancel_returns_none():
    import threading
    alignment, samples, _ = _build([
        (100.0, 200.0, FRAGS_A),
        (200.0, 214.0, FRAGS_B),
    ])
    cancel = threading.Event()
    cancel.set()
    assert cluster_analytes(
        alignment, samples, ClusterParams(), cancel_event=cancel,
    ) is None
