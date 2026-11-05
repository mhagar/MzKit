"""DDA precursor seeding honours parent_threshold (features are built at cofeature_threshold)."""
from types import SimpleNamespace

import numpy as np

import core.cli.auto_extract_ensembles as aee


def test_precursor_seeds_respect_parent_threshold(monkeypatch):
    pfs = [aee.PrecursorFeature(mz_lane_idx=i, apex_scan=i, mz=100.0 + i, apex_intsy=v,
                                ms2_scan_idxs=np.array([i]))
           for i, v in enumerate([2e3, 5e4, 1.9e4, 3e4])]
    monkeypatch.setattr(aee, '_build_precursor_features', lambda injection, config: pfs)
    injection = SimpleNamespace(scan_array_ms1=None,
                                scan_array_ms2=SimpleNamespace(precursor_mz_arr=np.zeros(1)))
    config = aee.dda_config(min_intsy=1e3, parent_threshold=2e4)

    seeder = aee.PrecursorSeedProvider(injection, config)
    seeds = iter(seeder.next_seed, None)
    assert [s.apex_intsy for s in seeds] == [5e4, 3e4]
