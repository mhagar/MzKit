"""
Benchmark harness for align_ensembles.

Loads a multi-sample .mzk, times align_ensembles across all samples,
and reports wall-time + analyte-count stats. Used to compare the
pre/post-rewrite aligner.

Usage:
    uv run python scratch/bench_align.py [path-to.mzk] [--dup N]

--dup N replicates the loaded samples N times (fresh UUIDs) to probe
how runtime scales toward ~22 samples.
"""
import argparse
import copy
import time
import uuid as _uuid
from collections import Counter
from pathlib import Path

from core.data_structs.alignment import AlignmentParams
from core.cli.align_ensembles import align_ensembles
from core.utils.persistence import load_project

DEFAULT_MZK = Path("tests/test_files/test_alignment.mzk")


def _dup_samples(samples, factor):
    """Replicate samples `factor` times with fresh sample UUIDs."""
    out = []
    for k in range(factor):
        for s in samples:
            if k == 0:
                out.append(s)
                continue
            c = copy.deepcopy(s)
            c.uuid = _uuid.uuid4().int
            c.name = f"{s.name}__dup{k}"
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mzk", nargs="?", default=str(DEFAULT_MZK))
    ap.add_argument("--dup", type=int, default=1,
                    help="replicate samples this many times")
    args = ap.parse_args()

    t0 = time.perf_counter()
    samples, _ = load_project(Path(args.mzk))
    t_load = time.perf_counter() - t0

    if args.dup > 1:
        samples = _dup_samples(samples, args.dup)

    n_ens = sum(
        len(s.injection.ensembles)
        for s in samples if s.injection
    )
    print(f"loaded {len(samples)} samples, {n_ens} ensembles "
          f"in {t_load:.1f}s")

    params = AlignmentParams(
        rt_tolerance=10.0,
        mz_tolerance=0.01,
        ms1_similarity_threshold=0.7,
        ms2_similarity_threshold=0.6,
    )

    t0 = time.perf_counter()
    alignment = align_ensembles(samples, params)
    dt = time.perf_counter() - t0

    sizes = Counter(len(a.ensemble_map) for a in alignment.analytes)
    n_multi = sum(c for k, c in sizes.items() if k >= 2)
    print(f"align: {dt:.2f}s  ->  {alignment.analyte_count} analytes "
          f"({n_multi} multi-sample, {sizes.get(1, 0)} singletons)")
    print(f"  group-size histogram: {dict(sorted(sizes.items()))}")
    print(f"  ENSEMBLES={n_ens}  SAMPLES={len(samples)}  SECONDS={dt:.2f}")


if __name__ == "__main__":
    main()
