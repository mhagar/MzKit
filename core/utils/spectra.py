"""
Functions utilities for:
- Generating consensus spectra
- Calculating spectral entropy similarity
"""
from __future__ import annotations

from .array_types import SpectrumArray, ConsensusSpectrumArray, to_spec_arr

import numpy as np
from ms_entropy import calculate_entropy_similarity

from typing import Iterable, Generator, Optional


def merge_spectra(
        spectra: Iterable[SpectrumArray],
        bin_width: float,
        max_mz: Optional[float] = None,
) -> ConsensusSpectrumArray:
    """
    Merges spectra using an adaptation of 'BIN' method,
    as described by Luo & Bittremieux et al. (2022)

    0. All SpectrumArrays are intsy-normalized (0 -> 1)

    1. A consensus vector with `bin_width` is
    constructed, up to `max_mz`.
        * Note: if `max_mz` not given, first finds the
           highest m/z across the SpectrumArrays

    2. For each spectrum, peak m/z and intsy is assigned
     to corresponding bin in consensus spectrum
     (tallest peak within bin wins)

    3. Vector is converted into a consensus spectrum
     by averaging m/z and intensity values per bin

    In the original method, peaks are only included
     if present in greater than 25% of child spectra.

    Here, the filtering is differred. The output
     ConsensusSpectrumArray has per-bin `freq`
     (ranges [0, 1]).

    You can threshold at display / print time
    (see `threshold_consensus`)

    Parameters
    :param: spectra: Iterable containing SpectrumArrays
    :param: bin_width: width (in m/z) of each consensus bin
    :param: max_mz: upper m/z bound of the consensus vector;
        inferred from the input spectra when omitted
    """
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")

    # Materialize so we can make two passes (find max_mz, then merge)
    spectra = [s for s in spectra if len(s) > 0]

    if not spectra:
        return _empty_consensus()

    if max_mz is None:
        # Find largest m/z across all input spectra
        max_mz = max(float(s['mz'].max()) for s in spectra)

    if bin_width > (max_mz / 10):
        raise ValueError(
            "bin_width must be smaller than 10% of max_mz"
        )

    # Bins span [0, max_mz]; the peak at exactly max_mz lands in the last bin.
    num_bins = int(np.floor(max_mz / bin_width)) + 1

    # Running per-bin accumulators across spectra
    sum_mz = np.zeros(num_bins, dtype='f8')
    sum_intsy = np.zeros(num_bins, dtype='f8')
    counts = np.zeros(num_bins, dtype='i8')
    # Number of spectra that actually contributed an in-range peak;
    # the denominator for relative frequency.
    n_contributing = 0

    for spectrum in spectra:
        mz = spectrum['mz']
        intsy = spectrum['intsy'].astype('f8', copy=True)

        # Discard peaks above max_mz (out of the consensus range)
        in_range = mz <= max_mz
        mz = mz[in_range]
        intsy = intsy[in_range]
        if len(mz) == 0:
            continue
        n_contributing += 1

        # 0. Normalize intensities to 0 -> 1
        peak_max = intsy.max()
        if peak_max > 0:
            intsy = intsy / peak_max

        # Assign each peak to a bin
        bin_idx = np.floor(mz / bin_width).astype('i8')

        # 2. Tallest peak within a bin wins (per spectrum).
        # Sort ascending by intensity so the tallest peak is written last
        # and overwrites shorter peaks sharing its bin.
        order = np.argsort(intsy, kind='stable')
        bin_mz = np.full(num_bins, np.nan, dtype='f8')
        bin_intsy = np.full(num_bins, np.nan, dtype='f8')
        bin_mz[bin_idx[order]] = mz[order]
        bin_intsy[bin_idx[order]] = intsy[order]

        present = ~np.isnan(bin_intsy)
        sum_mz[present] += bin_mz[present]
        sum_intsy[present] += bin_intsy[present]
        counts[present] += 1

    if n_contributing == 0:
        return _empty_consensus()

    # 3. Average m/z and intensity per bin; drop empty bins. `freq` is the
    # relative frequency: fraction of contributing spectra the bin appeared in.
    filled = counts > 0
    result = np.zeros(
        int(filled.sum()),
        dtype=[('mz', 'f8'), ('intsy', 'f8'), ('freq', 'f8')],
    )
    result['mz'] = sum_mz[filled] / counts[filled]
    result['intsy'] = sum_intsy[filled] / counts[filled]
    result['freq'] = counts[filled] / n_contributing

    return ConsensusSpectrumArray(result)


def _empty_consensus() -> ConsensusSpectrumArray:
    return ConsensusSpectrumArray(
        np.zeros(
            0,
            dtype=[('mz', 'f8'), ('intsy', 'f8'), ('freq', 'f8')],
        )
    )


def threshold_consensus(
        consensus: ConsensusSpectrumArray,
        min_freq: float = 0.25,
) -> SpectrumArray:
    """
    Flattens a ConsensusSpectrumArray to a SpectrumArray,
    keeping only peaks freq >= `min_freq`

    :param min_freq: minimum relative frequency (in [0, 1]) to keep
    """
    names = consensus.dtype.names or ()
    if 'freq' in names:
        consensus = consensus[consensus['freq'] >= min_freq]

    return to_spec_arr(
        mz_arr=consensus['mz'].astype('f8'),
        intsy_arr=consensus['intsy'].astype('f8'),
    )


def entropy_similarity(
        spectrum_a: SpectrumArray,
        spectrum_b: SpectrumArray,
        mz_tol_da: float = 0.02,
        mz_tol_ppm: float = -1.0,
        clean_spectra: bool = False,
) -> float:
    """
    Calculate entropy similarity for two spectrum arrays
     as per Li & Fiehn et al. (2021), via the reference
     MSEntropy implementation.

    Peaks within the given m/z tolerance are matched; the
    similarity is the entropy-weighted overlap, in [0, 1]
    (1 == identical, 0 == no shared peaks).

    :param spectrum_a: first SpectrumArray
    :param spectrum_b: second SpectrumArray
    :param mz_tol_da: peak-match tolerance in Daltons
        (used when > 0; the default MSEntropy behaviour)
    :param mz_tol_ppm: peak-match tolerance in ppm;
        takes precedence over mz_tol_da when > 0
    :param clean_spectra: if True, lets MSEntropy
        normalize/merge peaks first
    """
    # MSEntropy has nothing to match against if either side is empty.
    if len(spectrum_a) == 0 or len(spectrum_b) == 0:
        return 0.0

    return float(
        calculate_entropy_similarity(
            _to_peaks(spectrum_a),
            _to_peaks(spectrum_b),
            ms2_tolerance_in_da=mz_tol_da,
            ms2_tolerance_in_ppm=mz_tol_ppm,
            clean_spectra=clean_spectra,
        )
    )


def _to_peaks(
        spectrum: SpectrumArray
) -> np.ndarray:
    """
    Convert a structured SpectrumArray to the (n, 2) [mz, intsy]
    float32 array MSEntropy expects
    """
    peaks = np.empty(
        shape=(len(spectrum), 2),
        dtype=np.float32,
    )
    peaks[:, 0] = spectrum['mz']
    peaks[:, 1] = spectrum['intsy']
    return peaks


def normalize_spectrum(
        spec: SpectrumArray | np.ndarray,
        max_range: float = 1.0,
) -> np.ndarray:
    """
    Normalize spectrum intensities to a range [0, max_intsy]
    """
    if spec.size == 0:
        return spec

    max_intsy = spec['intsy'].max()
    if max_intsy <= 0:
        return spec

    out = spec.copy()
    out['intsy'] = out['intsy'] / max_intsy * max_range
    return out
