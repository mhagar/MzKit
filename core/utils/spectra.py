"""
Functions utilities for:
- Generating consensus spectra
- Calculating spectral entropy similarity
"""
from .array_types import SpectrumArray, ConsensusSpectrumArray

import numpy as np

from typing import Iterable, Optional


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
     In this case - the output ConsensusSpectrumArray
     has a field `freq` which can be used post-hoc to
     filter out infrequent peaks according to desired
     threshold.

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

    for spectrum in spectra:
        mz = spectrum['mz']
        intsy = spectrum['intsy'].astype('f8', copy=True)

        # Discard peaks above max_mz (out of the consensus range)
        in_range = mz <= max_mz
        mz = mz[in_range]
        intsy = intsy[in_range]
        if len(mz) == 0:
            continue

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

    # 3. Average m/z and intensity per bin; drop empty bins.
    filled = counts > 0
    result = np.zeros(
        int(filled.sum()),
        dtype=[('mz', 'f8'), ('intsy', 'f8'), ('freq', 'i8')],
    )
    result['mz'] = sum_mz[filled] / counts[filled]
    result['intsy'] = sum_intsy[filled] / counts[filled]
    result['freq'] = counts[filled]

    return ConsensusSpectrumArray(result)


def _empty_consensus() -> ConsensusSpectrumArray:
    return ConsensusSpectrumArray(
        np.zeros(
            0,
            dtype=[('mz', 'f8'), ('intsy', 'f8'), ('freq', 'i8')],
        )
    )


def entropy_similarity(
        spectrum_a: SpectrumArray,
        spectrum_b: SpectrumArray,
):
    """
    Calculate entropy similarity for two spectrum arrays
     as per Li & Fiehn et al. (2021)
    """
    
