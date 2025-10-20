"""
Defines custom numpy array type (I use this alot; ergonomic)
"""

import numpy as np
from numpy.typing import NDArray

from typing import NewType

SpectrumArray = NewType(
    name='SpectrumArray',
    tp=NDArray[
        np.dtype(  # type: ignore
            [
                ('mz', 'f8'),
                ('intsy', 'f8'),
            ]
        )
    ]
)

def to_spec_arr(
    mz_arr: NDArray[np.float64],
    intsy_arr: NDArray[np.float64],
) -> SpectrumArray:
    result = np.zeros(
        len(mz_arr),
        dtype= [
            ('mz', 'f8'),
            ('intsy', 'f8'),
        ]
    )
    result['mz'] = mz_arr
    result['intsy'] = intsy_arr
    return SpectrumArray(result)

ChromArray = NewType(
    name='ChromArray',
    tp=NDArray[
        np.dtype(  # type: ignore
            [
                ('rt', 'f8'),
                ('intsy', 'f8'),
            ]
        )
    ]
)

def to_chrom_arr(
    rt_arr: NDArray[np.float64],
    intsy_arr: NDArray[np.float64],
) -> ChromArray:
    result = np.zeros(
        len(rt_arr),
        dtype= [
            ('rt', 'f8'),
            ('intsy', 'f8'),
        ]
    )
    result['rt'] = rt_arr
    result['intsy'] = intsy_arr
    return ChromArray(result)

EnsembleArray = NewType(
    name='EnsembleArray',
    tp=NDArray[
        np.dtype(  # type: ignore
            [
                ('mz', 'f8'),
                ('intsy', 'f8'),
                ('rt', 'f8'),
            ]
        )
    ]
)

def to_ensemble_arr(
    rt_arr: NDArray[np.float64],
    mz_arrs: list[NDArray[np.float64]],
    intsy_arrs: list[NDArray[np.float64]],
) -> EnsembleArray:
    num_cofeatures: int = len(mz_arrs)
    num_scans: int = len(mz_arrs[0])

    ensemble_array = np.zeros(
        shape=(num_scans, num_cofeatures),
        dtype=[
            ("mz", "f8"),
            ("intsy", "f8"),
            ("rt", "f8"),
        ],
    )

    for idx, ( mz_arr, intsy_arr  ) in enumerate(
        zip(
            mz_arrs, intsy_arrs
        )
    ):
        ensemble_array[:, idx]['mz'] = mz_arr
        ensemble_array[:, idx]['intsy'] = intsy_arr
        ensemble_array[:, idx]['rt'] = rt_arr

    return EnsembleArray(ensemble_array)
