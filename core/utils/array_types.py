"""
Defines custom numpy array type (I use this alot; ergonomic)
"""

import numpy as np
from numpy.typing import NDArray

from typing import NewType

SpectrumArray = NewType(
    name='SpectrumArray',
    tp=NDArray[
        np.dtype(
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
        np.dtype(
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

