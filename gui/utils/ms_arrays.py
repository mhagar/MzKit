import numpy as np

def zero_pad_arrays(
        mz_arr: np.ndarray,
        intsy_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a spectrum array (dtype 'mz' and 'intsy'), returns
    a copy that's zero-padded (i.e. ideal for making stickplots)
    """
    zero_padded_mz = np.zeros(
        mz_arr.size * 2,
    )

    zero_padded_intsy = np.zeros(
        intsy_arr.size * 2,
    )

    # Fill even indices with original data
    zero_padded_mz[0::2] = mz_arr
    zero_padded_intsy[0::2] = intsy_arr

    # Fill odd indices with zero-intensity data
    zero_padded_mz[1::2] = mz_arr
    zero_padded_intsy[1::2] = 0.0

    return zero_padded_mz, zero_padded_intsy


def strip_empty_values(
    chrom_array: np.ndarray
) -> np.ndarray:
    """
    Given a chrom array (dtype 'rt' and 'intsy'), returns
    a copy with no rt = 0 elements

    (ideal for plotting chromatograms)
    :param chrom_array:
    :return:
    """
    return chrom_array[
        np.where(chrom_array['rt'] > 0)
    ]