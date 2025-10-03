"""
Ergonomic wrappers for working with PyOpenMS
"""
import numpy as np
import pyopenms as oms

from typing import Optional, Literal

from pyopenms import MSChromatogram


def generate_chromatogram(
    exp: oms.MSExperiment,
    min_rt: Optional[float] = None,
    max_rt: Optional[float] = None,
    min_mz: Optional[float] = None,
    max_mz: Optional[float] = None,
    ms_level: int = 1,
    chrom_type: Literal['XIC', 'BPC'] = 'BPC'
) -> oms.MSChromatogram:
    """
    Given an MSExperiment, generates a base peak chromatogram.

    # TODO: extractXICsFromMatrix() can take many ranges at once; consider
    # TODO: implementing that functionality here

    If min/max rt/mz are not specified, extracts entire experiment
    :param ms_level: Default: 1
    :param chrom_type: 'XIC' or 'BPC'. Default: XIC
    :param exp: MSExperiment object
    :param min_rt:
    :param max_rt:
    :param min_mz:
    :param max_mz:
    :return:
    """
    if not min_rt and not max_rt:
        min_rt = exp.getMinRT()
        max_rt = exp.getMaxRT()

    if not min_mz and not max_mz:
        min_mz = exp.getMinMZ()
        max_mz = exp.getMaxMZ()

    match chrom_type:
        case 'BPC':
            agg = b'max'
        case 'XIC':
            agg = b'sum'
        case _:
            raise ValueError(
                f"Invalid chrom_type argument ({chrom_type}"
            )

    ranges_matrix = oms.MatrixDouble.fromNdArray(
        np.array(
            [[min_mz, max_mz, min_rt, max_rt]]
        )
    )
    chroms: list[MSChromatogram] = exp.extractXICsFromMatrix(
        ranges=ranges_matrix,
        ms_level=ms_level,
        mz_agg=agg,
    )

    # chrom.setChromatogramType(3)  # Corresponds to BPC

    if len(chroms) > 0:
        return chroms[0]

    return MSChromatogram()


def retrieve_spectrum_at_rt(
    exp: oms.MSExperiment,
    rt: float,
) -> oms.MSSpectrum:
    """
    Given an MSExperiment and a retention time, retrieves the
    spectrum from the closest retention time
    :param exp:
    :param rt:
    :return:
    """
    pass

