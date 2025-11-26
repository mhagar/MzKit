"""
This script contains functions for exporting MS spectra in different formats
"""
import numpy as np

def to_sirius_ms(
    compound: str,
    parent_mz: float,
    ms1_spec_arr: np.ndarray,
    ms2_spec_arr: np.ndarray,
) -> str:
    output = [
        f">compound {compound}",
        f">parentmass {parent_mz}",
        "",
        ">ms1",
    ]
    output += _format_spectrum_array(
        arr=ms1_spec_arr,
    )

    output.append("")
    output.append(">collision 60")
    output += _format_spectrum_array(
        arr=ms2_spec_arr
    )

    return "\n".join(output)

def to_mgf(
    pepmass: float,
    charge: int,
    mslevel: int,
    spec_arr: np.ndarray,
    metadata: dict[str, str] | None = None,
) -> str:
    """
    Export spectrum to MGF (Mascot Generic Format)

    Args:
        pepmass: Parent mass (precursor m/z)
        charge: Charge state (e.g., 1 for 1+)
        mslevel: MS level (typically 2 for MS2)
        spec_arr: Spectrum array with 'mz' and 'intsy' fields
        metadata: Optional dictionary of additional metadata fields

    Returns:
        MGF format string
    """
    output = [
        "BEGIN IONS",
        f"PEPMASS={pepmass}",
        f"CHARGE={charge}+",
        f"MSLEVEL={mslevel}",
    ]

    # Add optional metadata fields if provided
    if metadata:
        for key, value in metadata.items():
            output.append(f"{key}={value}")

    # Add spectrum peaks
    output += _format_spectrum_array(arr=spec_arr)

    output.append("END IONS")

    return "\n".join(output)

def _format_spectrum_array(
    arr: np.ndarray,
) -> list[str]:
    """
    Bullshit function, just need to go fast
    """
    output = []
    for mz, intsy in zip(
        arr['mz'],
        arr['intsy'],
    ):
        output.append(f"{mz:.5f} {intsy:.0f}")

    return output