import numpy as np

def find_closest_point(
        tgt_x: float | int,
        tgt_y: float | int,
        data_x: np.ndarray,
        data_y: np.ndarray,
        find_idx_only: bool = False,
) -> int | tuple[int, float|int]:
    """
    Given the X and Y position of a point,
    Finds the nearest point in data_X/data_y (based on euclidean distance).

    Either returns just the idx, or both idx and dist**2 depending on
    `find_idx_only`
    """
    dists_squared: np.ndarray = (data_x - tgt_x)**2 + (data_y - tgt_y)**2

    # No need to compute square root, expensive and unneccessary

    idx = int(np.nanargmin(dists_squared))

    if find_idx_only:
        return idx

    return idx, dists_squared[idx]