"""
This script contains functions used to generate the
'fingerprint graphic' displayed in the fingerprint viewer
"""

from PyQt5 import QtGui
import numpy as np
import pyqtgraph as pg

def array_to_pixmap(
    array: np.ndarray,
    colormap: pg.colormap.ColorMap,
) -> QtGui.QPixmap:
    """
    Given a 1D array, converts it into a QPixmap with
    dimensions 1px * Npx (where N is the size of the array)

    It's best to call this function once then cache the results

    :param colormap:
    :param array:
    :return:
    """
    height = 1
    width = len(array)

    image = QtGui.QImage(
        width,
        height,
        QtGui.QImage.Format_RGB32,
    )

    # Fill image pixel by pixel
    for x in range(width):
        value = array[x]
        color = colormap.map(
            value,
            mode=pg.ColorMap.QCOLOR,
        )
        image.setPixelColor(x, 0, color)

    return QtGui.QPixmap.fromImage(
        image
    )