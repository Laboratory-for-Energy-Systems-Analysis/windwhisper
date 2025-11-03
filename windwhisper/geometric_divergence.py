"""
This module models the geometric divergence of sound waves.
"""

import numpy as np


def get_geometric_spread_loss(distance: np.array) -> np.array:
    """Calculate the geometric spread loss in accordance with ISO 9613-2:2024.

    :param distance: Distance between source and receiver in metres.
    :type distance: numpy.ndarray | float
    :returns: Geometric divergence loss in dB.
    :rtype: numpy.ndarray | float
    """
    return 20 * np.log10(distance / 1) + 11
