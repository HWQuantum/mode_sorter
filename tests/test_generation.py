"""Tests for `mode_sorter` package."""

import pytest
from mode_sorter import generation
import numpy as np


def test_generate_circular_spots_with_central_spot():
    x, y = np.mgrid[-1:1:100j, -1:1:100j]
    modes = generation.generate_circular_spots_with_central_spot(
        3, 0.2, 0.2, 0, 0.05, x, y)
    assert modes.shape == (3, *x.shape)
