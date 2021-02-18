#!/usr/bin/env python

"""Tests for `mode_sorter` package."""

import pytest


from mode_sorter import mode_sorter
import numpy as np


def test_transfer_coefficient():
    x, y = np.mgrid[-1:1:100j, -1:1:100j]
    assert type(mode_sorter.transfer_coefficient(x, y, 0.0001)) == np.ndarray
