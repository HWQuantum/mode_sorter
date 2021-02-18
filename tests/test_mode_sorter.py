#!/usr/bin/env python

"""Tests for `mode_sorter` package."""

import pytest


from mode_sorter import mode_sorter, generation
import numpy as np


def test_transfer_coefficient():
    x, y = np.mgrid[-1:1:100j, -1:1:100j]
    assert type(mode_sorter.transfer_coefficient(x, y, 0.0001)) == np.ndarray


def test_transfer_matrix_generation_works_with_multiple_wavelengths():
    x, y = np.mgrid[-1:1:100j, -1:1:100j]
    t = mode_sorter.transfer_matrix(x, y, 0.001, 0.002, 0.2)
    assert t.shape == (100, 100)
    t = mode_sorter.transfer_matrix(x, y, [0.001, 0.002], 0.2, 0.2)
    assert t.shape == (2, 100, 100)


def test_mode_propagation_doesnt_crash():
    x, y = np.mgrid[-1:1:100j, -1:1:100j]
    wavelengths = [0.001, 0.002]
    in_field = np.zeros((3, 2, *x.shape), dtype=np.complex128)
    out_field = np.zeros((3, 2, *x.shape), dtype=np.complex128)
    t = mode_sorter.transfer_matrix(x, y, wavelengths, 0.2)
    in_field[0, 0] = generation.generate_spot(
        (0, 0), 0.2, wavelengths[0], 0, x, y)
    in_field[0, 1] = generation.generate_spot(
        (0, 0), 0.2, wavelengths[1], 0, x, y)
    out_field[-1,
              0] = generation.generate_spot((0, 0), 0.2, wavelengths[0], 0, x, y)
    out_field[-1,
              1] = generation.generate_spot((0, 0), 0.2, wavelengths[1], 0, x, y)
    masks = np.ones((3, *x.shape), dtype=np.complex128)
    mode_sorter.propagate_field(in_field, out_field, masks, t)
    assert np.nonzero(in_field[-1])
