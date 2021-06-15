#!/usr/bin/env python
"""Tests for `mode_sorter` package."""

import pytest

from mode_sorter import mode_sorter, generation
import numpy as np


def test_transfer_coefficient():
    x, y = np.mgrid[-1:1:100j, -1:1:100j]
    coeff = mode_sorter.transfer_coefficient(x, y, 0.0001)
    assert type(coeff) == np.ndarray
    assert coeff.shape == x.shape


def test_transfer_matrix_generation_works_with_multiple_wavelengths():
    x, y = np.mgrid[-1:1:100j, -1:1:100j]
    t = mode_sorter.transfer_matrix(x, y, 0.001, 0.002, 0.2)
    assert t.shape == (1, 1, 100, 100)
    t = mode_sorter.transfer_matrix(x, y, [0.001, 0.002], 0.2, 0.2)
    assert t.shape == (1, 2, 100, 100)
    assert np.allclose(
        mode_sorter.transfer_matrix(x, y, 0.001, 0.002, 0.2),
        mode_sorter.transfer_matrix(x, y, [0.001], [0.002], 0.2))


def test_mode_propagation_with_multiple_wavelengths_doesnt_crash():
    x, y = np.mgrid[-1:1:100j, -1:1:100j]
    wavelengths = [0.001, 0.002]
    in_field = np.zeros((3, 2, *x.shape), dtype=np.complex128)
    out_field = np.zeros((3, 2, *x.shape), dtype=np.complex128)
    t = mode_sorter.transfer_matrix(x, y, wavelengths, 0.2)
    in_field[0, 0] = generation.generate_spot((0, 0), 0.2, wavelengths[0], 0,
                                              x, y)
    in_field[0, 1] = generation.generate_spot((0, 0), 0.2, wavelengths[1], 0,
                                              x, y)
    out_field[-1, 0] = generation.generate_spot((0, 0), 0.2, wavelengths[0], 0,
                                                x, y)
    out_field[-1, 1] = generation.generate_spot((0, 0), 0.2, wavelengths[1], 0,
                                                x, y)
    masks = np.ones((3, *x.shape), dtype=np.complex128)
    mode_sorter.propagate_field(in_field,
                                out_field,
                                masks,
                                t,
                                transfer_indices=[(None, 1), (1, None)])
    assert np.nonzero(in_field[-1])
    assert in_field.shape == (3, 2, *x.shape)
    assert t.shape == (1, 2, *x.shape)


def test_mode_propagation_with_single_wavelength_doesnt_crash():
    x, y = np.mgrid[-1:1:100j, -1:1:100j]
    wavelength = 0.001
    in_field = np.zeros((3, 2, *x.shape), dtype=np.complex128)
    out_field = np.zeros((3, 2, *x.shape), dtype=np.complex128)
    t = mode_sorter.transfer_matrix(x, y, [wavelength, wavelength], [0.2, 0.2])
    in_field[0, 0] = generation.generate_spot((0, 0), 0.2, wavelength, 0, x, y)
    in_field[0, 1] = generation.generate_spot((0, 0), 0.2, wavelength, 0, x, y)
    out_field[-1, 0] = generation.generate_spot((0, 0), 0.2, wavelength, 0, x,
                                                y)
    out_field[-1, 1] = generation.generate_spot((0, 0), 0.2, wavelength, 0, x,
                                                y)
    masks = np.ones((3, *x.shape), dtype=np.complex128)
    mode_sorter.propagate_field(in_field, out_field, masks, t)
    assert np.nonzero(in_field[-1])
    assert in_field.shape == (3, 2, *x.shape)
    assert t.shape == (2, 2, *x.shape)


def test_generating_transfer_function_with_multiple_distances():
    x, y = np.mgrid[-1:1:100j, -1:1:100j]
    wavelength = 0.001
    distances = np.linspace(10e-3, 20e-3, 4)
    transfers = mode_sorter.transfer_matrix(x, y, wavelength, distances)
    assert transfers.shape == (distances.size, 1, *x.shape)
    transfers = mode_sorter.transfer_matrix(x, y, [wavelength, wavelength],
                                            distances)
    assert transfers.shape == (distances.size, 2, *x.shape)


def test_thresholding_function():
    masks = np.exp(1j * 2 * np.pi * np.random.rand(3, 4, 4))
    field = np.random.rand(3, 5, 4, 4)
    assert np.all(mode_sorter.threshold_masks(masks, field, 0) == masks)
    assert np.all(
        mode_sorter.threshold_masks(masks, field, 1.1) == np.ones_like(masks))
