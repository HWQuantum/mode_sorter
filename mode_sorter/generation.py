"""This module contains code to generate different modes.
"""
import numpy as np
from numpy import pi, sqrt, arctan, exp
from scipy.special import assoc_laguerre, eval_hermite


def state_representation(states: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Given a set of state coefficients and a basis to represent them in, return the representation.
    states should be an (m, n) array where m is the number of states and n is the number of basis elements.
    basis should be a (n, x, y) array where n is the number of basis elements and x and y are the coordinates.
    returns an array (m, x, y) where m indexes the different states.

    The basis and states should be normalised, eg:
        (states * states.conj()).sum(axis=1) = [1, 1, 1, 1, ...]
        (basis * basis.conj()).sum(axis=(-1, -2)) = [1, 1, 1, 1, ...]
    Args:
        states: An array containing the different states to represent in the given ``basis``
        basis: An array containing the different basis states

    Returns:
        np.ndarray: An array of the states represented in the given basis
    """
    return (states.reshape((*states.shape, 1, 1)) * basis.reshape(
        (1, *basis.shape))).sum(axis=1)


def generate_spot(centre, diameter, wavelength, distance_to_plane, X, Y):
    """Generate a spot at a certain position
    use the equation from wikipedia:
    https://en.wikipedia.org/wiki/Gaussian_beam

    Args:
        centre: (x, y)
        diameter: the mode diameter
        wavelength: the wavelength of light
        distance_to_plane: the distance to the first plane from the point the spots are generated
        X, Y: 2d arrays of the X- and Y-coordinates

    Returns:
        np.ndarray: An array of the spot represented over the given coordinates
    """

    # set up all the terms required for the equation
    R2 = (X - centre[0])**2 + (Y - centre[1])**2
    k = 2 * pi / wavelength
    w0 = diameter / 2
    z = distance_to_plane
    zr = w0**2 * pi / wavelength
    wz = w0 * sqrt(1 + (z / zr)**2)
    if z != 0:
        Rz_inv = 1 / (z * (1 + (zr / z)**2))
    else:
        Rz_inv = 0
    psi = arctan(z / zr)
    spot = (w0 / wz) * exp(-R2 / wz**2) * exp(
        -1j * (k * z + k * R2 * Rz_inv / 2 - psi))
    norm = sqrt((np.abs(spot)**2).sum())
    if norm > 0:
        spot /= norm
    return spot


def generate_HG(l, m, centre, diameter, wavelength, distance_to_plane, X, Y):
    """Generate a normalised HG mode
    """
    R2 = (X - centre[0])**2 + (Y - centre[1])**2
    k = 2 * pi / wavelength
    w0 = diameter / 2
    z = distance_to_plane
    zr = w0**2 * pi / wavelength
    wz = w0 * sqrt(1 + (z / zr)**2)
    if z != 0:
        Rz_inv = 1 / (z * (1 + (zr / z)**2))
    else:
        Rz_inv = 0
    psi = (l + m + 1) * arctan(z / zr)
    hg_mode = (w0 / wz * eval_hermite(l,
                                      sqrt(2) * (X - centre[0]) / wz) *
               eval_hermite(m,
                            sqrt(2) * (Y - centre[1]) / wz) * exp(-R2 / wz**2)
               * exp(-1j * k * R2 * Rz_inv / 2)
               * exp(1j * psi) * exp(-1j * k * z))
    norm = sqrt((np.abs(hg_mode)**2).sum())
    if norm != 0:
        hg_mode /= norm
    return hg_mode


def generate_LG(p, l, centre, diameter, wavelength, distance_to_plane, X, Y):
    """Generate a normalised LG mode
    """
    l_abs = np.abs(l)
    R2 = (X - centre[0])**2 + (Y - centre[1])**2
    R = np.sqrt(R2)
    k = 2 * pi / wavelength
    w0 = diameter / 2
    z = distance_to_plane
    zr = w0**2 * pi / wavelength
    wz = w0 * sqrt(1 + (z / zr)**2)
    if z != 0:
        Rz_inv = 1 / (z * (1 + (zr / z)**2))
    else:
        Rz_inv = 0
    psi = arctan(z / zr) * (l_abs + 2 * p + 1)
    lg_mode = w0 / wz * np.power(R * sqrt(2) / wz, l_abs) * exp(
        -R2 / wz**2) * assoc_laguerre(2 * R2 / wz**2, p, l_abs) * exp(
            -1j * k * R2 * Rz_inv / 2) * np.exp(-1j * l * np.arctan2(
                Y - centre[1], X - centre[0])) * np.exp(1j * psi)
    norm = sqrt((np.abs(lg_mode)**2).sum())
    if norm != 0:
        lg_mode /= norm
    return lg_mode


def generate_circular_spots(mode_count,
                            radius,
                            wavelength,
                            distance_to_plane,
                            spot_diameter,
                            X,
                            Y,
                            centre=(0, 0)):
    """Generate a set of spots in a circle with a given radius
    """
    p = np.exp(2 * np.pi * 1j * np.linspace(0, 1, mode_count + 1)[:-1])
    spot_x = np.cos(np.angle(p)) * radius
    spot_y = np.sin(np.angle(p)) * radius
    field = np.zeros((mode_count, *X.shape), dtype=np.complex128)
    for i, pos in enumerate(zip(spot_x, spot_y)):
        field[i, :, :] = generate_spot(pos, spot_diameter, wavelength,
                                       distance_to_plane, X - centre[0], Y - centre[1])
    return field


def generate_triangular_spots(rows,
                              mode_diameter,
                              distance_to_tip,
                              centre,
                              wavelength,
                              distance_to_plane,
                              X,
                              Y,
                              rotation=0):
    """Generate a set of spots in a triangle (for use with HG modes)
    give it the number of rows and it'll return a set of modes of row(row+1)/2 dimensions

    mode diameter is the diameter of each individual mode

    distance_to_tip is the distance from the centre of the triangle to the tip and the back

    rotation is the rotation of the triangle in radians. It goes anti-clockwise from the x-axis

    centre decides the position of the centre of the triangle. It rotates around this centre.
    """
    if rows > 1:
        row_separation = 4 * distance_to_tip / (rows - 1)
    else:
        row_separation = 0

    x_y_sep = sqrt((row_separation**2) / 2)
    positions = [
        x_y_sep * ((col + 1j * (row - col)) -
                   ((rows - 1) / 4 + 1j *
                    (rows - 1) / 4)) * np.exp(1j *
                                              (3 * np.pi / 4 + rotation)) +
        (centre[0] + 1j * centre[1]) for row in range(rows)
        for col in range(row + 1)
    ]
    n_modes = (rows * (rows + 1)) // 2
    modes = np.empty((n_modes, *X.shape), dtype=np.complex128)
    for i, p in enumerate(positions):
        modes[i] = generate_spot((p.real, p.imag), mode_diameter, wavelength,
                                 distance_to_plane, X, Y)
    return modes
