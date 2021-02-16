"""This module contains code to generate different modes.
"""

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
