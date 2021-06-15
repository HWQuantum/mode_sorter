"""Main module."""
try:
    import cupy as np
except ImportError as e:
    print("Couldn't find cupy, using numpy instead.")
    import numpy as np
from typing import Literal, Optional, Sequence, Union, List, Tuple


def transfer_coefficient(x: np.ndarray, y: np.ndarray,
                         wavelength: float) -> np.ndarray:
    """Given the x and y grids and the wavelength, produce the frequencies in
        the z direction

    Args:
        x: The x positions grid
        y: The y positions grid
        wavelength: The wavelength of the light

    Returns:
        np.ndarray: The transfer coefficients array
    """
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    nx, ny = x.shape

    f_x, f_y = nx / (max_x - min_x), ny / (max_y - min_y)
    vx = np.linspace(-nx / 2, nx / 2 - 1, int(nx)) / nx * f_x
    vy = np.linspace(-ny / 2, ny / 2 - 1, int(ny)) / ny * f_y
    vx, vy = np.meshgrid(vx, vy)

    return (2 * np.pi * np.sqrt(1 / wavelength**2 - vx**2 - vy**2)
            ).T  # do the transpose to have the same shape as the input x and y


def transfer_matrix(x: np.ndarray,
                    y: np.ndarray,
                    wavelength: Union[float, Sequence[float]],
                    distance: Union[float, Sequence[float]],
                    max_k: Union[float, None] = None) -> np.ndarray:
    """Generate the transfer matrix for the given ``distance``

    Args:
        x: The x positions grid
        y: The y positions grid
        wavelength: The wavelength of the light, or a list of wavelengths of light
        distance: The distance that this matrix should move the input mode. This can be a single float, or 
    a sequence of floats, in which case this generates a different transfer matrix for each plane
        max_k (float): If defined, this is the maximum k that the
            transfer matrix should support.
            Coefficients at larger k are cut off

    Returns:
        np.ndarray: The transfer matrix or matrices, depending on
            ``wavelength`` being a float or a sequence of floats
    """
    dist_array = np.array([distance]) if isinstance(
        distance, float) else np.array(distance)
    if isinstance(wavelength, Sequence):
        T = np.empty((len(dist_array), len(wavelength), *x.shape),
                     dtype=np.complex128)
        for i, w in enumerate(wavelength):
            T[:, i] = np.exp(-1j *
                             transfer_coefficient(x, y, w)[np.newaxis, :, :] *
                             dist_array[:, np.newaxis, np.newaxis])
    else:
        T = np.exp(-1j * transfer_coefficient(x, y, np.array(
            [wavelength]))[np.newaxis, :, :] *
                   dist_array[:, np.newaxis, np.newaxis])[:, np.newaxis, ...]

    if max_k is not None:
        R = np.sqrt(x**2 + y**2)
        T[..., R > (max_k * np.max(R))] = 0
    return np.asarray(np.fft.fftshift(T, axes=(-1, -2)))


def update_mask(mask: np.ndarray,
                forward_field: np.ndarray,
                backward_field: np.ndarray,
                mask_offset: float = 0,
                symmetric_mask: bool = False):
    """Update the given mask (at a single plane) with the forward and backward field at that plane
    and optionally add an offset

    The update is done in place.

    Args:
        mask: The plane to update, should have dimensions ``(X, Y)``
        forward_field: The field in the forward direction at the given plane. Should have dimensions ``(N_modes, X, Y)``
        backward_field: The field in the backward direction at the given plane. Should have dimensions ``(N_modes, X, Y)``
        mask_offset: An optional value to add to the new mask which can help promote smoother solutions.
        symmetric_mask: If true, the returned mask will be symmetric.

    Returns:
        None
    """
    d_mask = forward_field * backward_field.conj()
    norm = (np.sqrt((np.abs(forward_field)**2).sum(axis=(-1, -2)) *
                    (np.abs(backward_field)**2).sum(axis=(-1, -2))))
    d_mask[np.where(norm != 0)] /= norm[np.where(norm != 0)].reshape(
        (-1, 1, 1))
    new_m = (d_mask * np.exp(-1j * np.angle(
        (d_mask * np.exp(-1j * np.angle(mask))).sum(axis=(-1, -2))).reshape(
            (-1, 1, 1)))).sum(axis=0)
    if symmetric_mask:
        mask[:] = (new_m + new_m[::-1, :]) / 2 + mask_offset
    else:
        mask[:] = new_m + mask_offset


def propagate_field(forward_field: np.ndarray,
                    backward_field: np.ndarray,
                    masks: np.ndarray,
                    forward_transfer: np.ndarray,
                    transfer_indices: Optional[List[Tuple[int, int]]] = None,
                    should_update_mask: bool = False,
                    mask_offset: float = 0,
                    symmetric_mask: bool = False):
    """Propagates the forward and backward field through the set of masks

    This updates ``forward_field`` and ``backward_field`` in place, and optionally ``masks``.

    Args:
        forward_field: The forward moving field. Should have dimensions ``(N_planes, N_modes, X, Y)``
        backward_field: The backward moving field. Should have dimensions ``(N_planes, N_modes, X, Y)``
        masks: The set of masks. Should have dimensions ``(N_planes, X, Y)``
        forward_transfer: An array giving the free propagation of a mode through the space between mask planes.
    It'll have indices
        transfer_indices: This allows having different transfer matrices for different modes.
            The transfer matrices are zipped with this list along the first axis, and the tuples
            determine which modes of the field they apply to (taken as a slice).
            For example, ``[(None, 10), (10, None)]`` will apply the first transfer matrix
            to the slice [:10] and the second transfer matrix to the slice [10:]
        should_update_mask: If true, the masks variable is updated as the modes are propagated through.
        mask_offset: An offset that can be added to each mask to try to prevent convergence to local minima.
        symmetric_mask: If true the mask is made to be symmetric.

    Returns:
        None
    """
    assert (forward_transfer.shape[0] == 1) or (
        forward_transfer.shape[0] == masks.shape[0] -
        1), "Number of transfer matrices should be 1 less than number of masks"
    backward_transfer = forward_transfer.conj()

    # iterate forwards
    for i in range(len(masks) - 1):
        transfer_mask_index = 0 if forward_transfer.shape[0] == 1 else i
        if should_update_mask:
            update_mask(masks[i],
                        forward_field[i],
                        backward_field[i],
                        mask_offset,
                        symmetric_mask=symmetric_mask)
        if transfer_indices is None:
            forward_field[i + 1] = np.fft.ifftn(
                forward_transfer[transfer_mask_index, 0] * np.fft.fftn(
                    forward_field[i] * np.exp(-1j * np.angle(masks[i])),
                    axes=(1, 2)),
                axes=(1, 2))
        else:
            for (mi, ma), t in zip(transfer_indices,
                                   forward_transfer[transfer_mask_index]):
                forward_field[i + 1, mi:ma] = np.fft.ifftn(t * np.fft.fftn(
                    forward_field[i, mi:ma] * np.exp(-1j * np.angle(masks[i])),
                    axes=(1, 2)),
                                                           axes=(1, 2))

    # now iterate backwards
    for i in range(len(masks) - 1, 0, -1):
        transfer_mask_index = 0 if backward_transfer.shape[0] == 1 else (i - 1)
        if should_update_mask:
            update_mask(masks[i],
                        forward_field[i],
                        backward_field[i],
                        mask_offset,
                        symmetric_mask=symmetric_mask)
        if transfer_indices is None:
            backward_field[i - 1] = np.fft.ifftn(
                backward_transfer[transfer_mask_index, 0] * np.fft.fftn(
                    backward_field[i] * np.exp(1j * np.angle(masks[i])),
                    axes=(1, 2)),
                axes=(1, 2))
        else:
            for (mi, ma), t in zip(transfer_indices,
                                   backward_transfer[transfer_mask_index]):
                backward_field[i - 1, mi:ma] = np.fft.ifftn(t * np.fft.fftn(
                    backward_field[i, mi:ma] * np.exp(1j * np.angle(masks[i])),
                    axes=(1, 2)),
                                                            axes=(1, 2))


def threshold_masks(
        masks: np.ndarray,
        field: np.ndarray,
        threshold_value: float,
        threshold_type: Literal["maximum",
                                "average"] = "maximum") -> np.ndarray:
    """Take in a set of masks, and the field over the masks and return the masks thresholded by
    the field intensity.

    Args:
    masks: The masks to threshold
    field: The field over all of these masks
    threshold_value: The fraction of maximum intensity at a given mask that we should threshold at.

    Returns:
    the modified masks
    """
    intensity = np.abs(field)**2
    new_masks = masks.copy()
    if threshold_type == "maximum":
        field_threshold: np.ndarray = np.average(intensity, axis=1)
    elif threshold_type == "average":
        field_threshold: np.ndarray = np.max(intensity, axis=1)
    else:
        raise ValueError(
            "threshold_type should be one of [\"maximum\", \"average\"]")
    new_masks[np.where(
        field_threshold < (np.max(field_threshold, axis=(1, 2)) *
                           threshold_value)[:, np.newaxis, np.newaxis])] = 1
    return new_masks
