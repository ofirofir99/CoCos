import numpy as np

from utils import RED_Y_LOCATION, RED_X_LOCATION, CROP_SIZE, rescale_float32_to_uint16


def parallel_ols(target_crops, basis_images):
    """
    Compute OLS coefficients for multiple target crops in parallel.

    Args:
        target_crops (numpy.ndarray): Array of target crops, shape (num_crops, 9, 19).
        basis_images (numpy.ndarray): Array of basis images, shape (num_basis, 9, 19).

    Returns:
        numpy.ndarray: Coefficients for each crop, shape (num_crops, num_basis).
    """
    num_crops, _, _ = target_crops.shape
    num_basis, _, _ = basis_images.shape

    # Flatten the target crops and basis images
    y = target_crops.reshape(num_crops, -1)  # Shape: (num_crops, 9*19)
    x = basis_images.reshape(num_basis, -1).T  # Shape: (9*19, num_basis)

    # Compute the OLS solution
    xt_x_inv = np.linalg.inv(x.T @ x)  # Shape: (num_basis, num_basis)
    xt_x_inv_xt = xt_x_inv @ x.T  # Shape: (num_basis, 9*19)
    b = y @ xt_x_inv_xt.T  # Shape: (num_crops, num_basis)

    return b


def reconstruct_rgb_crops_ols(crops, basis_images):
    num_basis, height, width = basis_images.shape
    num_crops = crops.shape[0]
    _num_ch = basis_images.shape[0] // 3
    norm_factor = np.max(crops, axis=(1, 2), keepdims=True)
    norm_crops = crops / norm_factor
    # start_time=time.time()
    betas = parallel_ols(norm_crops, basis_images)
    # end_time=time.time()

    basis_images_flat = basis_images.reshape(num_basis, -1)  # Shape: (num_basis, 9*19)
    basis_red_flat = basis_images_flat[::_num_ch]  # Shape: (num_basis/num_ch, 9*19)
    # Compute reconstructed crops
    reconstructed_flat = betas @ basis_images_flat  # Shape: (num_crops, 9*19)
    reconstructed_crops = np.clip(
        reconstructed_flat.reshape(num_crops, height, width), a_min=0, a_max=None
    )  # Shape: (num_crops, 9, 19)
    # Compute reconstructed RGB crops using only red basis images
    reconstructed_rgb_crops = np.zeros(
        (crops.shape[0], crops.shape[1], crops.shape[2], _num_ch)
    )
    for ch in range(_num_ch):
        reconstructed_rgb_flat = (
            betas[:, ch::_num_ch] @ basis_red_flat
        )  # Shape: (num_crops, 9*19)
        reconstructed_rgb_crops[..., ch] = np.clip(
            reconstructed_rgb_flat.reshape(num_crops, height, width),
            a_min=0,
            a_max=None,
        )

    # Reshape back to original crop shape
    reconstructed_crops *= norm_factor
    reconstructed_rgb_crops *= norm_factor[..., None]

    return reconstructed_crops, reconstructed_rgb_crops


def compose_im_from_rgb_crops(rgb_crops, rounded_peaks, im_size=(512, 512)):
    num_channels = rgb_crops.shape[-1]
    # Prepare output array
    composed_image = np.zeros((im_size[0], im_size[1], num_channels))
    # Prepare indexing arrays

    # Precompute all crops at once
    rows = (
        rounded_peaks[:, 1, None, None, None]
        + np.arange(RED_Y_LOCATION - CROP_SIZE[0], RED_Y_LOCATION)[None, :, None, None]
    )
    cols = (
        rounded_peaks[:, 0, None, None, None]
        + np.arange(-RED_X_LOCATION, CROP_SIZE[1] - RED_X_LOCATION)[None, None, :, None]
    )
    channels = (
        np.zeros((rounded_peaks.shape[0], 1, 1, 1))
        + np.arange(num_channels)[None, None, None, :]
    )

    rows, cols, channels = rows.astype(int), cols.astype(int), channels.astype(int)
    np.add.at(composed_image, (rows, cols, channels), rgb_crops)
    # for i in range(0, len(all_ch_crops),scatter_add_batch_size): batch_idx = slice(i,
    # min(i + scatter_add_batch_size, len(all_ch_crops))) cupyx.scatter_add(composed_image, (rows[batch_idx],
    # cols[batch_idx], channels[batch_idx]), cp.asarray(all_ch_crops[batch_idx])) #adding many pics at once,
    # on gpu - way faster than looping.
    return rescale_float32_to_uint16(composed_image)
