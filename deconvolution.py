import cupy as cp
from cupyx.scipy.fft import fft2, ifft2, fftshift, ifftshift


def sparse_deconvolution_ista_centered(ism_image, psf, lambda_reg=0.1, num_iterations=50):
    """
    Perform sparse deconvolution using ISTA with centered FFT.

    Args:
        ism_image (numpy.ndarray): Observed ISM image of shape (height, width, num_channels).
        psf (numpy.ndarray): Point spread function (PSF), shape (9, 19).
        lambda_reg (float): Regularization parameter controlling sparsity.
        num_iterations (int): Number of iterations for ISTA.

    Returns:
        numpy.ndarray: Reconstructed sparse image, shape (height, width, num_channels).
    """

    ism_image = cp.asarray(ism_image)
    psf = cp.asarray(psf)

    # Dimensions
    height, width, num_channels = ism_image.shape

    # Pad the PSF to match the ISM image dimensions
    psf_padded = cp.zeros((height, width), dtype=psf.dtype)
    psf_padded[height // 2 - psf.shape[0] // 2:height // 2 + psf.shape[0] // 2 + 1,
    width // 2 - psf.shape[1] // 2:width // 2 + psf.shape[1] // 2 + 1] = psf

    # Center the PSF in the spatial domain before FFT
    psf_padded_centered = fftshift(psf_padded)

    # Fourier transform of the centered PSF
    psf_ft = fft2(psf_padded_centered)
    psf_ft_conj = cp.conj(psf_ft)

    # Compute the step size for ISTA
    lipschitz_constant = cp.max(cp.abs(psf_ft) ** 2)
    step_size = 1.0 / lipschitz_constant

    # Initialize the reconstructed image
    reconstructed_image = cp.zeros_like(ism_image)

    # ISTA iterations
    for iteration in range(num_iterations):
        for channel in range(num_channels):
            # Fourier transform of the current estimate
            current_estimate_ft = fft2(fftshift(reconstructed_image[:, :, channel]))

            # Gradient of the data fidelity term
            fidelity_gradient_ft = psf_ft_conj * (
                        current_estimate_ft * psf_ft - fft2(fftshift(ism_image[:, :, channel])))
            fidelity_gradient = ifftshift(ifft2(fidelity_gradient_ft).real)

            # Update the reconstructed image with gradient descent
            updated_image = reconstructed_image[:, :, channel] - step_size * fidelity_gradient

            # Apply soft thresholding for sparsity
            reconstructed_image[:, :, channel] = cp.sign(updated_image) * cp.maximum(
                cp.abs(updated_image) - lambda_reg * step_size, 0)

        # Optionally monitor convergence (loss, PSNR, etc.)

    return cp.asnumpy(reconstructed_image)
