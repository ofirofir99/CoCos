import os
from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import cupyx
import numpy as np
import pandas as pd
import pims
import scipy.io as sio
import trackpy as tp
from scipy.io import loadmat
from tifffile import tifffile

from get_crops import RED_Y_LOCATION, RED_X_LOCATION, CROP_SIZE


def process_frame(frame):
    """
    Run trackpy locate, can add preprocessing if wanted.
    """
    return tp.locate(frame, diameter=7, minmass=100)


def find_peaks_using_trackpy(frames_numpy, border_exclude_size):
    """
    Find peaks in a stack of images.
    """
    with ThreadPoolExecutor() as executor:  # multithreading for speed
        futures = []
        for i, frame in enumerate(frames_numpy):
            futures.append(executor.submit(process_frame, frame))

    results = [f.result() for f in futures]
    peaks = []
    for features in results:
        features['x'] += border_exclude_size  # We give a cropped image to exclude borders, so shift back the locations.
        features['y'] += border_exclude_size
        p_x = features['x'].values
        p_y = features['y'].values
        peaks.append(np.column_stack((p_x, p_y)))
    return peaks


def new_analyze_ref_stack(stack_path, border_exclude_size=13):
    """
    Run peak finding on a stack of images without borders.
    """
    frames = pims.open(stack_path)
    frames_numpy = np.array(
        [frame[border_exclude_size:-border_exclude_size, border_exclude_size:-border_exclude_size] for frame in frames])
    peak_locations = find_peaks_using_trackpy(frames_numpy, border_exclude_size)
    return peak_locations


def estimate_noise_variance_gpu(crops, stack_std_val):
    """
    Estimate noise variance in the data, combining Poisson and Gaussian components.
    Args:
        data: 3D numpy array (noisy data - not normalized).
    Returns:
        Estimated noise variance.
    """

    sigma = 0.8
    data = cp.asarray(crops, dtype=cp.float32)
    dims = data.ndim

    # Poisson noise: mean signal level
    poisson_var = cp.mean(data, axis=(-2, -1))

    # # # Gaussian noise: residual variance after smoothing
    if dims == 3:
        border_mask = cp.ones_like(data[0, ...], dtype=bool)
    else:
        border_mask = cp.ones_like(data, dtype=bool)
    #
    border_mask[1:-1, 1:-1] = False
    smoothedX = cupyx.scipy.ndimage.gaussian_filter1d(data, sigma=sigma, axis=-2)  # Adjust sigma if needed
    smoothed = cupyx.scipy.ndimage.gaussian_filter1d(smoothedX, sigma=sigma, axis=-1)  # Adjust sigma if needed

    if dims == 3:
        border_mean = cp.mean(data[:, border_mask], axis=1)
        data_mean = cp.mean(data, axis=(-2, -1))
        bad_ind = cp.where(border_mean > data_mean)[0]
        residuals = data[:, border_mask] - smoothed[:, border_mask]
    else:
        residuals = data[border_mask] - smoothed[border_mask]

    # residuals=data-smoothed
    gaussian_var = cp.var(residuals, axis=(-2, -1))
    snr = cp.max(smoothed, axis=(-2, -1)) / (poisson_var + gaussian_var)
    snr[bad_ind] = 0.0

    # Total variance is the sum of both components
    return (poisson_var + gaussian_var ** 2).get(), snr.get()


def average_crop_one_picture(Im, centers):
    """
    Average all crops in a picture (reference)
    Do all operations with numpy (without loops) to ensure efficiency
    """
    centers = np.array(centers).astype(int)

    # Precompute all crops at once
    rows = centers[:, 1, None, None] + np.arange(RED_Y_LOCATION - CROP_SIZE[0], RED_Y_LOCATION)[None, :, None]
    cols = centers[:, 0, None, None] + np.arange(-RED_X_LOCATION, CROP_SIZE[1] - RED_X_LOCATION)[None, None, :]

    crops = Im[rows, cols]

    # Compute border median for all crops at once
    border = np.concatenate([crops[:, 0, :], crops[:, -1, :],
                             crops[:, :, 0], crops[:, :, -1]], axis=1)
    border_median = np.median(border, axis=1)[:, None, None]

    # Apply operations to all crops simultaneously
    crops = crops - border_median
    np.clip(crops, 0, None, out=crops)
    crops /= np.max(crops, axis=(1, 2))[:, None, None]

    # Compute average
    average_array = np.mean(crops, axis=0)

    return average_array


def average_crop_all_pictures(spectral_ref_stack_paths, peaks):
    """
    Average all crops for each color, on all images (reference).
    """

    num_channels = len(spectral_ref_stack_paths)
    average_crop_all_pictures_all_colors = np.zeros((CROP_SIZE[0], CROP_SIZE[1], num_channels))
    for i, color in enumerate(spectral_ref_stack_paths):
        with tifffile.TiffFile(color) as tif:
            num_images = len(tif.pages)
            stack = tif.asarray()  # Load all images at once for efficiency

        average_crop_all_pictures_one_color = np.zeros(CROP_SIZE)
        for j in range(num_images):
            average_crop_all_pictures_one_color += average_crop_one_picture(stack[j], np.round(
                peaks[j]))  #perhaps can be improved, but not critical.
        average_crop_all_pictures_one_color = average_crop_all_pictures_one_color / num_images
        average_crop_all_pictures_all_colors[:, :, i] = average_crop_all_pictures_one_color

    return average_crop_all_pictures_all_colors


def import_external_ref_psfs(external_ref_psfs_stack_path, external_ref_psfs_mat_path, external_ref_psfs_names_path):
    stack = tifffile.TiffFile(external_ref_psfs_stack_path).asarray()
    psfs_mat = sio.loadmat(external_ref_psfs_mat_path)
    psfs_names = sio.loadmat(external_ref_psfs_names_path)

    return stack, psfs_mat, psfs_names


def simulate_psfs(crop_size=(9, 19), red_shift=4, sigma=1.25):
    def load_spectrum_files(base_folder, pattern):
        """Load spectrum files matching a given pattern."""
        spectrum_files = []
        for root, _, files in os.walk(base_folder):
            for file in files:
                if file.endswith(pattern):
                    spectrum_files.append(os.path.join(root, file))
        return spectrum_files

    def read_spectrum(file_path):
        """Read a spectrum file into a NumPy array using pandas to handle complex formatting."""
        try:
            df = pd.read_csv(file_path, header=None, comment="%", delimiter='\t')
            return df.values
        except Exception as e:
            raise ValueError(f"Error reading spectrum file {file_path}: {e}")

    def gaussian_2d(x, y, x0, y0, sigma):
        """Generate a 2D Gaussian distribution."""
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def simulate_psf(spectral_data, poly3_coeffs, poly1_coeffs, filter_data, camera_eff_curve, crop_size, red_shift,
                     sigma=1.25):
        """Simulate PSFs based on spectral data, polynomial fits, and camera efficiency."""
        psf_image = np.zeros(crop_size)
        x = np.arange(crop_size[1])
        y = np.arange(crop_size[0])
        xv, yv = np.meshgrid(x, y)

        for wl, intensity in spectral_data:
            # Evaluate polynomial fits
            x_shift = np.polyval(poly3_coeffs, wl)
            y_shift = np.polyval(poly1_coeffs, x_shift)

            # Interpolate filter transmission and camera efficiency
            filter_transmission = np.interp(wl, filter_data[:, 0], filter_data[:, 1])
            camera_efficiency = np.interp(wl, camera_eff_curve[:, 0], camera_eff_curve[:, 1])

            if filter_transmission > 0 and camera_efficiency > 0:  # Process only valid wavelengths
                x0 = -(red_shift - 4) + x_shift
                y0 = y_shift - 1

                gaussian = gaussian_2d(xv, yv, x0, y0, sigma)
                psf_image += gaussian * intensity * filter_transmission * camera_efficiency

        return psf_image

    def process_all_spectral_files(base_folder, poly3_coeffs, poly1_coeffs, filter_data, camera_eff_curve, crop_size,
                                   red_shift=4, sigma=1.25, wl_lim=(400, 900)):
        """Process all spectral files and calculate spectral PSFs."""
        em_files = load_spectrum_files(base_folder, 'Em.txt')
        psf_results = {}
        max_emission_wl = []
        for file_path in em_files:
            spectral_data = read_spectrum(file_path)

            # Filter spectral data to within wavelength limits
            spectral_data = spectral_data[(spectral_data[:, 0] >= wl_lim[0]) & (spectral_data[:, 0] <= wl_lim[1])]
            max_emission_wl.append(np.where(spectral_data[:, 1] == max(spectral_data[:, 1]))[0])
            psf_image = simulate_psf(spectral_data, poly3_coeffs, poly1_coeffs, filter_data, camera_eff_curve,
                                     crop_size, red_shift, sigma)
            fluorophore_name = os.path.splitext(os.path.basename(file_path))[0].replace(" - Em", "")
            fluorophore_name = fluorophore_name.replace("FocalCheck ", "FC ")
            fluorophore_name = fluorophore_name.replace(" Ring", "")
            fluorophore_name = fluorophore_name.replace("Double", "")

            psf_results[fluorophore_name] = psf_image

            # # Plot and save the PSF for each file
            # plt.imshow(psf_image, cmap="hot", interpolation="nearest")
            # plt.colorbar()
            # plt.title(f"Simulated PSF: {fluorophore_name}")
            # plt.xlabel("X (pixels)")
            # plt.ylabel("Y (pixels)")
            # plt.savefig(os.path.join(os.path.dirname(file_path), f"{fluorophore_name}_PSF.png"))
            # plt.close()

        return psf_results, np.concatenate(max_emission_wl)

    # File paths (update these paths as necessary)
    base_folder = "/DATA2/Data_CoCoS_HD/COCOS_ISM/Matlab files"  # Base folder containing EM.txt files
    filter_file = "/DATA2/Data_CoCoS_HD/COCOS_ISM/Matlab files/FF01-440_521_607_700.txt"
    mat_file_processed = "/DATA2/Data_CoCoS_HD/COCOS_ISM/Matlab files/DispersionFitsProcessed.mat"  # MATLAB file containing polynomial coefficients

    # Load data
    filter_data = read_spectrum(filter_file)
    processed_data = loadmat(mat_file_processed)

    # Extract polynomial coefficients for dispersion fits
    poly3_coeffs = processed_data['coeffs_PixToWl'].flatten()
    poly1_coeffs = processed_data['coeffs_Y'].flatten()

    # Define camera efficiency curve
    camera_eff_vals = np.array([
        [400, 0.7],
        [450, 0.85],
        [500, 0.88],
        [600, 0.9],
        [650, 0.92],
        [700, 0.9],
        [750, 0.85],
        [800, 0.75],
        [850, 0.6],
        [900, 0.43]
    ])

    # Simulate PSFs for all spectral files
    # crop_size = (9, 19)
    psf_results, max_emission_wl = process_all_spectral_files(base_folder, poly3_coeffs, poly1_coeffs, filter_data,
                                                              camera_eff_vals, crop_size, red_shift=red_shift,
                                                              sigma=sigma)
    psfs = np.asarray(list(psf_results.values()))
    fl_names = list(psf_results.keys())
    ind_sorted = np.argsort(max_emission_wl, )
    psfs = psfs[ind_sorted[::-1]]
    fl_names[:] = [fl_names[ind] for ind in ind_sorted[::-1]]
    return psfs.transpose(1, 2, 0), fl_names
