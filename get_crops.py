from functools import reduce

import numpy as np

CROP_SIZE = (9, 19)
RED_X_LOCATION = int(CROP_SIZE[1] - 4)
RED_Y_LOCATION = np.ceil(CROP_SIZE[0] / 2).astype(int)


def get_crops(im, centers, median_calc=True):
    centers = np.round(np.array(centers)).astype(int)

    # Precompute all crops at once
    rows = centers[:, 1, None, None] + np.arange(RED_Y_LOCATION - CROP_SIZE[0], RED_Y_LOCATION)[None, :, None]
    cols = centers[:, 0, None, None] + np.arange(-RED_X_LOCATION, CROP_SIZE[1] - RED_X_LOCATION)[None, None, :]
    crops = im[rows, cols]

    # Compute border median for all crops at once
    if median_calc:
        border = np.concatenate([crops[:, 0, :], crops[:, -1, :],
                                 crops[:, :, 0], crops[:, :, -1]], axis=1)
        border_median = np.median(border, axis=1)[:, None, None]
        # Apply operations to all crops simultaneously
        crops = crops - border_median
        np.clip(crops, 0, None, out=crops)
    overall_mean = np.mean(crops)
    # Calculate standard deviation for each crop
    crops_std = np.std(crops, axis=(1, 2))
    # Calculate mean of standard deviations
    overall_std = np.mean(crops_std)
    # Calculate max of each crop
    crops_max = np.max(crops, axis=(1, 2))
    return crops, overall_mean, overall_std, crops_max


def get_crops_all_pictures(stack, crop_locations, median_calc=True):
    """
    Get crops from the stack of images according to the crop_locations.
    """
    total_number_of_peaks = reduce(lambda count, l: count + len(l), crop_locations, 0)
    all_crops = np.zeros((total_number_of_peaks, CROP_SIZE[0], CROP_SIZE[1]))
    overall_mean_array = np.zeros(len(stack))
    overall_std_array = np.zeros(len(stack))
    all_crops_max = np.zeros(total_number_of_peaks)

    current_peak = 0
    for i, image in enumerate(stack):
        centers = crop_locations[i]
        number_of_peaks = len(centers)
        crops, overall_mean, overall_std, crops_max = get_crops(image, centers, median_calc)
        all_crops[current_peak:current_peak + number_of_peaks, :, :] = crops
        all_crops_max[current_peak:current_peak + number_of_peaks] = crops_max
        overall_mean_array[i] = overall_mean
        overall_std_array[i] = overall_std
        current_peak += number_of_peaks
    return all_crops, all_crops_max, np.mean(overall_mean_array), np.mean(overall_std_array)