import numpy as np

CROP_SIZE = (9, 19)
RED_X_LOCATION = int(CROP_SIZE[1] - 4)
RED_Y_LOCATION = np.ceil(CROP_SIZE[0] / 2).astype(int)

def rescale_float32_to_uint16(arr):
    """
    Rescales a NumPy or CuPy array of type float32 to uint16.
    Args:
        arr (ndarray): Input array of type float32.
    Returns:
        ndarray: Rescaled array of type uint16.
    """
    min_val = arr.min()
    max_val = arr.max()

    if max_val == min_val:
        # Handle case where all values are the same
        return np.zeros_like(arr, dtype=np.uint16)

    # Normalize to [0, 1] and scale to [0, 65535]
    rescaled = (arr - min_val) / (max_val - min_val) * 65535
    rescaled = np.clip(rescaled, 0, 65535)  # Ensure values are within range
    return rescaled.astype(np.uint16)


