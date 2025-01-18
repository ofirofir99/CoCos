from enum import Enum

from tifffile import tifffile
import scipy.io as sio

class PsfSource(Enum):
    IMAGE_STACK = "IMAGE_STACK"
    MATLAB = "MATLAB"

def import_external_ref_psfs(external_ref_psfs_stack_path, external_ref_psfs_mat_path,psf_source:PsfSource):
    if psf_source == PsfSource.IMAGE_STACK:
        psfs = tifffile.TiffFile(external_ref_psfs_stack_path).asarray()
    elif psf_source == PsfSource.MATLAB:
        psfs = sio.loadmat(external_ref_psfs_mat_path)
    else:
        raise ValueError(f"Invalid psf_source: {psf_source}")
    return psfs

