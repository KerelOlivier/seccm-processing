import numpy as np
import cv2 as cv
import scipy.signal as sps

from typing import Tuple


def get_intensity_peaks(frame:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds the peaks and their width of the intensity values

    Params:
        frame: the grayscale frame

    Returns:
        A list of tuples(peak, width) with the peak center and peak width
    """
    
    # Calculate the histogram of intensity values
    hist = cv.calcHist([frame], [0], None, [256], [0,256]).flatten()
    hist = np.convolve(hist, np.ones(11)/11, mode='same')
    
    # find the peaks and their widths
    pks, stats = sps.find_peaks(hist, prominence=200, distance=10, width=10)

    widths = np.array(stats["widths"])
    peaks = np.array(pks)
    
    return peaks, widths, hist

