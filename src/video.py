import numpy as np
import cv2 as cv

from src.colour import okabe_ito

def highlight_blob(frame:np.ndarray, ids:np.ndarray, mask: np.ndarray, alpha:float=0.4) -> np.ndarray:
    """
    Highlights the blobs in the original frame by id

    Params:
        frame: The original grayscale frame
        ids: The id of each pixel
        mask: a binary mask(0, 255) of where the droplets are

    Returns:
        The original frame with the droplets highlighted in different colours
    """

    # blend overlay with original source
    frame_rgb = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

    # Create colours map based on id    
    colours = np.array([x.rgb() for x in okabe_ito]).astype(np.uint8)
    lbl_clr = colours[ids % len(colours)]

    # create blending mask
    blend_mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    
    blend = cv.addWeighted(frame_rgb, 1-alpha, lbl_clr, alpha, 0)

    result = np.where(blend_mask == 255, blend, frame_rgb)
    
    return result

def draw_centroids(frame: np.ndarray, centroids: np.ndarray, labels) -> np.ndarray:
    """
    Draw all the droplet centroids

    Params:
        frame: the original frame on which to draw the centroids
        centroids: a dictionary of the centroid locations
        labels: the currenlty active droplets

    Returns:
        The original frame with added centroid dots
    """
    for j, lbl in enumerate(labels):
        if j == 0:
            continue
        c = centroids[lbl]
        frame = cv.circle(
            frame, (int(c[0]), int(c[1])), 5, (j * 70, 0, j * 70), -1
        )

    return frame

    

