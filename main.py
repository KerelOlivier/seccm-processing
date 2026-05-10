import scipy.signal as sps
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sys
import os
import logging
from tqdm import tqdm
from dataclasses import dataclass, asdict
import argparse

from src.colour import okabe_ito
from src.video import *
from src.frame_processing import *

@dataclass
class Measurement:
    id: int = 0
    frame: int = 0
    area: float = 0.0
    intensity: float = 0.0

def getAreas(labels):
    """
    Calculates the area of a connected component in pixels
    """

    values, counts = np.unique(labels, return_counts=True)
    return dict(zip(values, counts))

def getIntensity(labels, source):
    """
    Calculates the average value of the area inside the connected component
    """

    values, count = np.unique(labels, return_counts=True)
    res = dict()

    for label, area in zip(values, count):
        intensity = np.sum(source[labels == label]) / area
        res[int(label)] = intensity

    return res

def analyseVideo(path, out_dir):
    """
    Analyses the video frames

    Params:
        path: The path to the video source (must be an MJPG)
        out_dir: The path to the video output directory
    """
    logger = logging.getLogger("analyse video")
    # Load video
    src_path = sys.argv[1]
    cap = cv.VideoCapture(src_path)

    # Get video stats
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    name = os.path.basename(path)

    logger.info(
        f"Loaded video:\n"
        f"\tname: {name}\n"
        f"\tsize: {width}x{height}\n"
        f"\tfps: {fps}\n"
        f"\tnumber of frames {frame_count}"
    )

    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    out = cv.VideoWriter(f"{out_dir}/highlighted.avi", fourcc, fps, (width, height))
    
    frames = np.array([cap.read()[1] for i in range(frame_count)])
    gray = [cv.cvtColor(frame, cv.COLOR_BGR2GRAY) for frame in frames]

    window = 7

    histogram = []

    measurements = []

    # Centroid tracking variables
    cid_cnt = 1
    centroid_ids = dict()

    # Tracks at what frame each id is made
    id2start = dict()

    pks_cnt = []


    for i in tqdm(range(window//2, frame_count - window//2)):
        frame = gray[i]
        m = np.median(gray[i - window//2:i+window//2], axis=0).astype(np.uint8)

        denoised = cv.bilateralFilter(m, 9, 75, 75)

        pks, widths, hist = get_intensity_peaks(denoised)

        histogram.append(hist)

        # Create the mask
        mask = np.zeros_like(frame)
        if len(pks) >= 2: # At least one droplet present

            mask = cv.threshold(denoised, 0, 255,
                   cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
            mask[denoised < pks[0] + widths[0]] = 0

        # Labeling of the blobs
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
            mask
        )
        
        # remove all labels that are not fully in frame and set them to 0
        for lbl in range(num_labels):
            x = stats[lbl, cv.CC_STAT_LEFT]
            y = stats[lbl, cv.CC_STAT_TOP]
            w = stats[lbl, cv.CC_STAT_WIDTH]
            h = stats[lbl, cv.CC_STAT_HEIGHT]
            area = stats[lbl, cv.CC_STAT_AREA]
            if area < 3000:
                labels[labels == lbl] = 0

        # Create a binary mask, with values 0 and 255
        mask = np.zeros_like(mask)
        mask[labels > 0] = 255

        # Calculate stats

        areas = getAreas(labels)
        intens = getIntensity(labels, denoised)

        # track centroids
        lbls = np.unique(labels)  # Unique labels
        ids = np.zeros_like(labels)

        new_cids = dict()

        for lbl in lbls:
            if lbl == 0:
                continue
            # Check if it is close enough to an already existing centroid
            mx_delta = 20
            found = False
            for id in centroid_ids.keys():
                norm = np.linalg.norm(centroids[lbl] - centroid_ids[id])
                if norm <= mx_delta:
                    # Close enough, update centroid location
                    new_cids[id] = centroids[lbl]
                    ids[labels == lbl] = id
                    found = True

                    # Add statistics
                    measurements.append(Measurement(id, i, areas[lbl], intens[lbl]))

                    break

            if not found:
                # A new centroid has been found
                new_cids[cid_cnt] = centroids[lbl]
                ids[labels == lbl] = cid_cnt
                measurements.append(Measurement(cid_cnt, i, areas[lbl], intens[lbl]))
                id2start[cid_cnt] = i
                cid_cnt += 1
        

        centroid_ids = new_cids

        # Video
        output = highlight_blob(frame, ids, mask)
        output = draw_centroids(output, centroids, lbls)

        out.write(output)

    cap.release()
    out.release()

    # Plot the peaks
    fig, ax = plt.subplots(1, 1);
    
    ax.plot(np.arange(len(pks_cnt)), pks_cnt)

    fig.savefig(f"{out_dir}/peaks.png", dpi=300)


    # plot histogram
    histogram = np.array(histogram)
    fig, ax = plt.subplots(1, 1)

    im = ax.imshow(
        histogram.T,
        aspect = 'auto',
        origin = 'lower',
        cmap = 'viridis'
    )


    x_tick_pos = np.arange(0, frame_count, 30*fps) 
    x_tick_lbl = (np.arange(0, frame_count, 30*fps)//fps).astype(np.uint32)

    ax.set_xticks(x_tick_pos)
    ax.set_xticklabels(x_tick_lbl)

    ax.set_title("Intensity value frequency over time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("intensity")
    fig.colorbar(im, ax=ax)
    fig.savefig(f"{out_dir}/histogram.png", dpi=300)


    logger.info("Saved video")

    return measurements

def measurements2dataframe(measurements):
    df = pl.DataFrame([asdict(r) for r in measurements])

    return df

def processMeasurements(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(time=pl.col("frame") / 20)

    # Moving averages
    df = df.sort("time").with_columns(
        area_mean=pl.col("area").rolling_mean(5, center=True, min_samples=1).over("id"),
        intensity_mean=pl.col("intensity")
        .rolling_mean(5, center=True, min_samples=1)
        .over("id"),
    )

    return df

def plotMeasurements(df, cpts, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    min_area = df["area_mean"].min()
    max_area = df["area_mean"].max()

    min_intense = df["intensity_mean"].min()
    max_intense = df["intensity_mean"].max()

    for lbl in df["id"].unique():

        lbl_color = okabe_ito[lbl%len(okabe_ito)].hex()
        df_sub = df.filter(pl.col("id") == lbl)


        legend_label = f"droplet: {lbl}"
        # Area
        axes[0].set_title("droplet area")
        axes[0].set_xlabel("time (s)")
        axes[0].set_ylabel("area (pixels)")
        axes[0].plot(df_sub["time"], df_sub["area"], color=lbl_color, alpha=0.25)
        axes[0].plot(
            df_sub["time"], df_sub["area_mean"], color=lbl_color, label=legend_label
        )

        if len(cpts) > 0:
            axes[0].vlines(
                cpts.filter((pl.col("id") == lbl) & (pl.col("col") == "area_mean"))[
                    "changepoints"
                ],
                min_area,
                max_area,
                color="red",
                linestyles="dashed",
            )


        axes[0].legend()

        # Intensity 
        axes[1].set_title("droplet intensity")
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("average intensity (0-255)")
        axes[1].plot(
            df_sub["time"], df_sub["intensity"], color=lbl_color, alpha=0.25
        )
        axes[1].plot(
            df_sub["time"],
            df_sub["intensity_mean"],
            color=lbl_color,
            label=legend_label,
        )

        if len(cpts) > 0:
            axes[1].vlines(
                cpts.filter((pl.col("id") == lbl) & (pl.col("col") == "intensity_mean"))[
                    "changepoints"
                ],
                min_intense,
                max_intense,
                color="red",
                linestyles="dashed",
            )
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(path)

def main():
    # Setup logger
    logger = logging.getLogger("main")
    
    # Parse arguments
    path = args.file

    # Create output directory if it doesn't already exist
    out_dir = os.path.dirname(path) + "/analysis"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if not os.path.isdir(out_dir):
        logger.critical("Unable to create output directory, already exists")
        exit(1)

    logger.info(f"outputting files into: {out_dir}")
    
    measurements = analyseVideo(path, out_dir)
    df = measurements2dataframe(measurements)
    df = processMeasurements(df)
    df.write_csv(f"{out_dir}/data.csv")
    logger.info("Saved data csv")

    cpts = [] #detectChangepoints(df, ["area_mean", "intensity_mean"])

    plotMeasurements(df, cpts, f"{out_dir}/plots.png")
    logger.info("Saved plots")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
                    prog="SECCM droplet analyser",
                    description="A visual droplet size and brightness analyser"+
                                "for my amazing girlfriend")
    parser.add_argument("file", help="The file path to the .avi source file")
    parser.add_argument("--log", help="The log level",choices=['info', 'debug', 'error'], default="info")

    args = parser.parse_args();

    # Setup logging format
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
        level={"info": logging.INFO, 
               "debug":logging.DEBUG, 
               "error": logging.ERROR}[args.log])
    # Run the analys
    main()
