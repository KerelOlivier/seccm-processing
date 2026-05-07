import scipy.signal as sps
import scipy as sc
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import cv2.ximgproc as xip
import sys
import os
import logging
from tqdm import tqdm
import ruptures as rpt
from dataclasses import dataclass, asdict
from sklearn.mixture import GaussianMixture
from collections import deque

cmap = np.array(
    [
        (28, 26, 228),  # #e41a1c
        (184, 126, 55),  # #377eb8
        (74, 175, 77),  # #4daf4a
        (163, 78, 152),  # #984ea3
        (0, 127, 255),  # #ff7f00
        (51, 255, 255),  # #ffff33
        (40, 86, 166),  # #a65628
        (191, 129, 247),  # #f781bf
        (153, 153, 153),  # #999999
    ],
    dtype=np.uint8,
)

plt_cmap = np.array(
    [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999",
    ]
)


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
    out = cv.VideoWriter(f"{out_dir}/output.avi", fourcc, fps, (width, height))
    
    frames = np.array([cap.read()[1] for i in range(frame_count)])
    gray = [cv.cvtColor(frame, cv.COLOR_BGR2GRAY) for frame in frames]

    window = 7

    histogram = []
    peaks = []

    measurements = []

    # Centroid tracking variables
    cid_cnt = 1
    centroid_ids = dict()

    # Tracks at what frame each id is made
    id2start = dict()


    for i in tqdm(range(window//2, frame_count - window//2)):
        frame = gray[i]
        m = np.median(gray[i - window//2:i+window//2], axis=0).astype(np.uint8)

        denoised = cv.bilateralFilter(m, 9, 75, 75)

        # Determine thresholding
        hist = cv.calcHist([denoised], [0], None, [256], [0,256]).flatten()
        hist = np.convolve(hist, np.ones(11)/11, mode='same')
        pks = sps.find_peaks(hist, prominence=200, distance=10, width=10)
        widths = pks[1]["widths"]
        pks = pks[0]
        if(len(pks)) != 2: 
            pks = [pks[0], 255]
            widths = [widths[0],0]
            

        peaks.append(pks)
        
        # Threshold values
        mask = np.zeros_like(frame)
        mask[denoised > pks[1] - widths[1] * 2] = 255
        mask = cv.threshold(denoised, 0, 255,
                     cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        mask[denoised < pks[0] + widths[0]] = 0

        histogram.append(hist)

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
        

        # Video
        centroid_ids = new_cids
        label_bgr = cmap[np.uint8(ids % len(cmap))]

        # blend overlay with original source
        coloured = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

        green_mask = cv.bitwise_and(label_bgr, label_bgr, mask=mask)

        alpha = 0.4
        output = cv.addWeighted(coloured, 1.0, green_mask, alpha, 0)

        for j, lbl in enumerate(lbls):
            if j == 0:
                continue
            c = centroids[lbl]
            output = cv.circle(
                output, (int(c[0]), int(c[1])), 5, (j * 70, 0, j * 70), -1
            )
            area = stats[lbl, cv.CC_STAT_AREA]
            output = cv.putText(
                output,
                str(area),
                (int(c[0]), int(c[1])),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )



        
        out.write(output)

            
    # plot histogram
    histogram = np.array(histogram)
    peak_img = np.zeros_like(histogram)
    print(len(peaks))
    peaks = np.array(peaks)
    for i, pks in enumerate(peaks):
        freqs = np.floor(pks).astype(np.int32)
        peak_img[i, freqs] = 1

    plt.imshow(
        histogram.T,
        aspect = 'auto',
        origin = 'lower',
        cmap = 'viridis'
    )
    plt.colorbar()
    plt.savefig(f"{out_dir}/histogram.png")


    cap.release()
    out.release()

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

def detectChangepoints(df: pl.DataFrame, cols: list[str]):
    logger = logging.getLogger("chd")
    res = pl.DataFrame()

    for lbl in df["id"].unique():
        for col in cols:
            df_sub = df.filter(pl.col("id") == lbl).sort("time")

            if len(df_sub) < 25 : 
                continue

            # Changepoint detection
            model = "rbf"
            algo = rpt.Pelt(model=model, min_size=50, jump=20).fit(
                df_sub[col].to_numpy()
            )
            penalty = 50 * np.log(len(df_sub))
            changepoints = np.array(algo.predict(pen=penalty))
            cp_times = df_sub["time"][changepoints[:-1]]

            if len(cp_times > 0):
                row = pl.DataFrame(
                    {
                        "id": [lbl] * len(cp_times),
                        "col": [col] * len(cp_times),
                        "changepoints": cp_times,
                    }
                )

                if len(res) > 0:
                    res = res.extend(row)
                else:
                    res = row
    return res

def plotMeasurements(df, cpts, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    alpha = 0.25

    min_area = df["area_mean"].min()
    max_area = df["area_mean"].max()

    min_intense = df["intensity_mean"].min()
    max_intense = df["intensity_mean"].max()

    for lbl in df["id"].unique():

        lbl_color = plt_cmap[lbl%len(plt_cmap)]
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

    plt.tight_layout()
    plt.savefig(path)

def main():
    logger = logging.getLogger("main")
    
    path = sys.argv[1]

    # Create output directory if it doesn't already exist
    out_dir = os.path.basename(path).split('.')[0]
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


def test(path):
    logger = logging.getLogger("test")
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
    out = cv.VideoWriter(f"test.avi", fourcc, fps, (width, height))
    
    frames = np.array([cap.read()[1] for i in range(frame_count)])
    gray = [cv.cvtColor(frame, cv.COLOR_BGR2GRAY) for frame in frames]

    window = 7

    histogram = []
    peaks = []

    measurements = []

    # Centroid tracking variables
    cid_cnt = 1
    centroid_ids = dict()

    # Tracks at what frame each id is made
    id2start = dict()


    for i in tqdm(range(window//2, frame_count - window//2)):
        frame = gray[i]
        m = np.median(gray[i - window//2:i+window//2], axis=0).astype(np.uint8)

        denoised = cv.bilateralFilter(m, 9, 75, 75)

        # Determine thresholding
        hist = cv.calcHist([denoised], [0], None, [256], [0,256]).flatten()
        hist = np.convolve(hist, np.ones(11)/11, mode='same')
        pks = sps.find_peaks(hist, prominence=200, distance=10, width=10)
        widths = pks[1]["widths"]
        pks = pks[0]
        if(len(pks)) != 2: 
            pks = [pks[0], 255]
            widths = [widths[0],0]
            

        peaks.append(pks)
        
        # Threshold values
        mask = np.zeros_like(frame)
        mask[denoised > pks[1] - widths[1] * 2] = 255
        mask = cv.threshold(denoised, 0, 255,
                     cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        mask[denoised < pks[0] + widths[0]] = 0

        histogram.append(hist)

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
        

        # Video
        centroid_ids = new_cids
        label_bgr = cmap[np.uint8(ids % len(cmap))]

        # blend overlay with original source
        coloured = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

        green_mask = cv.bitwise_and(label_bgr, label_bgr, mask=mask)

        alpha = 0.4
        output = cv.addWeighted(coloured, 1.0, green_mask, alpha, 0)

        for j, lbl in enumerate(lbls):
            if j == 0:
                continue
            c = centroids[lbl]
            output = cv.circle(
                output, (int(c[0]), int(c[1])), 5, (j * 70, 0, j * 70), -1
            )
            area = stats[lbl, cv.CC_STAT_AREA]
            output = cv.putText(
                output,
                str(area),
                (int(c[0]), int(c[1])),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )



        
        out.write(output)

            
    # plot histogram
    histogram = np.array(histogram)
    peak_img = np.zeros_like(histogram)
    print(len(peaks))
    peaks = np.array(peaks)
    for i, pks in enumerate(peaks):
        freqs = np.floor(pks).astype(np.int32)
        peak_img[i, freqs] = 1

    plt.imshow(
        histogram.T,
        aspect = 'auto',
        origin = 'lower',
        cmap = 'viridis'
    )
    plt.colorbar()
    plt.show()

    plt.imshow(
        peak_img.T,
        aspect = 'auto',
        origin = 'lower',
        cmap = 'viridis'
    )
    plt.colorbar()
    plt.show()

    cap.release()
    out.release()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
        level=logging.INFO)
    #test(sys.argv[1])
    main()
