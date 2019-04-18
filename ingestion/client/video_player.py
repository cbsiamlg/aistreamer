#!/usr/bin/env python2.7
import sys
import argparse
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
LABEL_DURATION = 0.15  # s -- means that we will display labels for _s after it's timestamp
NCOLORS = 5
# https://matplotlib.org/users/colormaps.html
# CMAP = cm.get_cmap('Spectral')  # Colour map (there are many others)


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i * nsc:(i + 1) * nsc, :] = rgb
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap


CMAP = categorical_cmap(NCOLORS, 1)


def parse_args(args):
    parser = argparse.ArgumentParser(description="Play output of streaming object tracking")
    parser.add_argument("-v", "--video", help="Video file to play")
    parser.add_argument("-l", "--labels", help="CSV file to read annotations from")
    return parser.parse_args(args)


def get_active_labels(labels, curr_ts):
    active_labels = labels[(labels.time > curr_ts) & (labels.time <= curr_ts + LABEL_DURATION)]
    print("[PLAYER] We have %d active labels" % len(active_labels))
    return active_labels


def draw_labels(img, labels):
    for idx, label in labels.iterrows():
        description = label.description
        tracking_id = label.track_id
        confidence = float(label.confidence)
        text = "{}: {:0.2f}".format(description, confidence)
        height, width, _ = img.shape
        leftpos = int(label.boxleft * width)
        rightpos = int(label.boxright * width)
        toppos = int(label.boxtop * height)
        bottompos = int(label.boxbottom * height)
        pos = (leftpos, toppos)
        end_x = rightpos
        end_y = bottompos
        tracking_color_index = tracking_id % NCOLORS
        color = CMAP.colors[tracking_color_index] * 255
        draw_label(img, text, pos, end_x, end_y, color)


def draw_label(img, text, pos, bb_end_x, bb_end_y, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    # Bounding box
    cv2.rectangle(img, pos, (bb_end_x, bb_end_y), bg_color, margin)
    # Text box
    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("[PLAYER] Error opening video stream")
    labels_generator = pd.read_csv(args.labels, chunksize=10, iterator=True, names=[
        "description",
        "track_id",
        "confidence",
        "time",
        "boxleft",
        "boxtop",
        "boxright",
        "boxbottom",
    ], dtype={
        "description": str,
        "track_id": int,
        "confidence": float,
        "time": float,
        "boxleft": float,
        "boxtop": float,
        "boxright": float,
        "boxbottom": float,
    })
    print("[PLAYER] Created labels generator")
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Get an initial set of labels
    labels = labels_generator.next().sort_values(by=["time"], ascending=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            ts = ts_ms / 1000.
            # Remove labels which are too old
            labels = labels[labels.time >= ts]
            # Fetch new labels if we're low
            if len(labels) <= 20:
                print("[PLAYER] Fetching new labels...")
                new_labels = labels_generator.next()
                labels = pd.concat([labels, new_labels]).sort_values(by=["time"], ascending=True)
            print("[PLAYER] We have %d labels" % len(labels))
            draw_labels(frame, get_active_labels(labels, ts))
            cv2.imshow("Video", frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
