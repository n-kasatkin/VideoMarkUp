import os
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import vtools as vt

from utils import (
    choose_detects,
    choose_labels,
    choose_video,
    get_arguments
)


SAVE_DIR = "../output/images/"


def main():
    st.set_page_config(
        page_title="Video App",
        page_icon=":shark:"
    )
    st.title("This is an app for manual football video markup")
    st.header("Postprocessing | saving images to disk")

    # Preprocess
    args = get_arguments()
    video_path = choose_video(args.data_dir)
    detects_path = choose_detects(args.data_dir)
    labels_path = choose_labels(args.output_dir)

    load_bar = st.progress(0)
    load_button = st.button("Start processing video")
    if load_button:
        process_data(labels_path, detects_path, video_path, pbar=load_bar)


def process_data(labels_file, detects_file, video_file, pbar=None):
    labels_df = pd.read_csv(
        labels_file, names=["track_id", "rel_frame", "label"], dtype=int)
    detects = pd.read_csv(detects_file, low_memory=False)
    track_ids = np.unique(
        detects[~np.isnan(detects['track_id'])]['track_id']).astype(int)
    tracks = {track_id: detects[detects['track_id']
                                == track_id] for track_id in track_ids}
    frame_sets = {track_id: set() for track_id in track_ids}
    labels = {track_id: {} for track_id in track_ids}
    for i, row in labels_df.iterrows():
        track_id = row["track_id"]
        frame_no = tracks[track_id].iloc[row["rel_frame"]]["frame_id"]
        frame_sets[track_id].add(frame_no)
        labels[track_id][frame_no] = row["label"]

    match_name = video_file[video_file.rfind(
        "\\") + 1: video_file.rfind(".mp4")]
    os.makedirs(os.path.join(SAVE_DIR, match_name,
                             "not_visible"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, match_name, "visible"), exist_ok=True)

    cap = cv2.VideoCapture(video_file)
    assert cap.isOpened(), "Can't open video"
    frame_no, last_frame = 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        for track_id in track_ids:
            if frame_no in frame_sets[track_id]:
                bbox = vt.bbox_from_df(tracks[track_id], frame_no)
                _, orig_bbox = vt.get_bbox(
                    frame, bbox, return_original_bbox=True)
                save_filename = f"track_{track_id}_frame_{frame_no}.png"
                if labels[track_id][frame_no] == -1:
                    cv2.imwrite(os.path.join(SAVE_DIR, match_name,
                                             "not_visible", save_filename), orig_bbox)
                else:
                    cv2.imwrite(os.path.join(SAVE_DIR, match_name,
                                             "visible", save_filename), orig_bbox)
        frame_no += 1
        if pbar:
            pbar.progress(frame_no / last_frame)
        if frame_no == last_frame:
            break


if __name__ == "__main__":
    main()
