import os
import os.path as osp
from shutil import copy, make_archive, rmtree

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import vtools as vt
from utils import (
    choose_detects,
    choose_labels,
    choose_video,
    config_page,
    get_arguments,
    load_sequences,
    title,
    Stage,
    create_dataframe
)


def postprocessing(args):
    title(stage=Stage.POSTPROCESSING)

    # Preprocess
    video_path = choose_video(args.data_dir)
    detects_path = choose_detects(args.data_dir)
    labels_path = choose_labels(args.output_dir)

    load_bar = st.progress(0)
    load_button = st.button("Start processing video")
    if load_button:
        process_data(labels_path, detects_path, video_path,
                     args.output_dir, pbar=load_bar)
        st.text("Video successfully postprocessed.")


def process_data(labels_file, detects_file, video_file, output_dir, pbar=None):
    sequences = load_sequences(labels_file)
    labels = []
    for track_id, track_sequences in sequences.items():
        for first, last, number in track_sequences.segments:
            for frame_id in range(first, last + 1):
                labels.append({
                    "track_id": track_id,
                    "rel_frame": frame_id,
                    "label": number,
                })
    labels_df = pd.DataFrame(labels, dtype=int)

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
    match_dir = os.path.join(output_dir, match_name)
    match_images_dir = os.path.join(match_dir, match_name)
    os.makedirs(match_images_dir, exist_ok=True)

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
                if labels[track_id][frame_no] == -1:
                    save_filename = f"label_nan_track_{track_id}_frame_{frame_no}.png"
                else:
                    save_filename = f"label_{labels[track_id][frame_no]}_track_{track_id}_frame_{frame_no}.png"
                cv2.imwrite(os.path.join(
                    match_images_dir, save_filename), orig_bbox)

        output_df_path = os.path.join(match_dir, match_name + ".csv")
        create_dataframe(match_images_dir, match_name, output_df_path)

        frame_no += 1
        if pbar:
            pbar.progress(frame_no / last_frame)
        if frame_no == last_frame:
            break

    labels_df.to_csv(osp.join(match_dir, osp.splitext(
        osp.basename(labels_file))[0] + ".csv"), index=False)

    # copy(labels_file, match_dir)
    # make_archive(os.path.join(output_dir, match_name), "zip", match_dir)
    # rmtree(match_dir)


def main():
    config_page()
    postprocessing(get_arguments())


if __name__ == "__main__":
    main()
