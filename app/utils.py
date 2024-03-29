import json
import os
import re
from argparse import ArgumentParser
from enum import Enum

import pandas as pd

import streamlit as st

from segments import TrackNumberSegments


class Stage(Enum):
    PREPROCESSING = "Preprocessing"
    LABELING = "Labeling"
    POSTPROCESSING = "Postprocessing"


@st.cache
def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="../output")
    return parser.parse_args()


def config_page():
    st.set_page_config(
        page_title="Jersey App",
        page_icon=":soccer:",
        layout="wide",
    )


def title(stage=None):
    title = "Jersey Number Labeling"
    if stage is not None:
        title += f": {stage.value.title()}"
    st.title(title)


def choose_detects(data_dir):
    detects = [name for name in os.listdir(data_dir) if name.endswith(".csv")]
    detects_file = st.selectbox("Choose detects file", options=detects)
    path = os.path.join(data_dir, detects_file)
    return path


def choose_labels(output_dir):
    labels = [name for name in os.listdir(
        output_dir) if name.endswith(".json")]
    labels_file = st.selectbox("Choose labels", options=labels)
    path = os.path.join(output_dir, labels_file)
    return path


def choose_video(data_dir):
    videos = [name for name in os.listdir(data_dir) if name.endswith(".mp4")]
    video_file = st.selectbox("Choose video", options=videos)
    path = os.path.join(data_dir, video_file)
    st.video(path)
    return path


def load_sequences(file, track_ids=None):
    if os.path.exists(file):
        with open(file) as inpf:
            sequences = json.load(inpf).items()
        sequences = {int(track_id): TrackNumberSegments(val)
                     for track_id, val in sequences}
    else:
        sequences = dict()

    if track_ids is not None:
        for track_id in track_ids:
            if track_id not in sequences:
                sequences[track_id] = TrackNumberSegments()

    return sequences


def save_sequences(sequences, file):
    sequences = {key: track_segments.segments for key,
                 track_segments in sequences.items()}
    with open(file, "w+") as outf:
        json.dump(sequences, outf)


def create_dataframe(images_path, rel_dir, save_path=None):
    images = os.listdir(images_path)
    data = []
    for image in images:
        info = re.match(r"label_(.+)_track_(.+)_frame_(.+)\..+", image)
        data.append([*info.groups()] + [rel_dir + "/" + image])

    df = pd.DataFrame(data=data, columns=[
                      "number", "track_id", "frame_id", "rel_path"])
    if save_path:
        df.to_csv(save_path, index=False)
    return df
