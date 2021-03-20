import os
from argparse import ArgumentParser

import streamlit as st


@st.cache
def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="../output")
    return parser.parse_args()


def config_page_and_title(stage=None):
    st.set_page_config(
        page_title="Video App",
        page_icon=":soccer:",
        layout="wide",
    )
    title = "Jersey Number Labeling"
    if stage is not None:
        title += f": {stage.title()}"
    st.title(title)


def choose_detects(data_dir):
    detects = [name for name in os.listdir(data_dir) if name.endswith(".csv")]
    detects_file = st.selectbox("Choose detects file", options=detects)
    path = os.path.join(data_dir, detects_file)
    return path


def choose_labels(output_dir):
    labels = [name for name in os.listdir(output_dir) if name.endswith(".txt")]
    labels_file = st.selectbox("Choose labels", options=labels)
    path = os.path.join(output_dir, labels_file)
    return path


def choose_video(data_dir):
    videos = [name for name in os.listdir(data_dir) if name.endswith(".mp4")]
    video_file = st.selectbox("Choose video", options=videos)
    path = os.path.join(data_dir, video_file)
    st.video(path)
    return path
