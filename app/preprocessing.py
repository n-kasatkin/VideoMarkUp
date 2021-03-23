import streamlit as st
import vtools as vt

from utils import (
    choose_detects,
    choose_video,
    config_page,
    get_arguments,
    title,
    Stage
)


def preprocessing(args):
    title(stage=Stage.PREPROCESSING)

    # Preprocess
    video_path = choose_video(args.data_dir)
    detects_path = choose_detects(args.data_dir)
    load_bar = st.progress(0)
    load_button = st.button("Start preprocessing video")
    if load_button:
        vt.preprocess_data(video_path, detects_path, args.data_dir, pbar=load_bar)
        st.text("Video successfully preprocessed.")


def main():
    config_page()
    preprocessing(get_arguments())


if __name__ == "__main__":
    preprocessing()
