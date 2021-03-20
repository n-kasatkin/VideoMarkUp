import streamlit as st
import vtools as vt

from utils import (
    choose_detects,
    choose_video,
    config_page_and_title,
    get_arguments
)


def main():
    config_page_and_title(stage="preprocessing")

    # Preprocess
    args = get_arguments()
    video_path = choose_video(args.data_dir)
    detects_path = choose_detects(args.data_dir)
    load_bar = st.progress(0)
    load_button = st.button("Start preprocessing video")
    if load_button:
        vt.preprocess_data(video_path, detects_path, pbar=load_bar)


if __name__ == "__main__":
    main()