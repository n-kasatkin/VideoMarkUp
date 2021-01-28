import os
import streamlit as st
import vtools as vt


def main():
    st.set_page_config(
        page_title="Video App",
        page_icon=":shark:"
    )
    st.title("This is an app for manual football video markup")
    st.header("Preprocessing")

    # Preprocess
    video_path = choose_video()
    detects_path = choose_detects()
    load_bar = st.progress(0)
    load_button = st.button("Start preprocessing video")
    if load_button:
        vt.preprocess_data(video_path, detects_path, pbar=load_bar)


def choose_video():
    videos = [name for name in os.listdir("../data") if name.endswith(".mp4")]
    video_file = st.selectbox("Choose video", options=videos)
    path = os.path.join("../data", video_file)
    st.video(path)
    return path


def choose_detects():
    detects = [name for name in os.listdir("../data") if name.endswith(".csv")]
    detects_file = st.selectbox(
        "Choose detects file", options=detects)
    path = os.path.join("../data", detects_file)
    return path


if __name__ == "__main__":
    main()
