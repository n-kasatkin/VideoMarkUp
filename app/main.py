import os
import numpy as np
import pandas as pd
import cv2
import pickle

import streamlit as st

import vtools as vt


def main():
    st.set_page_config(
        page_title="Video App",
        page_icon=":shark:"
    )
    st.title("This is an app for manual football video markup")

    # Preprocess
    expander = st.beta_expander(label="Preprocessing")
    with expander:
        video_path = choose_video()
        detects_path = choose_detects()
        load_bar = st.progress(0)
        load_button = st.button("Start preprocessing video")
        if load_button:
            vt.preprocess_data(video_path, detects_path, pbar=load_bar)

    # Load data
    data, track_ids = load_preprocessed_data()

    # Track
    track_id = choose_track(track_ids)
    frames = data[track_id]
    st.text(f"There are {len(frames):5d} frames for this track id.")

    # Canvas
    expander = st.beta_expander(label="Look at the whole track")
    image = canvas(frames, n=int(np.sqrt(len(frames))))
    expander.image(image, output_format='PNG')

    # Frame
    frame_no = choose_frame_no(len(frames) - 1)
    st.text(f"This is {frame_no:5d} frame.")

    # Images
    show_images(frame_no, frames, N=19)
    show_images(frame_no, frames, N=39, show_caption=False)

    # Create file with marked data
    adding_sequences(frames)


def adding_sequences(frames):
    columns = st.beta_columns(3)
    with columns[0]:
        visible_number = st.number_input("Visible number", min_value=-1)
    with columns[1]:
        first_frame = st.number_input("First frame", min_value=0)
    with columns[2]:
        last_frame = st.number_input("Last frame", min_value=0)
    add_string_button = st.button("Add this sequence")
    if add_string_button:
        with open("../output/final.csv", "a") as file:
            for i in range(first_frame, last_frame + 1):
                file.write(frames[i] + "," + str(visible_number) + "\n")


def load_preprocessed_data(workdir="../output/preprocessed_videos"):
    st.header("Load data")
    pickles = [name for name in os.listdir(
        workdir) if name.endswith(".pickle")]
    pickle_file = st.selectbox("Choose file", options=pickles)
    with open(os.path.join(workdir, pickle_file), 'rb') as handle:
        data = pickle.load(handle)
    track_ids = data.keys()
    return data, track_ids


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


def choose_track(track_ids):
    st.header("Choose track id")
    st.text(
        f"There are {len(track_ids)} tracks, starting from {min(track_ids)}.")
    track_id = st.number_input("Track id", min_value=min(
        track_ids), max_value=max(track_ids), step=1)
    return track_id


def load_track(track_id):
    bar = st.progress(0)
    frames, bbox_get_func = vt.get_track(track_id, pbar=bar)
    bbox_get_func = st.cache(bbox_get_func, show_spinner=False)
    return frames, bbox_get_func


def choose_frame_no(max_val):
    st.header('Choose frame')
    frame_no = st.slider('Frame_no', min_value=0, max_value=max_val)
    return frame_no


def show_images(frame_no, frames, N=19, show_caption=True):
    size = 640 // N
    columns = st.beta_columns(N)
    for i, column in enumerate(columns):
        with column:
            rel_no = i - (N - 1) // 2
            abs_no = rel_no + frame_no
            if abs_no < 0 or abs_no >= len(frames):
                frame = np.zeros((size, size, 3))
            else:
                frame = vt.load_image(frames[rel_no + frame_no])
                frame = cv2.resize(frame, (size, size))
            if show_caption:
                caption = str(rel_no + frame_no) if rel_no != 0 else "↑"
            else:
                caption = "" if rel_no != 0 else "↑"
            st.image(frame[:, :, ::-1], caption=caption, output_format='PNG')


@st.cache(max_entries=2)
def canvas(frames, n=70, size=32):
    image = np.zeros((size*n, size*n, 3))
    for idx, img_path in enumerate(frames):
        img = vt.load_image(img_path)
        img = cv2.resize(img, (size, size))
        vt.write_text(img, str(idx))
        i, j = idx // n, idx % n
        image[i*size:(i+1)*size, j*size:(j+1)*size] = img
        if idx == n**2 - 1:
            break
    # image = cv2.resize(image[:, :, ::-1] / 255., (640, 640), cv2.INTER_AREA)
    return image[:, :, ::-1] / 255.


if __name__ == "__main__":
    main()
