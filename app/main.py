import os
import numpy as np
import cv2
import h5py
import streamlit as st
import vtools as vt


def main():
    st.set_page_config(
        page_title="Video App",
        page_icon=":shark:",
        layout='wide'
    )
    st.title("This is an app for manual football video markup")

    # Choose .h5 file
    files = [name for name in os.listdir("../data") if name.endswith(".h5")]
    data_file = st.selectbox("Choose data", options=files)
    data_path = os.path.join("../data", data_file)

    # Track
    track_ids = get_track_ids(data_path)
    track_id = choose_track(track_ids)
    frames = load_images(track_id, data_path)
    st.text(f"There are {len(frames):5d} frames for this track id.")

    # Button to save canvas for given track_id
    save_canvas_button = st.button("Save the whole track as an image to disk")
    if save_canvas_button:
        image = canvas(frames, n=int(np.sqrt(len(frames))))
        cv2.imwrite("../output/current_track.png", image)

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


def get_track_ids(data_path):
    file = h5py.File(data_path, "r+")
    track_ids = [int(i) for i in file.keys()]
    file.close()
    return track_ids


@st.cache
def load_images(track_id, data_path):
    file = h5py.File(data_path, "r+")
    images = np.array(file[str(track_id)])
    file.close()
    return images


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
    size = 800 // N
    columns = st.beta_columns(N)
    for i, column in enumerate(columns):
        with column:
            rel_no = i - (N - 1) // 2
            abs_no = rel_no + frame_no
            if abs_no < 0 or abs_no >= len(frames):
                frame = np.zeros((size, size, 3))
            else:
                frame = frames[rel_no + frame_no]
                frame = cv2.resize(frame, (size, size))
            if show_caption:
                caption = str(rel_no + frame_no) if rel_no != 0 else "↑"
            else:
                caption = "" if rel_no != 0 else "↑"
            st.image(frame[:, :, ::-1], caption=caption, output_format='PNG')


def canvas(frames, n=70, size=32):
    image = np.zeros((size*n, size*n, 3))
    for idx, bbox in enumerate(frames):
        img = bbox.copy()
        vt.write_text(img, str(idx))
        i, j = idx // n, idx % n
        image[i*size:(i+1)*size, j*size:(j+1)*size] = img
        if idx == n**2 - 1:
            break
    return image


if __name__ == "__main__":
    main()
