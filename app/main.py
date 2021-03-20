import os
import numpy as np
import cv2
import h5py
import streamlit as st
import vtools as vt
from albumentations.augmentations.functional import brightness_contrast_adjust, clahe

from utils import config_page_and_title, get_arguments


def main():
    config_page_and_title(stage="Labeling")
    args = get_arguments()

    # Sidebar options
    data_path, track_id, frames = choose_data_and_track(args.data_dir, args.output_dir)
    n_rows, images_per_row, stride = choose_image_grid_params()
    image_transforms = choose_image_transforms()

    # Frame
    step = (n_rows - 1) * images_per_row * stride
    frame_no = choose_frame_no(len(frames) - 1, step)
    st.text(f"This is {frame_no:5d} frame.")

    # Images
    for row in range(n_rows):
        row_frame_no = frame_no + row * images_per_row * stride
        show_images(row_frame_no, frames, N=images_per_row, stride=stride,
                    image_transforms=image_transforms, frame_arrow=(row == 0))

    # Create file with marked data
    adding_sequences(frames, track_id, save_filename=os.path.basename(data_path)[:-3])


def choose_data_and_track(data_dir, output_dir):
    data_path = choose_data(data_dir)
    track_id = choose_track(data_path)
    frames = load_images(track_id, data_path)
    st.text(f"There are {len(frames):5d} frames for this track id.")
    save_canvas_button = st.sidebar.button("Save the whole track canvas to disk")
    if save_canvas_button:
        image = canvas(frames, n=int(np.sqrt(len(frames))))
        cv2.imwrite(os.path.join(output_dir, "current_track.png"), image)

    return data_path, track_id, frames


def choose_image_grid_params():
    st.sidebar.header("Image Grid params")
    n_rows = st.sidebar.slider("Number of rows", min_value=1, max_value=10, value=5, step=1)
    images_per_row = st.sidebar.slider("Number of images per row", min_value=9, max_value=49, value=19, step=10)
    stride = st.sidebar.slider("Stride", min_value=1, max_value=25, value=5, step=1)
    return n_rows, images_per_row, stride


def choose_image_transforms():
    st.sidebar.header("Image transforms")
    image_transforms = {
        "brightness": st.sidebar.slider("Brightness", min_value=-1.0, max_value=1.0, value=0.0, step=0.1),
        "contrast": st.sidebar.slider("Contrast", min_value=-1.0, max_value=1.0, value=0.0, step=0.1),
    }
    # CLAHE params
    if st.sidebar.checkbox("Use CLAHE"):
        image_transforms["clahe"] = {
            "clip_limit": st.sidebar.slider("CLAHE clip_limit", min_value=0.0, max_value=8., value=4.0, step=0.1),
            "tile_grid_size": (
                st.sidebar.slider("CLAHE tile_grid_height", min_value=1, max_value=16, value=2, step=1),
                st.sidebar.slider("CLAHE tile_grid_width", min_value=1, max_value=16, value=2, step=1),
            ),
        }
    else:
        image_transforms["clahe"] = None

    return image_transforms


def choose_data(data_dir):
    files = [name for name in os.listdir(data_dir) if name.endswith(".h5")]
    st.sidebar.header("Choose data")
    data_file = st.sidebar.selectbox("", options=files)
    return os.path.join(data_dir, data_file)


def adding_sequences(frames, track_id, save_filename):
    columns = st.beta_columns(3)
    with columns[0]:
        visible_number = st.number_input("Visible number", min_value=-1)
    with columns[1]:
        first_frame = st.number_input("First frame", min_value=0)
    with columns[2]:
        last_frame = st.number_input("Last frame", min_value=0)
    add_string_button = st.button("Add this sequence")
    if add_string_button:
        with open(os.path.join("../output/", save_filename + ".txt"), "a+") as file:
            for i in range(first_frame, last_frame + 1):
                file.write(str(track_id) + "," + str(i) + "," +
                           str(visible_number) + "\n")


def get_track_ids(data_path):
    file = h5py.File(data_path, "r+")
    track_ids = [int(i) for i in file.keys()]
    file.close()
    return track_ids


def load_images(track_id, data_path):
    file = h5py.File(data_path, "r+")
    images = np.array(file[str(track_id)])
    file.close()
    return images


def choose_track(data_path):
    track_ids = get_track_ids(data_path)
    st.sidebar.header("Choose track id")
    st.sidebar.text(
        f"There are {len(track_ids)} tracks, starting from {min(track_ids)}.")
    track_id = st.sidebar.number_input(
        "Track id", min_value=min(track_ids), max_value=max(track_ids), step=1)
    return track_id


def load_track(track_id):
    bar = st.progress(0)
    frames, bbox_get_func = vt.get_track(track_id, pbar=bar)
    bbox_get_func = st.cache(bbox_get_func, show_spinner=False)
    return frames, bbox_get_func


def choose_frame_no(max_val, step=10):
    st.header('Choose frame')
    frame_no = st.slider('Frame_no', min_value=0, max_value=max_val, step=step)
    return frame_no


def show_images(frame_no, frames, N=19, stride=1, image_transforms=None,
                show_caption=True, frame_arrow=True):
    hsize = wsize = 800 // N

    def get_frame(_rel_no):
        abs_no = _rel_no + frame_no
        if abs_no < 0 or abs_no >= len(frames):
            return np.zeros((hsize, wsize, 3))

        frame = frames[abs_no]
        frame = cv2.resize(frame, (hsize, wsize))

        if image_transforms is not None:
            # Brightness and Contrast
            alpha, beta = 1.0 + image_transforms["brightness"], image_transforms["contrast"]
            frame = brightness_contrast_adjust(frame, alpha=alpha, beta=beta, beta_by_max=True)

            # CLAHE
            if image_transforms["clahe"] is not None:
                frame = clahe(frame, **image_transforms["clahe"])

        return frame[:, :, ::-1]

    def get_caption(_rel_no):
        abs_no = _rel_no + frame_no
        if show_caption:
            return str(abs_no) if rel_no != 0 or not frame_arrow else "↑"

        return "" if rel_no != 0 or not frame_arrow else "↑"

    columns = st.beta_columns(N)
    for i, column in enumerate(columns):
        with column:
            rel_no = (i - (N - 1) // 2) * stride
            frame, caption = get_frame(rel_no), get_caption(rel_no)
            st.image(frame, caption=caption, output_format='PNG')


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
