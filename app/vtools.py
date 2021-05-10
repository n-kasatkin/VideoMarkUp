import os
import numpy as np
import pandas as pd
import cv2
import h5py


# For writing number
font = cv2.FONT_HERSHEY_SIMPLEX
org = (0, 7)
fontScale = 0.25
color = (255, 0, 0)
thickness = 1


def write_text(image, text):
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                fontScale, color, thickness, cv2.LINE_AA)


def preprocess_data(video_file, detects_file, data_dir, pbar=None):
    detects = pd.read_csv(detects_file, low_memory=False)
    NUMBERS_AVAILIBLE = True if "jersey_number" in detects.columns.values else False
    track_ids = np.unique(
        detects[~np.isnan(detects['track_id'])]['track_id']).astype(int)
    tracks = {track_id: detects[detects['track_id']
                                == track_id] for track_id in track_ids}
    frame_sets = {track_id: set(
        tracks[track_id]['frame_id'].values) for track_id in track_ids}

    match_name = video_file[video_file.rfind(
        "\\") + 1: video_file.rfind(".mp4")]

    cap = cv2.VideoCapture(video_file)
    assert cap.isOpened(), "Can't open video"
    frame_no, last_frame = 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracks_images = {track_id: [] for track_id in track_ids}
    tracks_meta = {track_id: [] for track_id in track_ids}
    while True:
        ret, frame = cap.read()
        for track_id in track_ids:
            if frame_no in frame_sets[track_id]:
                if NUMBERS_AVAILIBLE:
                    bbox, label = bbox_from_df(
                        tracks[track_id], frame_no, return_jersey_number=True)
                    bbox_img = get_bbox(frame, bbox)
                    tracks_images[track_id].append(
                        cv2.resize(bbox_img, (32, 32)))
                    tracks_meta[track_id].append(label)
                else:
                    bbox = bbox_from_df(
                        tracks[track_id], frame_no, return_jersey_number=False)
                    bbox_img = get_bbox(frame, bbox)
                    tracks_images[track_id].append(
                        cv2.resize(bbox_img, (32, 32)))
        frame_no += 1
        if pbar:
            pbar.progress(frame_no / last_frame)
        if frame_no == last_frame:
            break

    file = h5py.File(os.path.join(data_dir, f"{match_name}.h5"), "w")
    for track_id in track_ids:
        file.create_dataset(
            str(track_id), np.shape(tracks_images[track_id]),
            h5py.h5t.STD_U8BE,
            data=tracks_images[track_id]
        )
        if NUMBERS_AVAILIBLE:
            file.create_dataset(
                str(track_id) + "_meta", np.shape(tracks_meta[track_id]),
                dtype='i4',
                data=tracks_meta[track_id]
            )
    file.close()


def pad(image, size=(288, 288), upsample=1, half_image=True):
    img = image.copy()
    if half_image:
        h, w, _ = img.shape
        img = img[:h//2, :, :]
    if upsample != 1:
        h, w, _ = img.shape
        new_w, new_h = int(np.floor(w*upsample)), int(np.floor(h*upsample))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    h, w, _ = img.shape
    if h > size[0]:
        c = h // 2
        img = img[c-size[0]//2:c+size[0]//2, :, :]
    if w > size[1]:
        c = w // 2
        img = img[:, c-size[1]//2:c+size[1]//2, :]
    h, w, _ = img.shape

    bh, ah = int(np.floor((size[0] - h)/2)), int(np.ceil((size[0] - h)/2))
    bw, aw = int(np.floor((size[1] - w)/2)), int(np.ceil((size[1] - w)/2))
    out = np.pad(img, ((bh, ah), (bw, aw), (0, 0)))
    return out


def get_bbox(frame, bbox, return_original_bbox=False):
    """bbox is [x, y, w, h]"""
    x, y, w, h = bbox
    bbox_img = frame[y:y+h, x:x+w]
    h, w, _ = bbox_img.shape
    h1, h2 = h // 7, h // 2
    if return_original_bbox:
        return bbox_img[h1: h2], bbox_img
    else:
        return bbox_img[h1: h2]


def bbox_from_df(df, frame_no, return_jersey_number=False):
    row = df[df['frame_id'] == frame_no].iloc[0]
    bbox = [row['x_camera'], row['y_camera'], row['box_w'], row['box_h']]
    if return_jersey_number:
        return bbox, row['jersey_number']
    return bbox
