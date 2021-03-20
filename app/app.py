import streamlit as st

from labeling import labeling
from postprocessing import postprocessing
from preprocessing import preprocessing
from utils import config_page, Stage


def choose_stage():
    st.sidebar.header("Choose Stage")
    options = [Stage.PREPROCESSING.value, Stage.LABELING.value, Stage.POSTPROCESSING.value]
    stage = st.sidebar.selectbox("", options=options)
    return Stage(stage)


def main():
    config_page()

    stage = choose_stage()
    if stage == Stage.PREPROCESSING:
        return preprocessing()
    elif stage == Stage.LABELING:
        return labeling()
    elif stage == Stage.POSTPROCESSING:
        return postprocessing()
    else:
        raise ValueError(f"Unexpected stage: {stage.value}")


if __name__ == '__main__':
    main()
