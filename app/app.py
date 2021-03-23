import streamlit as st

from labeling import labeling
from postprocessing import postprocessing
from preprocessing import preprocessing
from utils import config_page, get_arguments, Stage


def choose_stage():
    st.sidebar.header("Choose Stage")
    options = [Stage.PREPROCESSING.value, Stage.LABELING.value, Stage.POSTPROCESSING.value]
    stage = st.sidebar.selectbox("", options=options)
    return Stage(stage)


def main():
    config_page()

    args = get_arguments()
    stage = choose_stage()
    if stage == Stage.PREPROCESSING:
        return preprocessing(args)
    elif stage == Stage.LABELING:
        return labeling(args)
    elif stage == Stage.POSTPROCESSING:
        return postprocessing(args)
    else:
        raise ValueError(f"Unexpected stage: {stage.value}")


if __name__ == '__main__':
    main()
