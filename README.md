# VideoMarkUp

## Usage

#### How to run the app

1. Create new virtualenv and activate it
```
python3 -m virtualenv .venv
source .venv/bin/activate
```

2. Install requirements
```
python -m pip install -r requirements.txt
```

3. Create directory "data" in the repo and copy video and detects files to it.
Or you can specify custom data directory using --data_dir arg.

4. Run
```
streamlit run app/app.py [--data_dir, --output_dir]
```

#### How to use

1. Preprocessing — extracting player boxes from the video and save them for labeling.

- Choose video and detects, run Start and wait all boxes to be extracted.

2. Labeling

- Choose data for labeling on the sidebar
- Adjust image grid
- Play with image transforms to improve images quality
- Label track by track

3. Postprocessing

TBD
