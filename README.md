# TDT05-SSL
Repository for group project in TDT05 - Self-supervised Learning at NTNU.

### Setup

Navigate to the root directory and run
```pip install -r requirements.txt```


### Folder structure
```
.
├── README.md
├── requirements.txt
└── src
    ├── data
    │   ├── test_32x32.mat
    │   └── train_32x32.mat
    ├── models
    │   ├── __init__.py
    │   └── lightning_predictor.py
    ├── processing
    │   ├── __init__.py
    │   └── custom_datasets.py
    ├── train_vit_self_supervised.py
    └── utils.py
```

- `src/data/`: This is where the SVHN dataset will be saved the first time running code
- `src/models/`: ML models goes here
- `src/processing/`: Dataset classes and processing of data goes here, so it can easily be imported in training
