# Model Fusion for Personalized Learning

This repository implements all experiments in the paper Model Fusion for Personalized Learning (submitted and under review at ICML 2021).

## Requirements

- Python >= 3.6
- numpy, scikit-learn, pytorch, panda, matplotlib, d2l
- Run the following to install the requirements
```bash
pip install -r requirements.txt
```

## Instructions for running experiment

- Set the parameters `BASE` and `PROJECT_NAME` in `utilities/util.py` to configure the path to save the models and results. Default is `/tmp/personalized-learning`.
- Turn on/off the model alignment by setting `do_alignment` to `true/false`
- Run the code as follows
```
PYTHONPATH=/path/to/the/code/directory python sine_experiment.py
PYTHONPATH=/path/to/the/code/directory python movielens_experiment.py
```

#### Sine Experiment

- The code for running the sine experiment is in `experiments/sine_experiment.py`.

#### Movie-Lens Experiment

- The code for running the Movie-Lens recommendation experiment is in `experiments/movielens_experiment.py`.

#### Meta Model implementation

- The meta model implementation is in `models/meta.py`