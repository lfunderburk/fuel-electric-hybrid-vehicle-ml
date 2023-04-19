ml-vehicle-classification
==============================

Classifying data from hybrid, fuel only and electric vehicles

Clone the repo

```
git clone https://github.com/lfunderburk/fuel-electric-hybrid-vehicle-ml.git
cd fuel-electric-hybrid-vehicle-ml
```

## Setting up, with Docker

Ensure you have Docker installed

```
docker build -t my_pipeline .
docker run -it --rm -p 5000:5000 my_pipeline
```

## Setting up - locally


Create and activate a virtual environment

```
conda create --name mlenv python==3.10
conda activate mlenv
```

Install dependencies

```
pip install -r requrements.txt
```

## Executing the data pipeline - locally

From command line at the project root directory level

```
ploomber build
```

This command will execute the following data pipeline

```
tasks:
  - source: src/data/data_extraction.py
    product:
      nb: notebooks/data_extraction.ipynb
  - source: src/models/train_model.py
    product:
      nb: notebooks/train_model.ipynb
  - source: src/models/predict_model.py
    product:
      nb: notebooks/predict_model.ipynb
```

Sample output

```
name             Ran?      Elapsed (s)    Percentage
---------------  ------  -------------  ------------
data_extraction  True          13.0449       13.2142
train_model      True          39.0849       39.5921
predict_model    True          46.5889       47.1936
```

## Running tests

From command line at the project root directory level

```
pytest
```


#### Project Organization


    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
