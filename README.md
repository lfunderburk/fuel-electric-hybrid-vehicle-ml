ml-vehicle-classification
==============================

Classifying data from hybrid, fuel only and electric vehicles

# Set up 

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
      model: models/hard_voting_classifier_co2_fuel.pkl
  - source: src/models/predict_model.py
    product:
      nb: notebooks/predict_model.ipynb
  - source: src/models/clustering.py
    product:
      nb: notebooks/clustering.ipynb
```

Sample output

```
name             Ran?      Elapsed (s)    Percentage
---------------  ------  -------------  ------------
data_extraction  True          29.371        8.13723
train_model      True         136.637       37.8553
predict_model    True          52.2234      14.4685
clustering       True         142.715       39.5391
```

## Running tests

From command line at the project root directory level

```
pytest
```

## Deployment methods:

1. This application consists of a Dash app with a dashboard that allows the user to visualize trends in different kinds of vehicles and consumer trends with a time component. 

2. The data pipeline is scheduled to refresh and retrain the model in batches, and saves the model's results to a database/api for easier retrieval. 


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
