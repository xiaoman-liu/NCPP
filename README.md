Copyright (C) <2024> Intel Corporation

SPDX-License-Identifier: Apache-2.0

NCPP - Nova CPU performance predictor
==============================

use several approach of neural networks to predict hardware performance and compare them

The purpose of this file structure is to allow data science exploration to easily transition into deployed services and applications on the OpenShift platform.

------------

## Introduction
NCPP is a machine learning model designed for CPU benchmark performance prediction. It includes modules for data processing, model training, and prediction, specific developed for Intel X86 products.

## Installation Guide

Before starting, make sure you have Python and pip installed. Then follow these steps to install the NCPP model and its dependencies:

```bash
git clone [ Repository URL]
cd [Project Directory]
pip install -r requirements.txt
python setup.py install
```

## File Structure



    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │                         also used to install packages for s2i application
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── module                <- Source code for use in this project.
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
    │   │   ├── predict_model.py <- the predict function called from Flask
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    ├── .gitignore               <- standard python gitignore
    │   └── environment          <- s2i environment settings
    └── requirements.txt   <- The requirements file for use inside Jupyter notebooks and 

--------


## Usage Instructions
### Training the Model
```bash
python module/train/train.py
```
### Predicting
```bash
python module/predict/infer.py
```
## Dependencies
List all the dependencies required by your project in the requirements.txt file. Ensure they are all installed using the pip command provided in the installation guide.

## License
This project is licensed under the Apache-2.0. See the LICENSE file for more details.  If you use this code or build upon the ideas presented in this model, please cite our paper [paper link] and/or this GitHub repository [https://github.com/xiaoman-liu/NCPP/tree/main]
## Report Paper

## Contributing
If you wish to contribute to this project, please refer to the CONTRIBUTING.md file for guidelines.

## Contact

For any queries or further assistance, please contact <span style="font-size: 16px; font-weight: bold; color: #FF0000; text-decoration: underline;">xiaoman.liu</span>@<span style="font-size: 16px; font-weight: bold; color: #007bff; text-decoration: underline;">intel.com</span>.
