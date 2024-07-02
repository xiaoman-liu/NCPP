# Copyright (C) <2024> Intel Corporation

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
git clone https://github.com/xiaoman-liu/NCPP.git
cd NCPP
pip install -r requirements.txt
```

## File Structure

```plaintext
NCPP
│
├── LICENSE                          <- The project's license file, detailing copyright and licensing information.
├── README.md                        <- The project's README file, providing an overview, installation instructions, and usage information.
│
├── data                             <- Data directory, containing the samples of dataset.
│   ├── external                     <- External data from public source.
│   ├── interim                      <- Intermediate data that has been transformed.
│   ├── processed                    <- The final, canonical datasets for modeling.
│   └── raw                          <- The original, immutable data.
│       └── SPR                      <- Data specific to a particular experiment.
│           ├── characteristic_description.md <- Characteristic description files.
│           ├── test_data.csv        <- Test dataset.
│           └── train_data.csv       <- Training dataset.
│
├── docs                             <- Documentation directory, containing default Sphinx project documentation.
│
│
├── module                           <- Source code directory, containing all project code.
│   ├── __init__.py                  <- Initialization file, making this directory a Python package.
│   ├── predict                      <- Prediction module, containing code related to predictions.
│   ├── train                        <- Training module, containing code related to model training.
│   └── visualization                <- Visualization module, containing code related to data visualization.
│       └── __init__.py              <- Initialization file, making this directory a Python package.
│
├── .gitignore                       <- Git ignore file, listing files and directories to be excluded from version control.
├── contributing.md                  <- Contribution guidelines, providing instructions on how to contribute to the project.
├── requirements.txt                 <- Lists the Python dependencies required by the project.
└── setup.py                         <- The project's installation script, containing metadata and installation information.
```



## Usage Instructions
### Training the Model
```bash
python module/train/train.py
```
### Predicting
```bash
python module/predict/infer.py
```

## License
This project is licensed under the Apache-2.0. See the LICENSE file for more details.  

## Report Paper

## Citation
If you use this code or build upon the ideas presented in this model, please cite our paper [paper link] and/or this GitHub repository [https://github.com/xiaoman-liu/NCPP/tree/main]
```bash

```

## Contact

For any queries or further assistance, please contact <span style="font-size: 16px; font-weight: bold; color: #FF0000; text-decoration: underline;">xiaoman.liu</span>@<span style="font-size: 16px; font-weight: bold; color: #007bff; text-decoration: underline;">intel.com</span>.
