[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<a href="https://www.codefactor.io/repository/github/surfaceanalytics/xpsdeeplearning"><img src="https://www.codefactor.io/repository/github/surfaceanalytics/xpsdeeplearning/badge?s=e9dbf0ea5d4ce269c738ea85e856fd6811b425ce" alt="CodeFactor" /></a>
![](https://github.com/FAIRmat-NFDI/pynxtools/actions/workflows/pytest.yml/badge.svg)
![](https://github.com/FAIRmat-NFDI/pynxtools/actions/workflows/pylint.yml/badge.svg)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/lukaspie/xpsdeeplearning/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/444112446.svg)](https://zenodo.org/badge/latestdoi/444112446)

<table align="center">
<tr><td align="center" width="10000">

# <strong> Deep learning for XPS spectra </strong>

<p>
    Dr. Lukas Pielsticker,
    Dr. Rachel L. Nicholls,
    Dr. Serena DeBeer,
    Dr. Mark Greiner
    <br>
</p>

<p>
    <a href="https://www.cec.mpg.de/en/research/scientific-infrastructure/dr-walid-hetaba">XPS Group </a> @ <a href="https://www.cec.mpg.de">MPI for Chemical Energy Conversion </a>
</p>

<p>
    <a href="#installation">Installation</a> • <a href="#usage">Usage</a> • <a href="#publications">Publications</a> • <a href="#contributing">Contributing</a>
</p>

</td></tr></table>

`xpsdeeplearning` is a toolbox for X-ray Photoelectron Spectroscopy (XPS) data analysis using Convolutional Neural Networks, designed for speeding up your XPS analysis.

The package combines
1) simulation tools for creating large data sets of synthetic (yet realistic) transition-metal XP spectra base on reference data and
2) a supervised machine learning framework based on artificial convolutional neural networks that is able to accurately perform automated quantitative analysis of the phases present in such XP spectra.

# Installation
It is recommended to use **python 3.11** (you will need a version between python 3.9 and 3.11) with a dedicated virtual environment for this package.
Learn how to manage [python versions](https://github.com/pyenv/pyenv) and
[virtual environments](https://realpython.com/python-virtual-environments-a-primer/).

The quickest way to get started is to clone this repository:

```shell
git clone https://github.com/surfaceanalytics/xpsdeeplearning.git
cd xpsdeeplearning
```
...and install it with [uv](https://docs.astral.sh/uv/) (recommended) or pip:
```shell
# with uv
uv pip install .
# with pip
pip install .
```

## A note on model visualization
In case you want to save and plot an image of your Tensorflow/Keras model during training/prediction, you can call the `save_and_print_model_image` method of the [`Classifier`](https://github.com/surfaceanalytics/xpsdeeplearning/tree/main/xpsdeeplearning/network/classifier.py) class. This method utilizes the [`plot_model`](https://keras.io/api/utils/model_plotting_utils/) function of Keras to convert the model to dot format and save the image to a file. However, for the conversion to work, you need to have `graphviz` installed locally. You can simply [download the binary](https://www.graphviz.org/download/source/) and install.

For conda users, it is also possible to install the binary in a decicated environment by running
```shell
conda install anaconda::graphviz
```

# Usage
## Data simulation
The software tools for data simulation are located inside [`simulation`](https://github.com/surfaceanalytics/xpsdeeplearning/tree/main/xpsdeeplearning/simulation), with exemplary JSON files containing the parameters used during simulation available in the [`params`](https://github.com/surfaceanalytics/xpsdeeplearning/tree/main/xpsdeeplearning/simulation/params) subfolder. The reference data that is used as input for the simulation is available inside [`data`](https://github.com/surfaceanalytics/xpsdeeplearning/tree/main/xpsdeeplearning/data). Simulation runs can be performed using the [`run`](https://github.com/surfaceanalytics/xpsdeeplearning/tree/main/xpsdeeplearning/simulation/run.py) script or the `simulate` tool from the command line (see below.)

## Model training and prediction
The code for convolutional neural network training and quantitative prediction is located inside [`network`](https://github.com/surfaceanalytics/xpsdeeplearning/tree/main/xpsdeeplearning/network). The design of the model architectures as well as the training and testing of the neural networks were performed using [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/). In order to run the training and prediction pipelines, it is generally advised to use the [`notebooks`](https://github.com/surfaceanalytics/xpsdeeplearning/tree/main/notebooks), either through your local Jupyter installation or through [`Google Colab`](https://colab.research.google.com/). These notebooks allow visualization of the training/prediction inputs and outputs as well as saving of all results.

In addition, both training and prediction can be run from the command line using JSON files as input (see the [README](https://github.com/surfaceanalytics/xpsdeeplearning/tree/main/xpsdeeplearning/network/README.md)) for more information.

## Command line tools
There are several tools available that allow the user to run the simulation, training, and quantification routines from the command line:
- [**simulate**](https://github.com/surfaceanalytics/xpsdeeplearning/tree/main/xpsdeeplearning/simulation/README.md): Simulate large artificial XPS data sets.
- [**train**](https://github.com/surfaceanalytics/xpsdeeplearning/tree/main/xpsdeeplearning/network/README.md): Simplified pipeline for tensorflow model training.
- [**predict**](https://github.com/surfaceanalytics/xpsdeeplearning/tree/main/xpsdeeplearning/network/README.md): Simplified pipeline for quantitative prediction using trained models.

You can find more information on how to use these tools in the individual READMEs available through the links above.

## Data availability
Some simulated data sets and trained models are available through the [`KEEPER`](https://keeper.mpdl.mpg.de/d/25ebee640ba54622864a/) archive of the Max Planck society.

## Tests
All tools are shipped with unit tests located in [`tests`](https://github.com/surfaceanalytics/xpsdeeplearning/tree/main/tests).

# Publications
## Software design, model architectures, and applications
*[Convolutional neural network framework for the automated analysis of transition metal X-ray photoelectron spectra](https://doi.org/10.1016/j.aca.2023.341433)* - L. Pielsticker, R. L. Nicholls, S. DeBeer, and M. Greiner, *Anal. Chim. Acta*, 2023, **1271**, 341433.
## Datasets and trained models
*[DeepXPS: Data Sets and Trained Models](https://keeper.mpdl.mpg.de/d/25ebee640ba54622864a/)* - L. Pielsticker, R. L. Nicholls, S. DeBeer, and M. Greiner, KEEPER archive, 2023.

# Contributing
## Development install
Install the package with its dependencies:

```shell
git clone https://github.com/surfaceanalytics/xpsdeeplearning.git \\
    --branch main xpsdeeplearning
cd xpsdeeplearning
uv pip install -e ".[dev]"
```

There is also a [pre-commit hook](https://pre-commit.com/#intro) available
which formats the code and checks the linting before actually commiting.
It can be installed with
```shell
pre-commit install
```
from the root of this repository.

## Test this software

Especially relevant for developers, there exists a basic test framework written in
[pytest](https://docs.pytest.org/en/stable/) which can be used as follows:

```shell
python -m pytest -sv tests
```

# Questions, suggestions?

To ask further questions, to make suggestions how we can improve `xpsdeeplearning`, to get advice
on how to build on this work, or to get your XPS data ready for neural network analysis, you can:

- Open an issue on the [xpsdeeplearning GitHub](https://github.com/surfaceanalytics/xpsdeeplearning)
- Contact directly the lead developer (Lukas Pielsticker).

# License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.