# Correlations between heart sound components and hemodynamic variables
This repository contains the signal processing code used in the study titled "Correlations between heart sound components and hemodynamic variables" which is published in the Scientific Reports journal.

# Overview
The aim of this study was to investigate the correlations between heart sound components and hemodynamic variables in a swine model. By analyzing the relationships between heart sound characteristics and hemodynamic parameters, we aimed to explore the potential of heart sounds as non-invasive indicators of cardiovascular function.
Heart sounds can potentially provide a non-invasive monitoring method to differentiate the cause of hemodynamic variations.
For more details refer to our paper: [[Link to paper](https://rdcu.be/dEQil)]

# Citation
If you use our code in your research, please cite our work published by Park YS, Kim HS, Lee SA, Hwang GS, Jung W, Moon B, Kang KM, Seo WY, Song JG, Kim SH. Correlations between heart sound components and hemodynamic variables. Sci Rep. 2024 Apr 13;14(1):8602. doi: 10.1038/s41598-024-59362-3.

Code release DOI: [![DOI](https://zenodo.org/badge/779175198.svg)](https://zenodo.org/doi/10.5281/zenodo.10906818)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# Instructions for use
Inside the data folder, you'll find sample data for the dobutamine and esmolol cases.
`code/signalprocessing.py` contains functions for processing heart sounds, arterial blood pressure (ABP), and electroencephalogram (ECG).
`code/Demo.ipynb` is a notebook containing code to generate representative plots of variations in hemodynamic status and heart sounds after administration of dobutamine and esmolol.
Make sure to install the following requirements:
```
numpy
scipy
matplotlib
joblib
```

