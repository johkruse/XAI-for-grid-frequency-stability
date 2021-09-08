# Explainable Machine Learning for power grid frequency stability

Code accompanying the mansucript "Revealing drivers and risks for power grid frequency stability with explainable AI".
Preprint: <https://arxiv.org/abs/2106.04341>


## Install

The code is written in Python (tested with python 3.7). To install the required dependencies execute the following commands:

```[python]
python3.7 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Usage

The `scripts` folder contains scripts to create the paper results and `notebooks` contains a notebook to reproduce the paper figures. The `utils` folder comprises two modules for processing ENTSO-E data and grid frequency stability indicators.

The `scripts` contain a pipeline of six different stages:

* `1_download.sh`: A bash script to download the external features from the ENTSO-E Transparency Platform. 
* `2_stability_indicator_prep.py`: Create HDF files from grid frequency CSV file and then extract frequency stability indicators.
* `3_entsoe_data_prep.py`: Collect and aggregate external features within each synchronous area.
* `4_external_feature_prep.py`: Add additional engineered features to the set of external features.
* `5_train_test_split.py`: Split data set into train and test set and save data in a version folder.
* `6_model_fit.py`: Fit the XGBoost model, optimize hyper-parameters and calculate SHAP values.


## Input data and results
All the raw data is publicly available and we have uploaded the processed data and our results on zenodo. The data and the (intermediate) results can be used to run the scripts.

* **External features, frequency stability indicators and results of hyper-parameter optimization and model interpretation**: The output of scripts 2 to 6 are available on [zenodo](https://zenodo.org/record/5118352). The data is assumed to reside in the repository directory within `./data/` and the results should reside in `./results/`. In particular, the data of external features and stability indicators can be used to re-run the model fit. 
* **Raw grid frequency data**: We have used pre-processed [grid frequency data](https://zenodo.org/record/5105820) as an input to `2_stability_indicator_prep.py`.  The CSV files from the repository are assumed to reside in `../Frequency_data_base/` relative to this code repository.  The frequency data is originally based on publicly available measurements from [TransnetBW](https://www.transnetbw.de/de/strommarkt/systemdienstleistungen/regelenergie-bedarf-und-abruf).
* **Raw ENTSO-E data**: The output of `1_download_data.sh` is not available on the zenodo repository, but can be downloaded from the [ENTSO-E Transparency Platform](transparency.entsoe.eu/) via the bash script. The ENTSO-E data is assumed to reside in `../../External_data/ENTSO-E` relative to this code repository.

