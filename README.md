# Data-Driven Degradation Estimation

This repository belongs to an experiment that will be published in the 15th International Conference on Research Challenges in Information Science (RCIS) under the title "Recommendations for Data-Driven Degradation Estimation with Case Studies from Manufacturing and Dry-Bulk Shipping". This repository contains code and the results of 104 experiments to estimate the degradation process for two different case studies each, one from manufacturing and one from dry-bulk shipping.

## Datasets
### Manufacturing
The data set used for the case study is the well-known bearing data set provided by FEMTO-ST institute within PRONOSTIA, an experimental platform dedicated to the testing and validation of bearing failure detection, diagnostic, and prognostic approaches.
The FEMTO bearing data set contains run-to-failure tests of 17 bearings, 6 for training, and 11 for testing that can be found here: https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset. 


### Dry-Bulk-Shipping
The second data set consists of sensor data for 15 vessels, 12 for training and 3 for testing, with data ranging from the beginning of 2016 to the end of 2020. The data set is provided by an anonymous shipping company. 

## Disclaimer

Please note, that the code provided here, is optimized to process bearing data provided by the FEMTO-ST institute within PRONOSTIA. Even though we tried to generalize the code as much as possible, some parts within the code are specific to the two case studies (here specific to the bearing data set). 

Data for the case-study from dry-bulk shipping is confidential and cannot be provided. Hence, we likewise only provide the code specific to the bearing data-set.

## Getting Started

To get started, clone the repository and perform the following steps. Code is tested under Python 3.8. To separate code from other projects, we recommend installing all project dependencies within a virtual environment (virtualenv). Data is not directly provided within the repository but has to be downloaded from the original data repository (see above). Run the following steps from the command line to get started.

1. Install virtualenv using pip by running `pip install virtualenv`. This step is only necessary if virtualenv is not installed yet.

2. Go into the cloned repository.

2. Create an environment to which all project dependencies are going to be installed by running `virtualenv -p C:\Path\to\Python38\python.exe ENV`. Specify the path to your local Python installation (only needed if having multiple Python versions installed). We recommend using Python in version 3.8. The environment is going to be installed under the name `ENV` within the root folder of the project.

3. Activate the installed environment. Depending on your host system the command is slightly different. Under Windows, run `env/Scripts/activate` from the root folder. Under Unix, run `source ENV/bin/activate` from the root folder. Note: To leave/exit the environment again, run `deactivate` from within the activated environment.

4. Install the needed project dependencies within the activated environment by running `pip install -r requirements.txt`.

After the execution of the above steps, the repository is initialized. Continue with the following steps to configure parameters corresponding to your host system.

5. Open the FEMTO-ST data repository from the above link and download all contents of the repository (either clone it or download as zip). The important folders are `Full_Test_Set`, `Learning_set`, and `Test_set`.

6. Place all three downloaded folders within one folder within your filesystem.

7. Go back to the code of this repository and open the `constants.py` file under `.\util`.

8. Adjust the path declaration in line `11` according to the folder in which you have placed the downloaded bearing data. Only provide the path to the root folder of the bearing data itself containing the folders named above. Please note, that paths under windows might have to be encoded, e.g. encode slashes within the path declaration by another slash (`C:\\Your\\Local\\Path\\To\\Bearing\\Data`). We recommend putting data within a `data` folder at the root of the repository itself.

9. Raw data is preprocessed for further analysis. Outputs of the preprocessing step are stored as CSV files within the filesystem. Therefore, create a new folder on your host system at any location and adjust the path declaration in line `13` accordingly. We recommend putting data within a `data` folder at the root of the repository itself.

10. Metrics (results of each experiment) are written throughout the process into a dedicated folder. Create a new folder on your host system at any location and adjust the path declaration in line `17` accordingly. We recommend creating a `metrics_dict` folder at the root of the repository itself. 

11. Depending on the available resources on the host system, intermediate results might not be able to be kept in memory. Memory can be swapped on disk. Create a new folder on your host system at any location and adjust the path declaration in line `21` accordingly. We recommend creating a `memory_cache` folder at the root of the repository itself.

12. Some experiments require outputs of CNN training, which are created in another preprocessing step. Create a new folder on your host system at any location and adjust the path declaration in line `24` accordingly. We recommend creating a `cnn_embedding_layers` folder at the root of the repository itself.

All other parameters within the `constants.py` file can be left to their default.

## Execution

The process is divided into two preprocessing steps and one step for the actual degradation estimation process. Before running the actual prediction, run the two preprocessing steps as follows. To remind again, the Python environment needs to be activated before performing the following steps. To avoid path issues, execute the following steps from the top-level folder of the repository.

1. From the top-level folder of the repository run `py .\pre_processing\__init__.py` to execute the first preprocessing step. To remind again, it is important to have all paths configured properly, as results from this preprocessing step are saved as a file into the processed data set folder.

2. From the top-level folder of the repository run `py .\rul_features\learned_features\supervised\cnn_multiscale_features.py` to execute the second preprocessing step.

3. Perform actual performance degradation estimation by running `py .\models\__init__.py` from the top-level folder of the repository.

## Results

After step 3 (see above steps) ran through, results are found within the previously configured `metrics_dict` folder. Each JSON document within the folder corresponds to one experiment. See below the notation of the experiments. For easier reference, you find the results of our experiments within the `./_results` folder of the repository.

## Running own experiments
This project was originally implemented for the analysis of bearing data from the FEMTO-ST dataset. 

### Adding own data
If you want to use existing methods for data pre-processing your data has to adhere to the format of that data set 
(One folder per experiment, one file per observation during the experiment, with one observation being one time series
of one sensor), otherwise you have to write your own methods for pre-processing, but you can re-use the features
implemented in [rul_features](rul_features/computed_features). If you want to use your own implementation four steps
are required:
1. Implement pre-processing
2. Represent new implementation with a new data set type in [DataSetType.py](models/DataSetType.py).
3. The stored, pre-processed data has to be read during the do_eval method in [evaluation.py](models/evaluation.py).
4. The models you want to use with your data have to be associated with the correct DataSetType during instantiation
   (typically done in [models/\_\_init\_\_.py](models/__init__.py)).

### Adding own models
If you want to use your own models for the evaluation you need to implement them by inheriting the abstract class
[DegradationModel](models/DegradationModel.py) you then need to instantiate them in 
[models/\_\_init\_\_.py](models/__init__.py) and create a models_dict for them.

### Adding own metrics
If you want to use your own metrics during evaluation you need to implement them in [util/metrics.py](util/metrics.py).
They should have the same signature as the other metrics:

```python
def metric(y_actual, y_predicted):
    pass
``` 
You can then add the new metric to the list of metrics in ``do_eval`` method in [evaluation.py](models/evaluation.py) 
under `metrics_list=[rmse, correlation_coefficient]`.

### Adjusting Hyperparameters
All model hyperparameters are set in the respective file in [rul_prediction](rul_prediction).
All feature hyperparameters are set in the respective file in [rul_features](rul_features).
Hyperparameters need to be adjusted manually.

## Notations for Experiments
We choose <img src="https://render.githubusercontent.com/render/math?math=Z_{i,j}">  to be an experiment, where <img src="https://render.githubusercontent.com/render/math?math=$Z\in\{A,\dots,M\}$"> denotes a combination of preprocessing steps listed in the table below, <img src="https://render.githubusercontent.com/render/math?math=$i\in\{\mathrm{true,false}\}$"> denotes if an health stage classifier is used, and <img src="https://render.githubusercontent.com/render/math?math=$j\in\{\text{MLR, GPR, ANN, SVR}\}$"> denotes the selected regression model for prediction, where <img src="https://render.githubusercontent.com/render/math?math=\text{MLR}$"> denotes Multiple Linear Regression, <img src="https://render.githubusercontent.com/render/math?math=\text{GPR}$"> denotes Gaussian Process Regression, <img src="https://render.githubusercontent.com/render/math?math=\text{ANN}$"> denotes Artificial Neural Network and <img src="https://render.githubusercontent.com/render/math?math=\text{SVR}$"> denotes Support Vector Machine.


<img width="442" alt="Bildschirmfoto 2021-01-29 um 11 53 38" src="https://user-images.githubusercontent.com/35696618/106266420-b0d69980-6228-11eb-9a6f-d4917cff2610.png">


For example, the notation <img src="https://render.githubusercontent.com/render/math?math=$E_{\mathrm{true},\mathrm{ANN}}$ "> represents the application of health stage classifier, followed by frequency analysis and statistical feature extraction, with a final regression of degradation using ANN. 
