# Machine Learning & Deep Learning for Machine Anomaly Detection

This project explores **machine failure and anomaly detection** using
the AI4I 2020 Predictive Maintenance dataset. It contains two
complementary approaches:

-   **Machine Learning (ML):** supervised classification and
    isolation-based anomaly detection.
-   **Deep Learning (DL):** an autoencoder trained on normal
    machine-operation data to identify unusual behaviour through
    reconstruction error.

The work is organized into two notebooks.

## Project Notebooks

  --------------------------------------------------------------------------------------
  Notebook                    Approach                Main objective
  --------------------------- ----------------------- ----------------------------------
  `anomaly-detection.ipynb`   Machine Learning        Predict machine failures and
                                                      compare
                                                      classification/anomaly-detection
                                                      models.

  `iot-anomaly.ipynb`         Deep Learning           Learn normal sensor patterns and
                                                      flag records with unusually high
                                                      reconstruction error.
  --------------------------------------------------------------------------------------

## Dataset

The notebooks use the **AI4I 2020 Predictive Maintenance Dataset**.

-   **Total records:** 10,000
-   **Total columns:** 14
-   **Target:** `Machine failure`
-   **Original target distribution:**
    -   Normal operation (`0`): 9,661
    -   Machine failure (`1`): 339

The dataset includes machine operating measurements such as:

-   Air temperature
-   Process temperature
-   Rotational speed
-   Torque
-   Tool wear

It also includes machine-failure mode indicators: TWF, HDF, PWF, OSF and
RNF.

## 1. Machine Learning Approach

Notebook: `anomaly-detection.ipynb`

### Exploratory Data Analysis

The notebook explores the dataset using descriptive statistics and
visualizations, including:

-   Class distribution
-   Correlation heatmap
-   Temperature distributions
-   Torque and rotational-speed boxplots
-   Failure-mode distributions

### Preprocessing

1.  Checked the dataset for missing values (none were found).
2.  Removed the `UDI` and `Product ID` identifier columns.
3.  Encoded the categorical `Type` feature using `LabelEncoder`.
4.  Separated the features from the `Machine failure` target.
5.  Applied SMOTE to address the imbalanced target classes.
6.  Split the resampled data into training and testing sets using an
    80/20 stratified split.
7.  Standardized the features using `StandardScaler`, fitting the scaler
    on the training data and applying it to the test data.

### How the ML Dataset Was Balanced

The original dataset was highly imbalanced: it contained 9,661 normal
records and only 339 failure records.

To balance the target classes, the notebook uses **SMOTE (Synthetic
Minority Over-sampling Technique)**. SMOTE generates synthetic
minority-class examples based on neighbouring samples; it does not
simply duplicate every failure row.

  Class             Before SMOTE   After SMOTE
  --------------- -------------- -------------
  Normal (`0`)             9,661         9,661
  Failure (`1`)              339         9,661
  **Total**           **10,000**    **19,322**

After balancing, the data is split into: - Training set: 15,457
records - Testing set: 3,865 records

**Methodology note:** In this notebook, SMOTE is applied before the
train-test split. For a more rigorous evaluation, split the original
data first and apply SMOTE only to the training data. This prevents
synthetic samples related to test observations from influencing model
training.

### ML Models

**Isolation Forest** - Unsupervised anomaly-detection algorithm. - Uses
`contamination=0.05`. - Predictions are converted to normal (`0`) and
anomaly (`1`) labels.

**Random Forest Classifier** - Supervised classification algorithm. -
Uses 200 decision trees (`n_estimators=200`). - Uses
`class_weight="balanced"`.

### Evaluation

The notebook evaluates predictions with: - Classification report
(precision, recall and F1-score) - Confusion matrix - Random Forest
feature importance

It also visualizes sensor measurements and machine-failure observations.

## 2. Deep Learning Approach

Notebook: `iot-anomaly.ipynb`

### Normal-Only Training Strategy

Unlike the ML notebook, the Deep Learning notebook does **not** balance
the classes with SMOTE.

Instead, it follows a normal-only training strategy:

1.  Separates normal records (`Machine failure = 0`) from failure
    records (`Machine failure = 1`).
2.  Selects five sensor features: air temperature, process temperature,
    rotational speed, torque and tool wear.
3.  Splits the normal records into training and validation sets (80/20).
4.  Fits `StandardScaler` on the normal training data and uses it to
    transform the validation and failure data.
5.  Trains the autoencoder using only normal training samples.

This approach teaches the network what normal machine behaviour looks
like, so unusual records can be identified by their reconstruction
errors.

### Autoencoder Architecture

The notebook implements a fully connected autoencoder using
**TensorFlow/Keras**.

  Part      Layers
  --------- --------------------------------------------------------
  Encoder   Input (5 features) → Dense (4, ReLU) → Dense (2, ReLU)
  Decoder   Dense (4, ReLU) → Dense (5, Linear)

**Training configuration:** - Optimizer: Adam - Loss: Mean Squared Error
(MSE) - Epochs: 50 - Batch size: 32 - Validation: normal validation
samples, with inputs and targets being the same

### Anomaly Detection with Reconstruction Error

The autoencoder attempts to reconstruct each input sensor record.

The notebook calculates the mean squared reconstruction error for each
record and sets the anomaly threshold to the **90th percentile of the
normal validation errors**.

-   Error above the threshold → **Anomaly**
-   Error at or below the threshold → **Normal**

The notebook compares reconstruction errors for normal validation
records and machine-failure records, plots their distributions, and
creates a confusion matrix.

The threshold produced in the notebook is approximately `0.5398`. On the
evaluated normal-validation and failure records, the displayed confusion
matrix is:

``` text
[[1739  194]
 [ 157  182]]
```

This result shows that the reconstruction-error method identifies some
failure records, but it also produces false positives and false
negatives. It should be treated as an experimental result, not as a
production-ready failure detector.

## ML vs. DL: Key Difference

  -----------------------------------------------------------------------
  Aspect                  Machine Learning        Deep Learning notebook
                          notebook                
  ----------------------- ----------------------- -----------------------
  Main methods            Isolation Forest,       Dense Autoencoder
                          Random Forest           

  Learning strategy       Unsupervised anomaly    Normal-only
                          detection and           reconstruction learning
                          supervised              
                          classification          

  Class balancing         SMOTE balances the ML   No resampling; trains
                          target before splitting only on normal records

  Input features          All prepared features   Five sensor
                          except the target       measurements

  Detection idea          Model predictions       Reconstruction error
                                                  and threshold
  -----------------------------------------------------------------------

## Technologies Used

-   Python
-   Pandas and NumPy
-   Matplotlib and Seaborn
-   Scikit-learn
-   Imbalanced-learn (SMOTE)
-   TensorFlow / Keras

## Running the Notebooks

1.  Clone or download this repository.
2.  Install the required libraries:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow
```

3.  Download the AI4I 2020 Predictive Maintenance Dataset.
4.  Update the dataset path in the notebook to match your local
    environment.
5.  Open and run the notebooks in Jupyter Notebook, JupyterLab, Google
    Colab or Kaggle.

## Repository Structure

``` text
.
├── anomaly-detection.ipynb
├── iot-anomaly.ipynb
└── README.md
```

## Scope and Limitations

This repository is an experimental comparison of ML and DL techniques on
tabular predictive-maintenance data. The dataset is not a live IoT
stream, so the notebooks demonstrate sensor-based anomaly detection
rather than a deployed real-time IoT monitoring system.

The ML notebook's SMOTE-before-split workflow should be revised before
using its test metrics as a final estimate of generalization. The
autoencoder also misses some failure records, so threshold selection and
further evaluation are important next steps.

## Acknowledgements

Dataset: **AI4I 2020 Predictive Maintenance Dataset**.
