# Diabetes Prediction using Support Vector Machine (SVM)

A machine learning project that predicts whether a person is diabetic based on diagnostic health measurements, using a Support Vector Machine classifier trained on the PIMA Indians Diabetes Dataset.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Predictive System Example](#predictive-system-example)
- [Tech Stack](#tech-stack)
- [Limitations and Future Work](#limitations-and-future-work)

## Overview

This project builds a binary classifier that labels a patient as **diabetic (1)** or **non-diabetic (0)** using eight medical predictor variables. The notebook walks through the full pipeline: data loading, exploration, standardization, train/test splitting, model training, evaluation, and a simple predictive system for new inputs.

## Dataset

**PIMA Indians Diabetes Dataset** (`diabetes.csv`)

- **Samples:** 768
- **Features:** 8
- **Target:** `Outcome` (0 = Non-Diabetic, 1 = Diabetic)
- **Class distribution:** 500 non-diabetic, 268 diabetic

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function (genetic risk score) |
| Age | Age in years |

Diabetic patients in the data show higher average glucose (about 141 vs 110), more pregnancies, and higher age than non-diabetic patients.

> The dataset is not included in this repository. It is publicly available on Kaggle and the UCI repository as the "PIMA Indians Diabetes Database". Place `diabetes.csv` in the project directory (or update the path in the notebook).

## Project Workflow

1. **Import dependencies:** NumPy, pandas, scikit-learn.
2. **Data collection and analysis:** load the CSV, inspect the shape, summary statistics, and class balance, and compare feature means by outcome.
3. **Data preprocessing:** separate features and labels, then standardize features with `StandardScaler`.
4. **Train/test split:** 80/20 split, stratified on `Outcome`, `random_state=2` (614 training and 154 test samples).
5. **Model training:** `sklearn.svm.SVC` with a **linear kernel**.
6. **Evaluation:** accuracy score on training and test data.
7. **Predictive system:** standardize a new input and classify it with the trained model.

## Results

| Dataset | Accuracy |
|---|---|
| Training | **78.66%** |
| Test | **77.27%** |

The small gap between training and test accuracy suggests the model generalizes reasonably well and is not significantly overfitting.

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas scikit-learn jupyter
```

## Usage

1. Download `diabetes.csv` and place it in the project folder.
2. In the notebook, update the data path, since it currently points to a Google Colab location:
   ```python
   diabetes_dataset = pd.read_csv('diabetes.csv')
   ```
3. Launch Jupyter and run all cells:
   ```bash
   jupyter notebook Project_3_Diabetes_Prediction.ipynb
   ```

The notebook can also be run on Google Colab by uploading `diabetes.csv` to the session.

## Predictive System Example

```python
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize using the scaler fitted on the training data
std_data = scaler.transform(input_data_reshaped)

prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
```

**Output:** `The person is diabetic`

Input order: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age`.

## Tech Stack

- Python 3
- NumPy
- pandas
- scikit-learn
- Jupyter Notebook / Google Colab

## Limitations and Future Work

- The scaler is fitted on the full dataset before splitting, which leaks a small amount of test information into training. Fitting it on the training set only (or using a `Pipeline`) would be cleaner.
- Several columns (e.g. Glucose, BloodPressure, Insulin) contain `0` values that likely represent missing data and could be imputed.
- Only accuracy is reported. Precision, recall, F1, and a confusion matrix would give a fuller picture, especially given the class imbalance.
- Possible improvements: hyperparameter tuning (`C`, kernel, `gamma`), cross-validation, and comparing against other models such as Logistic Regression, Random Forest, or Gradient Boosting.

## Disclaimer

This project is for educational purposes only and is **not** a medical diagnostic tool. Do not use it to make health decisions. Consult a qualified healthcare professional for medical advice.
