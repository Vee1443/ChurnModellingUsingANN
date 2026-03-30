# Churn Modelling using Artificial Neural Networks
This project builds an Artificial Neural Network (ANN) to predict customer churn for a retail bank using geo‑demographic and transactional features. The goal is to estimate the probability that a given customer will leave the bank and to rank customers by churn risk so the bank can proactively target retention campaigns.

## Dataset
The project uses the `Churn Modelling` dataset.

**File:** `Churn_Modelling.csv` <br>
**Rows:** 10,000 customers <br>
**Target:** *Exited – 1* if the customer left the bank within 6 months, *0 otherwise* <br>
**Main feature groups:** 
- Geo‑demographic (`Geography`, `Gender`, `Age`, `Tenure`) <br>
- Financial/behavioral: `CreditScore`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary` <br>

## Problem Statement
Given a customer’s profile and account information at time `t`, predict the probability that they will churn (leave the bank) in the next 6 months. Using these probabilities, the bank can:
- Identify high‑risk customers.
- Prioritize retention offers.
- Reduce overall churn and its associated loss in revenue.

This is a supervised binary classification problem with a probabilistic output (churn probability between 0 and 1).

## Approach
**1. Data Preprocessing**
   - Load dataset and separate features/target:
       - `X = dataset.iloc[:, 3:-1].values`
       - `y = dataset.iloc[:, -1].values`
   - Encode categorical variables:
       - `Gender` encoded via `LabelEncoder`.
       - `Geography` encoded via `OneHotEncoder` using `ColumnTransformer`, keeping all other columns.
   - Train–test split:
       - `train_test_split(X, y, test_size=0.3, random_state=1)` (70% train, 30% test).
   - Feature scaling:
       - Standardization with StandardScaler on all features.
           - `X_train = sc.fit_transform(X_train)`
           - `X_test = sc.transform(X_test)`

**2. Model Architecture**
   - An ANN is built using TensorFlow/Keras Sequential:
       - Input: Encoded and scaled feature vector (including one‑hot geography and numerical features).
       - Hidden layers:
           - Dense layer with 6 units, relu activation.
           - Dense layer with 6 units, relu activation.
       - Output layer: Dense layer with 1 unit, sigmoid activation to output churn probability.
       - The model is compiled with a binary classification setup (binary cross‑entropy loss and a suitable optimizer, e.g., adam) and trained on the training set.
   - Encode categorical variables:
       - `Gender` encoded via `LabelEncoder`.
       - `Geography` encoded via `OneHotEncoder` using `ColumnTransformer`, keeping all other columns.
   - Train–test split:
       - `train_test_split(X, y, test_size=0.3, random_state=1)` (70% train, 30% test).
   - Feature scaling:
       - Standardization with StandardScaler on all features.
           - `X_train = sc.fit_transform(X_train)`
           - `X_test = sc.transform(X_test)`

**3. Evaluation**
   - After training, the model:
       - Predicts churn probabilities for the test set.
       - Converts probabilities to binary predictions via a threshold (e.g., `0.5`).
       - Evaluates performance using:
           - Confusion matrix via `confusion_matrix(y_test, y_pred)`.
           - Accuracy via `accuracy_score(y_test, y_pred)`.
       - **Output layer:** Dense layer with 1 unit, sigmoid activation to output churn probability.
       - These metrics summarize how well the ANN distinguishes between churners and non‑churners on unseen data.

**4. Churn Probability Ranking**
   - Because the output is a probability between 0 and 1, customers can be sorted by predicted churn risk.
   - Typical usage:
       - Compute `y_proba = model.predict(X_test)` (probabilities).
       - Create a table joining customer IDs (from the original dataset) with their churn probabilities.
       - Sort descending on probability to produce a ranked churn‑risk list that the bank can use for targeted retention campaigns.


## Technologies Used
- Python
- `NumPy`, `Pandas` – data handling and preprocessing.
- `scikit‑learn`:
  - `LabelEncoder`, `OneHotEncoder`, `ColumnTransformer`
  - `train_test_split`, `StandardScaler`
  - `confusion_matrix`, `accuracy_score`
- `TensorFlow` / `Keras` – building and training the ANN model.
- Jupyter Notebook – experimentation and documentation.

## How to Run
- Clone the repository and move into the project directory.
- Create and activate a virtual environment (optional but recommended).
- Install dependencies and launch Jupyter.
- Open file and run the cells in order. Make sure `Churn_Modelling.csv` is in the same directory as the notebook.
