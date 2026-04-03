<!--# Credit Risk Analysis

- Build and train classification and/or regression models from the dataset in any suitable programming environment of your choosing (e.g., MATLAB) using three machine learning techniques of your choice.

- Justify the rationale behind the choice of your dataset, machine learning techniques, and programming environment.

- Compare and contrast the performance of the three machine learning techniques in terms of prediction or validation accuracy, training time, prediction speed, R-squared values, MSE values, and transparency (as may be applicable).

- Analyse the error matrices, the ROCs (and AUCs) for all three methods (as may be applicable).

- Comment on how the hyperparameters (if any) are tuned or optimized (if applicable) to enhance the built/trained models.

- Submit a report showing the work carried out. Report like what u do after every module.
  
**Insightful approach to assist me while building my model**

- Is the data normally distributed or skewed?
- What's the gender characteristics?
- Does the Age good indicator to measure the Risk Level vs Credit amount?
- What's the correlation between the features of the dataframe?

Ends Here!
-->

# 🏦 Credit Risk Classification — Applied Machine Learning Project

An end-to-end machine learning pipeline that classifies bank customers into **three credit risk levels** (Low, Medium, High) using the **German Credit Risk Dataset**. The project covers the full data science lifecycle: data cleaning, KNN-based missing value imputation, K-Means clustering to engineer the target variable, and benchmarking three classification models — Logistic Regression, SVM, and Random Forest.

---

## 📌 Project Overview

Banks assess loan applicants to estimate credit risk before approving funding. This project builds and evaluates three supervised classification models to predict whether a customer poses a **Low**, **Medium**, or **High** credit risk — using demographic and financial features from 1,000 customer records.

**Key design decisions:**
- The `Risk Level` target variable does not exist in the raw dataset — it is **engineered using K-Means clustering** (k=3)
- Missing values in `Saving accounts` and `Checking account` are **imputed using KNN classification** rather than simple mean/mode filling, to avoid bias
- Three models (LR, SVM, RF) are trained and compared using accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC curves

---

## 📁 Project Structure

```
├── ML___Applied_Machine_Learning_Project_.ipynb   # Full notebook
├── german_credit_data.csv                         # Source dataset
└── README.md
```

---

## 🗂️ Dataset

**Source:** UCI German Credit Risk Dataset via Kaggle  
**Citation:** UCI, 2014 — German Credit Risk. Kaggle. https://www.kaggle.com/datasets/uciml/german-credit  
**Size:** 1,000 observations, 10 features

| Feature | Type | Description |
|---|---|---|
| `Age` | Numeric | Customer age |
| `Sex` | Categorical | Male / Female |
| `Job` | Ordinal (0–3) | Skill level: unskilled non-resident → highly skilled |
| `Housing` | Categorical | Own / Rent / Free |
| `Saving accounts` | Categorical | Little / Moderate / Quite Rich / Rich |
| `Checking account` | Categorical | Numeric (DM) |
| `Credit amount` | Numeric | Loan amount in Deutsche Mark |
| `Duration` | Numeric | Loan duration in months |
| `Purpose` | Categorical | Car, education, furniture, etc. |

---

## 🔄 Full Pipeline

### 1. Exploratory Data Analysis
- Distribution analysis via histograms — identified right skew in Age, Credit amount, Duration
- Box plots by gender for outlier detection
- Correlation heatmap across all features
- Pie charts for Job skill distribution and loan Purpose breakdown
- Bar chart: total credit amount by job skill level

### 2. Data Pre-processing

**Missing values** (18.3% of Saving accounts, ~39% of Checking account):  
Rather than using mean/mode imputation, **KNN Classifier** was used to predict missing values based on the patterns of complete records — preserving the distribution integrity of both columns.

**Encoding:**
- Label Encoding applied to `Sex`, `Housing`, `Saving accounts`, `Checking account`
- `Gender` binary flag (1 = Male, 0 = Female) derived from `Sex`
- `Purpose` dropped after EDA (categorical with too many classes)

**Outlier removal:**
- Record at index 756: 74-year-old with no job and unusually low credit — removed
- Record at index 914: unusually high credit amount (18,424 DM) inconsistent with peer group — removed

### 3. Target Variable Engineering — K-Means Clustering
Since the dataset has no credit risk label, **K-Means clustering** (k=3, determined by the Elbow Method) was applied to generate the `Risk Level` target:

| Label | Risk Level |
|---|---|
| 0 | Low Risk |
| 1 | Medium Risk |
| 2 | High Risk |

An `AgeGroup` feature was also engineered (bins: 19–29, 30–40, 40–50, 51–60, 60–76) for exploratory analysis of risk by age cohort.

---

## 🤖 Models & Results

Three classifiers were trained on a 75/25 train/test split using features: `Age`, `Gender`, `Job`, `Housing`, `Credit amount`, `Duration`.

### Model Comparison

| Risk Level | Model | Accuracy | Precision | Recall | F1-Score | True Positives |
|---|---|---|---|---|---|---|
| Low | Logistic Regression | 96.70% | 96% | 97% | 96% | 171 |
| Medium | Logistic Regression | 99% | 79% | 85% | 81% | 11 |
| High | Logistic Regression | 94.90% | 86% | 82% | 84% | 49 |
| Low | **SVM** | **100%** | **100%** | **100%** | **100%** | 179 |
| Medium | **SVM** | — | 93% | — | — | 13 |
| High | **SVM** | — | — | — | — | 57 |
| Low | Random Forest | 100% | 100% | 100% | 100% | — |
| Medium | Random Forest | — | 86% | — | — | — |
| High | Random Forest | 100% | 100% | — | 100% | — |

**Winner: SVM (Linear Kernel)** — highest overall F1-scores across all three risk classes, most robust to noisy and imbalanced data.

### Key Findings
- **Logistic Regression** (linear model): sensitive to class imbalance; strong on Low Risk but weaker on Medium and High
- **SVM** (linear kernel): best performer overall; handles noisy, imbalanced data well; Low Risk precision = 100%
- **Random Forest** (100 estimators): feature importance revealed `Credit amount` and `Duration` as the two dominant predictors; model retrained on just these two features with comparable accuracy

### Feature Importance (Random Forest)
Top features by importance:
1. `Credit amount`
2. `Duration`
3. `Age`
4. `Job`
5. `Housing`
6. `Gender`

---

## 📊 Evaluation Artifacts

For each model, the following were generated:
- **Classification Report** (precision, recall, F1-score per class)
- **Confusion Matrix** (heatmap)
- **Multi-class ROC Curve** with AUC per class (binarized using one-vs-rest)
- **Decision Tree visualization** (Random Forest, estimator index 5)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Preprocessing, KNN, K-Means, LR, SVM, RF, metrics |
| `sklearn.preprocessing` | Label encoding, label binarization |
| `sklearn.metrics` | Accuracy, classification report, confusion matrix, ROC-AUC |

---

## 🚀 How to Run

1. Place `german_credit_data.csv` in the same directory as the notebook
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook "ML___Applied_Machine_Learning_Project_.ipynb"
   ```
4. Run all cells sequentially — the pipeline is ordered and stateful

---

## 📝 Notes & Limitations

- The `Risk Level` target is **self-generated via clustering**, not a ground-truth label — results should be interpreted accordingly
- `Purpose` column was dropped due to high cardinality; future work could apply one-hot encoding to include it
- The dataset is **gender-imbalanced** (more male records), which may affect model fairness across gender groups
- KNN imputation was applied to `Saving accounts` (k=4) and `Checking account` (k=3) separately, chosen to match their respective number of unique categories
- SVM training time is higher than LR and RF at this dataset size; for larger datasets, kernel approximation methods should be considered

---

## 📚 References

1. Sen et al., "A Weighted kNN approach to estimate missing values," SPIN 2016.
2. Faisal & Tutz, "Multiple imputation using nearest neighbor methods," *Information Sciences*, 2021.
3. UCI German Credit Risk Dataset — https://www.kaggle.com/datasets/uciml/german-credit

---

## 👤 Author

Built as a submission for the **Applied Machine Learning** module — covering classification, clustering, imputation, and model evaluation.
