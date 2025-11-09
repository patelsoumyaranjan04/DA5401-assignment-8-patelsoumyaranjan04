# DA5401 — Assignment 8

- **Name**: Soumya Ranjan Patel  
- **Roll Number**: DA25M029




## Repository Structure

```
DA5401-assignment-8-patelsoumyaranjan04/
├── DA5401_A8.ipynb
└──hour.csv
└── README.md 
```

##  Objective

This assignment demonstrates how **ensemble learning** methods — **Bagging**, **Boosting**, and **Stacking** — can improve regression performance through **bias-variance trade-offs** and **model diversity**.

---

##  Dataset

- **Name:** Bike Sharing Demand (UCI ML Repository)  
- **Samples:** ~17,000 hourly records  
- **Target Variable:** `cnt` (total count of rented bikes per hour)  
- **Citation:**  
  Fanaee-T, Hadi, and Gamper, H. (2014). *Bike-sharing data set.* UCI Machine Learning Repository.  
- **Download:** [UCI ML Repository – Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)

---
## ⚙️ Notebook Workflow Overview

### **1️⃣ Data Preprocessing and Baseline**
- Load `hour.csv`
- Drop irrelevant columns: `instant`, `dteday`, `casual`, `registered`
- Create datetime feature and perform **time-aware train/test split**
- Apply preprocessing pipeline using `ColumnTransformer`:
  - **Numeric:** median imputation + scaling
  - **Categorical:** mode imputation + one-hot encoding
- Train baseline models:
  - `DecisionTreeRegressor(max_depth=6)`
  - `LinearRegression`
- Evaluate RMSE and choose the better model as baseline.

| Model | RMSE |
|--------|------|
| Decision Tree (depth=6) | 158.71 |
| Linear Regression | 133.85 |

---

### **2️⃣ Bagging — Variance Reduction**
- **Hypothesis:** Bagging reduces model **variance** by averaging multiple weak learners.
- Implemented `BaggingRegressor` using Decision Tree (50 estimators).
- Compared RMSE with single Decision Tree.

| Model | RMSE |
|--------|------|
| Decision Tree | 158.71 |
| Bagging (50) | **155.43** |

**Observation:**  
Small but consistent improvement confirms variance reduction; limited due to already-shallow trees.

---
### **3️⃣ Boosting — Bias Reduction**
- **Hypothesis:** Boosting reduces **bias** by sequentially correcting residual errors.
- Implemented tuned `GradientBoostingRegressor` with grid search over:
  - `n_estimators ∈ {100,200}`
  - `learning_rate ∈ {0.05,0.1}`
  - `max_depth ∈ {3,4}`

| Model | RMSE |
|--------|------|
| Gradient Boosting (tuned) | **92.59** |

**Observation:**  
Substantial improvement over Bagging and single models; confirms bias reduction and superior generalization.

---

### **4️⃣ Stacking — Model Combination**
- **Hypothesis:** Stacking combines **diverse learners** to leverage complementary strengths.
- Base learners:
  - `KNeighborsRegressor`
  - `BaggingRegressor`
  - `GradientBoostingRegressor`
- Meta-learner: `RidgeRegression`

| Model | RMSE |
|--------|------|
| Stacking (KNN + Bagging + GBR) | **116.79** |

**Observation:**  
Stacking outperforms Bagging and Linear Regression; diversity among learners improves robustness though not as powerful as Boosting.

---
### **5️⃣ Comparative Summary**

| Model | RMSE | Rank |
|--------|------|------|
| Gradient Boosting (tuned) | **92.59** | 1 |
| Stacking (KNN + Bagging + GBR) | **116.79** | 2 |
| Linear Regression | **133.85** | 3 |
| Bagging (50) | **155.43** | 4 |
| Decision Tree (depth=6) | **158.71** | 5 |

---

### **6️⃣ Bias–Variance Trade-off**

| Technique | Bias | Variance | Typical Behavior | Observed Outcome |
|------------|------|-----------|------------------|------------------|
| Decision Tree | Low | High | Overfits | Worst RMSE |
| Bagging | Same Bias | Lower Variance | Smoother predictions | Small gain |
| Boosting | Lower Bias | Controlled Variance | Residual learning | Best result |
| Stacking | Medium Bias | Medium Variance | Combines models | Second best |
| Linear Regression | High Bias | Low Variance | Underfits | Moderate |

---

### **7️⃣ Visualization Story (Seven Commandments)**

- **Parity Plot:** Predicted vs true bike counts show near-diagonal alignment → good generalization.  
- **Residual Histogram:** Symmetric around zero with mild right skew → minimal bias.  
✅ Inclusive colors, axis labels with units, explicit linear scale, legends, titles, and short descriptive captions.

![placeholder for parity plot](./images/parity_plot.png)  
![placeholder for residuals plot](./images/residual_hist.png)

---
### **8️⃣ Why the Stacking (Best Ensemble) Outperformed the Baseline**

The **Stacking Regressor** achieved lower RMSE than any single model because it optimally balanced **bias** and **variance** by combining **diverse learners**.

- **Bias–Variance Explanation:**  
  Single models are limited — Decision Trees are high-variance, Linear Regression is high-bias.  
  Stacking combines both low-bias (trees) and low-variance (Ridge, KNN) learners.  
  The meta-learner re-weights predictions to minimize total generalization error.
  Averaging uncorrelated predictions reduces variance while complementary bias profiles lower bias overall.

- **Model Diversity Explanation:**  
  Each base learner captures different data patterns — local (KNN), global (GBR), and stabilized variance (Bagging).  
  Their combination provides richer hypothesis coverage, improving robustness and predictive accuracy.

---

### **9️⃣ Conclusions**

- **Bagging** → Reduced variance, minor gain.  
- **Boosting** → Reduced bias, major improvement.  
- **Stacking** → Leveraged diversity for balanced generalization.  
- **Best Performer:** `GradientBoostingRegressor` (RMSE = 92.59).  
- Results fully align with ensemble theory.

---
