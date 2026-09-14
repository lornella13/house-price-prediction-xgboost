House Prices Prediction with XGBoost
A machine learning project that predicts house sale prices using the classic Kaggle House Prices: Advanced Regression Techniques dataset (Ames, Iowa housing data 1,460 training observations, 80 features).

Overview
The pipeline in XG-boost.ipynb covers the full workflow from raw data to a Kaggle submission:

Exploratory Data Analysis  summary statistics and visual inspection of the target variable and key predictors
Target Variable Analysis SalePrice is heavily right-skewed, so it is log-transformed (log1p) to normalize the distribution and stabilize model training
Missing Value Handling features with meaningful “missingness” (PoolQC, MiscFeature, Alley, Fence, garage/basement attributes) are filled with "None"; numeric features are filled with 0
Outlier Removal — two extreme observations (GrLivArea > 4000 with very low sale price) are dropped, leaving 1,458 training rows
Feature Engineering — derived features including:
TotalSF (total square footage across floors)
HouseAge, RemodAge (temporal features)
TotalBath (aggregate bathroom count)
HasGarage and similar indicator flags
Encoding  one-hot encoding of categorical features via pd.get_dummies(drop_first=True)
Modeling  Ridge regression as a regularized linear baseline, then XGBoost (gradient boosted trees)
Evaluation  5-fold cross-validation on the log-transformed target (RMSE)
Prediction predictions transformed back to dollar scale with np.expm1 and exported to submission.csv
Results
Model	CV RMSE (log scale)
Ridge regression	~0.1150
XGBoost	~0.1135
A feature-importance analysis (top 15 features) highlights the strongest predictors, led by overall quality, total square footage, and living area.

Tech Stack
Python 3 — pandas, numpy, matplotlib, scikit-learn, XGBoost
Jupyter Notebook
Usage
Download the dataset from Kaggle and place train.csv / test.csv alongside the notebook
Install dependencies:
pip install pandas numpy matplotlib scikit-learn xgboost
Open and run XG-boost.ipynb — it produces submission.csv ready for Kaggle upload
