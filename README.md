# world-population-analysis
Analysis of 2020 world population data using Python, comparing regression models and PCA to predict annual population growth rates.

## Dataset

The dataset contains 235 countries and dependencies and 12 columns, including population, annual growth rate, population density, land area, migration, fertility rate, median age, and urban population percentage.

**Target variable:** Yearly Change (%)

## Tools

Python, Pandas, NumPy, scikit-learn, Matplotlib, and Seaborn.

## Approach

- Removed percentage signs and thousands separators.
- Converted relevant columns to numeric types.
- Examined missing values.
- Applied row normalization to the predictor variables.
- Split the dataset into 70% training and 30% test data.
- Used 10-fold cross-validation on the training data.
- Compared model performance using Root Mean Squared Error (RMSE).

## Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- Support Vector Regression (SVR)
- Principal Component Regression (PCA followed by Linear Regression)

The project also includes a plot comparing cross-validation RMSE across different numbers of principal components.

## Evaluation

Cross-validation RMSE and held-out test RMSE are calculated for each model. Lower RMSE indicates smaller prediction errors.

## Limitations and Improvements

This is a single-year dataset, so the project does not forecast future population trends.

The original implementation needs improvements to missing-value handling. Imputation and PCA should be fitted within each cross-validation fold using a scikit-learn Pipeline.

Some predictors, such as Net Change and Population 2020, are closely related to the target and should be reviewed for target leakage.

## Project Files

- Python script containing preprocessing, modeling, and evaluation.
- `world-population-by-country-2020.csv`: input dataset.

The CSV must be in the working directory when running the code. The script was written in a notebook style; metric tables can be displayed in Jupyter or printed when running it as a Python script.

## Author

Yasemin Aleyna Yilmaz
