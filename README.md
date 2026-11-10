# Rental Price Prediction Using Machine Learning

**Python | Machine Learning | Data Analysis | Regression**

## Overview

This project explores the prediction of apartment rental prices using machine learning techniques. It focuses on data exploration, preprocessing, feature engineering, and the implementation and evaluation of regression algorithms.

A key part of the project is implementing K-Nearest Neighbors (KNN) and Decision Tree regression algorithms from scratch and comparing their performance with Scikit-learn implementations.

## Objectives

- Explore and analyze apartment rental listings.
- Identify and preprocess numerical and categorical features.
- Handle missing values, outliers, and inconsistent data.
- Implement and evaluate regression algorithms.
- Compare custom implementations with library-based models.
- Investigate how model parameters influence prediction accuracy.

## Dataset

The project uses the **Apartments for Rent Classified 10K** dataset, containing 10,000 apartment rental listings.

The dataset includes property characteristics such as:

- Rental price
- Number of bedrooms and bathrooms
- Property size in square feet
- City and state
- Geographic coordinates
- Property type and amenities

**Prediction target:** Rental price (`price`).

The original CSV file is not included in this repository. To reproduce the analysis, place `apartments_for_rent_classified_10K.csv` in a `data/` directory.

## Methodology

### 1. Exploratory Data Analysis

The initial analysis examines feature types, descriptive statistics, missing values, distributions, and relationships between variables.

### 2. Data Preprocessing

The preprocessing workflow includes:

- Missing-value analysis and imputation using median and mode values
- Identification and treatment of outliers using the Interquartile Range (IQR)
- Feature selection and removal of irrelevant attributes
- Standardization of numerical features
- One-hot encoding of categorical features
- Splitting data into training (80%) and testing (20%) sets

### 3. Machine Learning Models

**K-Nearest Neighbors (KNN)**

A custom KNN regressor was implemented using Euclidean distance and nearest-neighbor selection. It was compared against Scikit-learn's `KNeighborsRegressor` using different values of k (3, 5, 7, and 9).

**Decision Tree Regression**

A custom Decision Tree regressor was developed to predict rental prices through recursive feature-based splitting. Different tree depths and minimum leaf sizes were evaluated and compared with Scikit-learn's `DecisionTreeRegressor`.

**Random Forest Regression**

Scikit-learn's `RandomForestRegressor` was used to investigate ensemble learning. Experiments varied the number of trees, maximum depth, and minimum samples per leaf.

### 4. Model Evaluation

Regression performance was evaluated using:

- **MAE:** Mean Absolute Error
- **MAPE:** Mean Absolute Percentage Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination

Graphs and comparison tables were used to analyze prediction accuracy and parameter sensitivity.

## Results

Selected results from the experiments:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Custom KNN (k=3) | 222.12 | 314.97 | 0.720 |
| Scikit-learn KNN (k=3) | 221.81 | 314.50 | 0.720 |
| Random Forest (100 trees, depth=15) | 209.27 | 290.69 | 0.761 |

The custom KNN implementation produced results close to its Scikit-learn counterpart. Among the tested configurations, Random Forest with 100 trees, maximum depth 15, and minimum leaf size 5 achieved the highest R² score.

These results demonstrate the importance of preprocessing, appropriate algorithm selection, and hyperparameter configuration.

## Technologies Used

- **Python** — programming and algorithm implementation
- **NumPy** — numerical computations
- **Pandas** — data manipulation and analysis
- **Matplotlib** — data visualization
- **Scikit-learn** — preprocessing, regression models, and evaluation
- **Jupyter Notebook** — interactive development and experimentation

## How to Run

1. Clone the repository:

   `git clone https://github.com/tamyeg/rental-price-prediction.git`

2. Navigate to the project directory:

   `cd rental-price-prediction`

3. Install the required libraries:

   `pip install numpy pandas matplotlib scikit-learn notebook`

4. Create a `data` folder and place the dataset inside it with the filename:

   `apartments_for_rent_classified_10K.csv`

5. Start Jupyter Notebook:

   `jupyter notebook`

6. Open `rental-price-prediction.ipynb` and execute the cells in order.

## Key Learning Outcomes

Through this project, I gained practical experience in:

- Implementing regression algorithms without relying entirely on machine learning libraries.
- Understanding distance calculations and recursive decision-tree splitting.
- Preparing mixed numerical and categorical data for machine learning.
- Comparing models using multiple regression metrics.
- Investigating the effects of hyperparameters on performance.
- Interpreting experimental results and visualizations.

## Author

**Tamara Yegorova**

MSc Artificial Intelligence in Computer Science  
Kaunas University of Technology

[GitHub](https://github.com/tamyeg) | [LinkedIn](https://www.linkedin.com/in/tamara-yegorova/)
