# Machine Learning: Project 1 - Higgs Boson

Project 1 of the Machine Learning course given at the EPFL Fall 2021.

## Stategy

The main strategy is the following:

- Split the train and the test set in 3 subsets:
    - `JET = 0`
    - `JET = 1`
    - `JET >= 2`
- Clean the subsets individually:
    - Remove the columns with the same data at each row.
    - Replace -999 by the median of the column.
    - Standardize the columns using the mean and the standard deviation of the train dataset.
- Expand the features using polynomial expansion. The degree is determined using cross validation.
- Perform ridge regression for each subset.
- Predict the labels for each test subset using the model determined with each train subset.
- Merge the results.

## Results

Model | Accuracy
--- | ---
Gradient Descent | **TODO**
Stochastic Gradient Descent | **TODO**
Least squares | 0.824
Ridge regression | 0.828
Logistic regression | **TODO**
Regularized logistic regression | **TODO**
