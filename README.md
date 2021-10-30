# Machine Learning: Project 1 - Higgs Boson

Project 1 of the Machine Learning course given at the EPFL Fall 2021.

## Strategy

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

## Structure

This is the structure of the repository:

- `data`: contains the datasets
- `docs`: contains the documentation
- `figs`: contains the figures (accuracies, confusion matrices, results of cross validation)
- `scripts`: contains the main scripts and the notebooks
    - `main_accuracy_by_jet.py`: compute the accuracy with a classifier
    - `main_cross_validation.ipynb`: perform cross validation
    - `parameters.json`: parameters for optimization algorithms
    - `run.py`: make predictions for AIcrowd using ridge regression
- `src`: source code
    - `clean_data.py`: functions to clean data
    - `cross_validation.py`: functions to perform cross validation
    - `gradient.py`: gradient functions
    - `helpers.py`: utils functions
    - `implementations.py`: Machine Learning algorithms implementations
    - `loss.py`: loss functions
    - `metrics.py`: score and performance functions
    - `plot_utils.py`: plot utils using matplotlib
    - `print_utils.py`: print utils
    - `split_data.py`: split functions to handle data
    - `stats_tests.py`: statistical tests

## Instructions

To reproduce our submission on [AIcrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs), move in the `scripts` folder and run:
```
python3 run.py
```
The csv file produced will be `out/predictions.csv`.


## Scripts

- `main_accuracy_by_jet.py`: compute the accuracy of a classifier. The dataset is splited in 3 subsets.
It loads the parameters of optimization algorithms in the `parameters.json` file. The figures are saved in the `figs` directory.
The program shows:
    - The global accuracy score and the one for each subset.
    - The global confusion matrix and the one for each subset.

Usage:
```
python3 main_accuracy_by_jet.py --clf [CLASSIFIER]
```
Where `CLASSIFIER` can be:
- `gradient_descent`
- `stochastic_gradient_descent`
- `least_squares`
- `ridge_regression` (default)
- `logistic_regression`
- `regularized_logistic_regression`

## Results

Model | Accuracy
--- | ---
Gradient Descent | **TODO**
Stochastic Gradient Descent | **TODO**
Least squares | 0.827
Ridge regression | 0.831
Logistic regression | **TODO**
Regularized logistic regression | **TODO**
