# flake8: noqa: E402
import os
import sys
import zipfile
import numpy as np

# Add src to path to import implementations
sys.path.append('../src')

# Import functions
from proj1_helpers import (create_csv_submission, load_csv_data, predict_labels,
                           standardize)
from implementations import logistic_regression

DATA_DIRECTORY = '../data'
DATA_TRAIN_PATH = os.path.join(DATA_DIRECTORY, 'train.csv')
DATA_TEST_PATH = os.path.join(DATA_DIRECTORY, 'test.csv')

OUTPUT_DIRECTORY = '../out'
OUTPUT_PATH = os.path.join(OUTPUT_DIRECTORY,
                           'predictions_logistic_regression.csv')

# Extract archives if needed
for csv_filename in (DATA_TRAIN_PATH, DATA_TEST_PATH):
    if not os.path.exists(csv_filename):
        zip_filename = csv_filename + '.zip'
        with zipfile.ZipFile(zip_filename, 'r') as zf:
            zf.extractall(DATA_DIRECTORY)

# Create output directory if needed
if not os.path.exists(OUTPUT_DIRECTORY):
    os.mkdir(OUTPUT_DIRECTORY)

# Load the train data
print('1/5 Load train data')
y, tX, ids = load_csv_data(DATA_TRAIN_PATH, label_b=0)
# Standardize data
tX = standardize(tX)

# Logistic regression
print('2/5 Run logistic regression')
initial_w = np.zeros((tX.shape[1], 1))
max_iters = 1000
gamma = 0.01
threshold = 1e-8
w, loss = logistic_regression(y, tX, initial_w, max_iters, gamma,
                              threshold, info=True, sgd=True)
print('w =', w)
print('Loss:', loss)

# Load test data
print('3/5 Load test data')
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
tX_test = standardize(tX_test)

# Generate predictions
print('4/5 Generate predictions')
y_pred = predict_labels(w, tX_test)

# Create submission
print('5/5 Create submission')
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

print(f'File {OUTPUT_PATH} created')
