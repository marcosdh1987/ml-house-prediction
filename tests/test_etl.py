import sys
import unittest

import pandas as pd

sys.path.append("./src/mlflow")
import os

from src.mlflow.etl import etl
from src.mlflow.utils.constants import (RAW_FILENAME,
                                        X_TEST_FILENAME, X_TRAIN_FILENAME,
                                        Y_TEST_FILENAME, Y_TRAIN_FILENAME)

# get the execution path
execution_path = os.getcwd()
# join the execution path with the path to the dataset
key = os.path.join(execution_path, RAW_FILENAME)
x_train_filename = X_TRAIN_FILENAME
y_train_filename = Y_TRAIN_FILENAME
x_test_filename = X_TEST_FILENAME
y_test_filename = Y_TEST_FILENAME

# create a etl test class with mock file path
class TestEtl(unittest.TestCase):
    # @patch('src.mlflow.etl.etl')
    def test_etl(self):
        filepath = "src/data/raw/challenge_houses-prices.csv"
        etl(filepath)
        # load the data from the processed folder
        X_train_tf = pd.read_parquet(os.path.join(execution_path, x_train_filename))
        X_test_tf = pd.read_parquet(os.path.join(execution_path, x_test_filename))
        y_train = pd.read_parquet(os.path.join(execution_path, y_train_filename))
        y_test = pd.read_parquet(os.path.join(execution_path, y_test_filename))
        # check the shape of the data
        self.assertEqual(X_train_tf.shape, (40000, 17))
        self.assertEqual(X_test_tf.shape, (10000, 17))
        self.assertEqual(y_train.shape, (40000, 1))
        self.assertEqual(y_test.shape, (10000, 1))


if __name__ == "__main__":
    # get the execution path
    execution_path = os.getcwd()
    unittest.main()
