import unittest
import pandas as pd
import numpy as np  # <--- ADD THIS LINE
from classification import load_and_prepare_data, train_model, evaluate_model

class TestClassificationPipeline(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.encoder = load_and_prepare_data("doctor31_cazuri_cleaned.csv")

    def test_data_shape(self):
        self.assertEqual(self.X.shape[0], self.y.shape[0], "Feature and label count should match")

    def test_model_training(self):
        model, X_test, y_test = train_model(self.X, self.y)
        self.assertTrue(hasattr(model, "predict"), "Model should have a predict method")
        self.assertGreater(len(X_test), 0, "Test set should not be empty")

    def test_evaluation_output(self):
        model, X_test, y_test = train_model(self.X, self.y)
        report, matrix = evaluate_model(model, X_test, y_test, self.encoder)
        self.assertIsInstance(report, str, "Report should be a string")
        # Now 'np.ndarray' will be recognized
        self.assertIsInstance(matrix, (list, pd.DataFrame, np.ndarray), "Confusion matrix should be a matrix-like object")

if __name__ == '__main__':
    unittest.main()