# Fix for test_classification.py
import unittest
import pandas as pd
import numpy as np
import os
import sys
import matplotlib # Important for backend selection if plt is used implicitly
matplotlib.use('Agg') # Use a non-interactive backend for tests

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.classification import run_imc_classification # Updated import

class TestIMCClassificationPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_csv_filename = "test_classification_data.csv"
        cls.output_plot_dir = os.path.join(project_root, "plots") # Plots saved relative to project root by the module
        cls.cm_plot_filename = "imc_confusion_matrix.png"
        cls.cm_plot_path = os.path.join(cls.output_plot_dir, cls.cm_plot_filename)

        data = {
            'id_cases': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'age_v': [25, 30, 35, 40, 45, 28, 32, 38, 42, 48, 22, 50, 60, 33, 29],
            'sex_v': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male'],
            'greutate': [70, 60, 80, 55, 90, 65, 75, 58, 85, 62, 45, 100, 110, 72, 68],
            'inaltime': [175, 165, 180, 160, 185, 170, 178, 162, 182, 168, 160, 170, 175, 170, 165],
            'IMC': ['Normal', 'Normal', 'Overweight', 'Normal', 'Obese', 'Normal', 'Overweight', 'Normal', 'Overweight', 'Normal', 'Underweight', 'Obese', 'Obese', 'Normal', 'Normal']
            # Added more data and 'Underweight' category to ensure more class diversity in splits
        }
        df = pd.DataFrame(data)
        df.to_csv(cls.test_csv_filename, index=False)

        # Ensure plots directory exists for the module to save into
        os.makedirs(cls.output_plot_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_csv_filename):
            os.remove(cls.test_csv_filename)
        if os.path.exists(cls.cm_plot_path):
            os.remove(cls.cm_plot_path)
        # Clean up plots directory if empty
        try:
            if os.path.exists(cls.output_plot_dir) and not os.listdir(cls.output_plot_dir):
                os.rmdir(cls.output_plot_dir)
        except OSError as e:
            print(f"Warning: Could not remove plots directory '{cls.output_plot_dir}': {e}")


    def test_run_imc_classification_success(self):
        report, matrix_image_path, error = run_imc_classification(self.test_csv_filename)
        
        self.assertIsNone(error, f"Expected no error, but got: {error}")
        self.assertIsInstance(report, str, "Classification report should be a string.")
        self.assertIsNotNone(report, "Classification report should not be None.")
        self.assertTrue(len(report) > 0, "Classification report should not be empty.")

        self.assertIsInstance(matrix_image_path, str, "Path to confusion matrix image should be a string.")
        self.assertIsNotNone(matrix_image_path, "Path to confusion matrix image should not be None.")
        self.assertEqual(os.path.basename(matrix_image_path), self.cm_plot_filename) # Check if filename matches
        self.assertTrue(os.path.exists(matrix_image_path), f"Confusion matrix image not found at: {matrix_image_path}")
        

    def test_run_imc_classification_file_not_found(self):
        report, matrix_image_path, error = run_imc_classification("non_existent_file.csv")
        self.assertIsNotNone(error, "Error should be reported for non-existent file.")
        self.assertIn("not found", error.lower(), "Error message should indicate file not found.")
        self.assertIsNone(report, "Report should be None on error.")
        self.assertIsNone(matrix_image_path, "Matrix image path should be None on error.")

    def test_run_imc_classification_missing_columns(self):
        # Create a CSV with missing required columns
        bad_data_filename = "test_classification_bad_data.csv"
        bad_data = {
            'age_v': [25, 30],
            # 'sex_v' is missing
            'greutate': [70, 60],
            'inaltime': [175, 165],
            'IMC': ['Normal', 'Normal']
        }
        pd.DataFrame(bad_data).to_csv(bad_data_filename, index=False)
        
        report, matrix_image_path, error = run_imc_classification(bad_data_filename)
        self.assertIsNotNone(error, "Error should be reported for missing columns.")
        self.assertTrue("missing" in error.lower() or "not found" in error.lower(), "Error message should indicate missing columns.")
        self.assertIsNone(report)
        self.assertIsNone(matrix_image_path)
        
        if os.path.exists(bad_data_filename):
            os.remove(bad_data_filename)

if __name__ == '__main__':
    unittest.main()