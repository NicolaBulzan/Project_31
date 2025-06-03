# Fix for test_second_classification.py
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for tests

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.second_classification import run_clustering_pipeline

class TestClusteringPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_csv_filename = "test_clustering_dummy_data.csv"
        cls.output_plot_dir = os.path.join(project_root, "plots") # Plots saved relative to project root
        cls.elbow_plot_filename = "clustering_elbow_plot.png"
        cls.scatter_plot_filename_template = "clustering_scatter_plot_k{}.png" # k will be filled

        sample_data_dict = {
            'id_cases': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'age_v': [30, 45, 22, 30, 40, 50, 35, 25, 55, 60],
            'sex_v': ['male', 'female', 'male', 'female', 'male', 'female', "1", 'male', '0', 'female'], # Mixed types
            'greutate': [70, 60, 80, 65, 75, 55, 90, 68, 72, 59],
            'inaltime': [175, 160, 180, 165, 170, 155, 120, 170, 165, 160],
            'other_col': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        }
        df_sample = pd.DataFrame(sample_data_dict)
        df_sample.to_csv(cls.test_csv_filename, index=False)
        
        # Ensure plots directory exists
        os.makedirs(cls.output_plot_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_csv_filename):
            os.remove(cls.test_csv_filename)
        
        # Clean up generated plots
        elbow_plot_path = os.path.join(cls.output_plot_dir, cls.elbow_plot_filename)
        if os.path.exists(elbow_plot_path):
            os.remove(elbow_plot_path)
        
        # Need to know k_optimal used in tests to clean up scatter plot
        # Assuming a default k_optimal for testing, e.g., 3
        k_test = 3 
        scatter_plot_path = os.path.join(cls.output_plot_dir, cls.scatter_plot_filename_template.format(k_test))
        if os.path.exists(scatter_plot_path):
            os.remove(scatter_plot_path)
            
        # Clean up plots directory if empty
        try:
            if os.path.exists(cls.output_plot_dir) and not os.listdir(cls.output_plot_dir):
                os.rmdir(cls.output_plot_dir)
        except OSError as e:
            print(f"Warning: Could not remove plots directory '{cls.output_plot_dir}': {e}")


    def test_run_clustering_pipeline_success(self):
        k_optimal_test = 3
        features = ["age_v", "sex_v", "greutate", "inaltime"]
        other_numeric = ["age_v", "greutate", "inaltime"]

        elbow_p, scatter_p, summary_s, sil_score, error = run_clustering_pipeline(
            filepath=self.test_csv_filename,
            features_to_cluster=features,
            other_numeric_features=other_numeric,
            k_optimal=k_optimal_test
        )

        self.assertIsNone(error, f"Expected no error, but got: {error}")
        
        self.assertIsInstance(elbow_p, str, "Elbow plot path should be a string.")
        self.assertTrue(os.path.exists(elbow_p), f"Elbow plot not found: {elbow_p}")
        self.assertEqual(os.path.basename(elbow_p), self.elbow_plot_filename)

        self.assertIsInstance(scatter_p, str, "Scatter plot path should be a string.")
        self.assertTrue(os.path.exists(scatter_p), f"Scatter plot not found: {scatter_p}")
        self.assertEqual(os.path.basename(scatter_p), self.scatter_plot_filename_template.format(k_optimal_test))

        self.assertIsInstance(summary_s, str, "Summary should be a string.")
        self.assertTrue(len(summary_s) > 0, "Summary string should not be empty.")
        
        if sil_score is not None: # Silhouette score can be None if k_optimal is 1 or not enough distinct labels
            self.assertIsInstance(sil_score, float, "Silhouette score should be a float or None.")
            self.assertTrue(-1 <= sil_score <= 1, "Silhouette score out of range.")


    def test_run_clustering_pipeline_file_not_found(self):
        _, _, _, _, error = run_clustering_pipeline("non_existent_clustering_file.csv")
        self.assertIsNotNone(error, "Error should be reported for non-existent file.")
        self.assertIn("not found", error.lower())

    def test_run_clustering_pipeline_missing_features(self):
        bad_data_filename = "test_clustering_missing_features.csv"
        bad_data = {
            'age_v': [25, 30],
            'sex_v': ['male', 'female']
            # 'greutate', 'inaltime' are missing from default features
        }
        pd.DataFrame(bad_data).to_csv(bad_data_filename, index=False)
        
        _, _, _, _, error = run_clustering_pipeline(
            filepath=bad_data_filename,
            features_to_cluster=["age_v", "sex_v", "greutate", "inaltime"] # Explicitly ask for missing ones
        )
        self.assertIsNotNone(error, "Error should be reported for missing features.")
        # Fix: The error message changed slightly. Check for "missing"
        self.assertIn("missing", error.lower())
        
        if os.path.exists(bad_data_filename):
            os.remove(bad_data_filename)
            
    def test_run_clustering_pipeline_non_numeric_conversion_issue(self):
        tricky_data_filename = "test_clustering_tricky_data.csv"
        tricky_data = {
            'age_v': [25, 30, 'text_age', 40], # 'text_age' will become NaN, then imputed
            'sex_v': ['male', 'female', 'male', 'female'],
            'greutate': [70, 'sixty', 80, 65], # 'sixty' will become NaN, then imputed
            'inaltime': [175, 160, 180, 165]
        }
        pd.DataFrame(tricky_data).to_csv(tricky_data_filename, index=False)

        elbow_p, scatter_p, summary_s, sil_score, error = run_clustering_pipeline(
            filepath=tricky_data_filename,
            features_to_cluster=["age_v", "sex_v", "greutate", "inaltime"],
            other_numeric_features=["age_v", "greutate", "inaltime"],
            k_optimal=2 # Changed k_optimal for this test as there are only 4 samples
        )
        
        # Expecting it to run without error due to coercion and imputation, and valid k_optimal
        self.assertIsNone(error, f"Pipeline should handle coercible non-numeric with imputation and valid k_optimal, but got: {error}")
        self.assertIsNotNone(elbow_p) # Check if plots were generated

        if os.path.exists(tricky_data_filename):
            os.remove(tricky_data_filename)


if __name__ == '__main__':
    unittest.main()