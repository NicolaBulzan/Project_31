import unittest
import os
import pandas as pd
import shutil 
import sys

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from second_classification import run_clustering_pipeline

TEST_DATA_DICT = {
    "id_cases": [1, 2, 3, 4, 5],
    "age_v": [25, 30, 22, 40, 35],
    "sex_v": ["M", "F", "F", "M", "F"],
    "greutate": [70, 60, 55, 80, 65],
    "inaltime": [175, 160, 162, 180, 168]
}
TEST_CSV_FILENAME = "test_data_unittest.csv"
PLOTS_DIR = "plots"

class TestClusteringPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_df = pd.DataFrame(TEST_DATA_DICT)
        test_df.to_csv(TEST_CSV_FILENAME, index=False)
        if os.path.exists(PLOTS_DIR):
            shutil.rmtree(PLOTS_DIR)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_CSV_FILENAME):
            os.remove(TEST_CSV_FILENAME)
        if os.path.exists(PLOTS_DIR):
            shutil.rmtree(PLOTS_DIR)

    def test_clustering_pipeline_runs_and_generates_outputs(self):
        features = ["age_v", "sex_v", "greutate", "inaltime"]
        other_numeric = ["age_v", "greutate", "inaltime"]
        k_opt = 2

        scatter_path, summary, df_processed, cohort_names, error = run_clustering_pipeline(
            filepath=TEST_CSV_FILENAME,
            features_to_cluster=features,
            other_numeric_features=other_numeric,
            k_optimal=k_opt,
        )

        self.assertIsNone(error, f"Clustering pipeline returned an error: {error}")
        self.assertIsNotNone(scatter_path, "Scatter plot path should not be None")
        self.assertTrue(os.path.exists(scatter_path), f"Scatter plot file not found at {scatter_path}")
        self.assertIsInstance(summary, str)
        self.assertIn("Cohort Characteristics", summary)
        self.assertIsNotNone(df_processed)
        self.assertIn('cohort_name', df_processed.columns)
        self.assertGreater(len(cohort_names), 0)

    def test_clustering_with_file_not_found(self):
        _, _, _, _, error = run_clustering_pipeline(filepath="non_existent_file.csv", k_optimal=2)
        self.assertIsNotNone(error)
       
        self.assertTrue("not found" in error.lower() or "no such file or directory" in error.lower())

    def test_clustering_with_missing_features(self):
        _, _, _, _, error = run_clustering_pipeline(
            filepath=TEST_CSV_FILENAME,
            features_to_cluster=["age_v", "sex_v", "non_existent_feature"],
            k_optimal=2
        )
        self.assertIsNotNone(error)
        self.assertIn("non_existent_feature", error)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)