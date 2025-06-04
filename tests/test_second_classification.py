import unittest
import os
import pandas as pd
import shutil 
import sys

try:
   
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
  
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
 
    from second_classification import run_clustering_pipeline
except (NameError, ImportError) as e:
    
    print(f"Warning: Could not automatically set up sys.path. "
          f"Ensure 'run_clustering_pipeline' is importable. Error: {e}")

    if 'run_clustering_pipeline' not in globals():
        
        pass 


TEST_DATA_DICT = {
    "age_v": [25, 30, 22, 40, 35], # Added one more point for k=2 to be more stable
    "sex_v": ["M", "F", "F", "M", "F"],
    "greutate": [70, 60, 55, 80, 65],
    "inaltime": [175, 160, 162, 180, 168]
}
TEST_CSV_FILENAME = "test_data_unittest.csv"
PLOTS_DIR = "plots" # Default plots directory used by the clustering script

class TestClusteringPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up for all tests in this class."""
        # Create the test CSV file
        test_df = pd.DataFrame(TEST_DATA_DICT)
        test_df.to_csv(TEST_CSV_FILENAME, index=False)
        # Ensure the plots directory does not exist from a previous run (or create it if needed by setup)
        if os.path.exists(PLOTS_DIR):
            shutil.rmtree(PLOTS_DIR) # Remove if it exists to ensure a clean state
        # The script itself will create PLOTS_DIR, so we don't need os.makedirs here.


    @classmethod
    def tearDownClass(cls):
        """Tear down after all tests in this class."""
        # Remove the test CSV file
        if os.path.exists(TEST_CSV_FILENAME):
            os.remove(TEST_CSV_FILENAME)
        # Remove the plots directory and its contents
        if os.path.exists(PLOTS_DIR):
            shutil.rmtree(PLOTS_DIR)

    def test_clustering_pipeline_runs_and_generates_outputs(self):
        """
        Test that the clustering pipeline runs without errors and produces
        the expected output files and summary.
        """
        self.assertTrue('run_clustering_pipeline' in globals() or 'run_clustering_pipeline' in sys.modules.get('second_classification', {}).__dict__,
                        "run_clustering_pipeline function not found. Check import.")

        features = ["age_v", "sex_v", "greutate", "inaltime"]
        other_numeric = ["age_v", "greutate", "inaltime"]
        k_opt = 2
        k_rng = range(1, 4) # k values 1, 2, 3

        elbow_path, scatter_path, summary, silhouette, error = run_clustering_pipeline(
            filepath=TEST_CSV_FILENAME,
            features_to_cluster=features,
            other_numeric_features=other_numeric,
            k_optimal=k_opt,
            k_range=k_rng
        )

        self.assertIsNone(error, f"Clustering pipeline returned an error: {error}")

        self.assertIsNotNone(elbow_path, "Elbow plot path should not be None")
        self.assertTrue(os.path.exists(elbow_path), f"Elbow plot file not found at {elbow_path}")

        self.assertIsNotNone(scatter_path, "Scatter plot path should not be None")
        self.assertTrue(os.path.exists(scatter_path), f"Scatter plot file not found at {scatter_path}")

        self.assertIsInstance(summary, str, "Summary should be a string")
        self.assertIn("cluster_label", summary, "Summary should contain 'cluster_label'")
        self.assertIn("Mean feature values per cluster", summary, "Summary should contain mean feature values")
        self.assertIn("Cluster sizes", summary, "Summary should contain cluster sizes")

        if k_opt > 1 and len(pd.DataFrame(TEST_DATA_DICT)) >= k_opt : # Basic check
             self.assertIsNotNone(silhouette, f"Silhouette score should not be None for k_optimal={k_opt}")
             self.assertGreaterEqual(silhouette, -1, "Silhouette score should be >= -1") # Silhouette can be -1
             self.assertLessEqual(silhouette, 1, "Silhouette score should be <= 1")
        else:
            self.assertIsNone(silhouette, f"Silhouette score should be None for k_optimal={k_opt} or insufficient data")


    def test_clustering_with_file_not_found(self):
        """Test behavior when the input CSV file is not found."""
        _, _, _, _, error = run_clustering_pipeline(
            filepath="non_existent_file.csv",
            k_optimal=2
        )
        self.assertIsNotNone(error, "Error should be reported for a non-existent file")
        self.assertIn("File 'non_existent_file.csv' not found", error)

    def test_clustering_with_missing_features(self):
        """Test behavior when features_to_cluster are missing from the CSV."""
        _, _, _, _, error = run_clustering_pipeline(
            filepath=TEST_CSV_FILENAME,
            features_to_cluster=["age_v", "sex_v", "non_existent_feature"],
            k_optimal=2
        )
        self.assertIsNotNone(error, "Error should be reported for missing features")
        self.assertIn("The following features are not in the CSV: ['non_existent_feature']", error)

if __name__ == '__main__':
    if 'run_clustering_pipeline' not in globals() and 'run_clustering_pipeline' not in sys.modules.get('second_classification', {}).__dict__:
        print("ERROR: 'run_clustering_pipeline' is not defined.")
        print("Please ensure the clustering code is correctly imported or defined before running tests.")
     
        
    unittest.main()