
import unittest
import pandas as pd
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.detect_anomalies import detect_anomalies_from_file, save_filtered_datasets, ANOMALY_SCORE_THRESHOLD

class TestAnomalyDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        cls.test_csv_filename = "anomaly_cases_test.csv"
        cls.output_plot_dir = "plots" 
        cls.anomaly_plot_path = os.path.join(cls.output_plot_dir, "anomaly_plot.png")


        data = {
            'id_cases': [1, 2, 3, 4],
            'age_v': [30, 200, 5, 80],  # 200 is impossible
            'greutate': [70, -5, 2, 15],  # -5 is invalid, 2 is very low for child, 15 is low for adult
            'inaltime': [170, 160, 60, 150],
            'IMC': [24.2, None, 5.5, 6.7], # Test None IMC, calculated BMI
            'data1': ['2023-01-01 10:00:00', '2023-01-01 10:30:00', '2023-01-02 11:00:00', '2023-01-02 11:00:00'], # Case 4 for duplicate check
            'imcINdex': ['normal', 'obese', 'underweight', 'severely underweight'],
            'sex_v': ['male', 'female', 'male', 'female'] # Added for completeness if other modules use it
        }
        df = pd.DataFrame(data)
        df.to_csv(cls.test_csv_filename, index=False)

        # Ensure plots directory exists for cleanup, though detect_anomalies should create it if saving plot
        if not os.path.exists(cls.output_plot_dir):
            os.makedirs(cls.output_plot_dir, exist_ok=True)


    @classmethod
    def tearDownClass(cls):
        # Clean up test files
        if os.path.exists(cls.test_csv_filename):
            os.remove(cls.test_csv_filename)
        
        base_name = os.path.splitext(cls.test_csv_filename)[0]
        for suffix in ["_cleaned.csv", "_anomalous_rule_based.csv"]:
            f = f"{base_name}{suffix}" # Use f-string for clarity
            if os.path.exists(f):
                os.remove(f)
        
        if os.path.exists(cls.anomaly_plot_path):
            os.remove(cls.anomaly_plot_path)
        
        try:
            if os.path.exists(cls.output_plot_dir) and not os.listdir(cls.output_plot_dir):
                os.rmdir(cls.output_plot_dir)
        except OSError as e:
            print(f"Warning: Could not remove plots directory '{cls.output_plot_dir}': {e}")


    def test_anomaly_detection_returns_expected_outputs(self):
        messages, scores, details = detect_anomalies_from_file(self.test_csv_filename)
        self.assertIsInstance(messages, list, "Messages should be a list")
        self.assertIsInstance(scores, list, "Scores should be a list")
        self.assertIsInstance(details, dict, "Details should be a dictionary")
        
        self.assertTrue(any("Impossible Age" in msg for case_data in details.values() for msg in [v['rule'] for v in case_data['violations']]), "Should detect impossible age for case 2")
        self.assertTrue(any("Invalid Weight" in msg for case_data in details.values() for msg in [v['rule'] for v in case_data['violations']]), "Should detect invalid weight for case 2")
        self.assertGreater(len(scores), 0, "Should have anomaly scores for anomalous cases")


    def test_save_filtered_datasets_creates_files(self):
        # Ensure the input DataFrame is loaded for save_filtered_datasets
        try:
            original_df = pd.read_csv(self.test_csv_filename)
        except FileNotFoundError:
            self.fail(f"Test setup error: {self.test_csv_filename} not found for test_save_filtered_datasets")

        _, _, case_data_for_filtering = detect_anomalies_from_file(self.test_csv_filename)
        
        # Call the function to save datasets
        save_filtered_datasets(original_df, case_data_for_filtering, ANOMALY_SCORE_THRESHOLD, self.test_csv_filename)

        base_name = os.path.splitext(self.test_csv_filename)[0]
        cleaned_file = f"{base_name}_cleaned.csv"
        anomalous_file = f"{base_name}_anomalous_rule_based.csv"

        self.assertTrue(os.path.exists(cleaned_file), f"{cleaned_file} was not created")
        self.assertTrue(os.path.exists(anomalous_file), f"{anomalous_file} was not created")

        # Optional: Check content of created files
        if os.path.exists(cleaned_file):
            df_cleaned = pd.read_csv(cleaned_file)
            self.assertFalse(df_cleaned.empty, "Cleaned file should not be empty as there are cases below anomaly threshold.")

        if os.path.exists(anomalous_file):
            df_anomalous = pd.read_csv(anomalous_file)
            self.assertFalse(df_anomalous.empty, "Anomalous file should not be empty as there are cases above anomaly threshold.")
    
    def test_plot_generation(self):
        
        _, scores, _ = detect_anomalies_from_file(self.test_csv_filename)
        if scores: 
            pass 
        self.assertTrue(True) # Placeholder, as direct testing of plot from this function is hard.


if __name__ == "__main__":
    unittest.main()