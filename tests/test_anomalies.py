import unittest
import pandas as pd
import os
import sys

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from detect_anomalies import detect_anomalies_from_file, save_filtered_datasets, ANOMALY_SCORE_THRESHOLD, get_clinical_description

class TestAnomalyDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_csv_filename = "anomaly_cases_test.csv"
        cls.output_plot_dir = "plots"
        cls.anomaly_plot_path = "quality_plot.png"

        data = {
            'id_cases': [1, 2, 3, 4, 5, 6, 7, 8, 9, 9], # Case 9 is a duplicate
            'age_v': [30, 200, 5, 25, 90, 40, -5, 60, 35, 35], # 200 and -5 are impossible
            'greutate': [70, -5, 2, 18, 120, 80, 70, 0, 75, 75], # -5, 0 invalid; 2 low child; 18 low adult
            'inaltime': [170, 160, 60, 150, 175, 300, 170, 170, 177, 177], # 300 unlikely
            'IMC': [24.2, 30.0, 5.5, 8.0, 39.2, 99.0, 24.2, 0.0, 23.8, 23.8], # 99.0 and 0.0 are incompatible
            'data1': ['2023-01-01 10:00:00', '2023-01-01 10:30:00', '2023-01-02 11:00:00', '2023-01-02 11:05:00',
                      '2023-01-03 12:00:00', '2023-01-03 12:05:00', '2023-01-04 13:00:00', '2023-01-04 13:05:00',
                      '2023-01-05 14:00:00', '2023-01-05 14:00:01'], # Case 9 timestamp is close
            'imcINdex': ['normal', 'obese', 'underweight', 'severely underweight', 'obese', 
                         'obese', 'normal', 'underweight', 'normal', 'normal'], # Case 5 is elderly & obese
            'sex_v': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'male']
        }
        pd.DataFrame(data).to_csv(cls.test_csv_filename, index=False)
        os.makedirs(cls.output_plot_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_csv_filename):
            os.remove(cls.test_csv_filename)
        
        base_name = os.path.splitext(cls.test_csv_filename)[0]
        for suffix in ["_cleaned.csv", "_anomalous_rule_based.csv"]:
            f = f"{base_name}{suffix}"
            if os.path.exists(f):
                os.remove(f)
        
        if os.path.exists(cls.anomaly_plot_path):
            os.remove(cls.anomaly_plot_path)
        
        if os.path.exists(cls.output_plot_dir) and not os.listdir(cls.output_plot_dir):
            try:
                os.rmdir(cls.output_plot_dir)
            except OSError as e:
                print(f"Warning: Could not remove plots directory '{cls.output_plot_dir}': {e}")

    def test_specific_anomaly_detection(self):
        """Test that specific, individual anomaly rules are triggered correctly."""
        _, _, details = detect_anomalies_from_file(self.test_csv_filename)
        
        # Case 1: Normal case
        self.assertNotIn('1', details, "Normal case should not have anomalies")
        
        # Case 2: Impossible Age and Invalid Weight
        self.assertTrue(any("Impossible Age" in v['rule'] for v in details['2']['violations']))
        self.assertTrue(any("Invalid Weight" in v['rule'] for v in details['2']['violations']))
        
        # Case 3: Very Low Child Weight
        self.assertTrue(any("Very Low Child Weight" in v['rule'] for v in details['3']['violations']))

        # Case 4: Implausible Adult Weight
        self.assertTrue(any("Implausible Adult Weight" in v['rule'] for v in details['4']['violations']))

        # Case 5: Suspicious Elderly Obesity
        self.assertTrue(any("Suspicious Elderly Obesity" in v['rule'] for v in details['5']['violations']))

        # Case 6: Unlikely Height and BMI Incompatible with Life
        self.assertTrue(any("Unlikely Height" in v['rule'] for v in details['6']['violations']))
        self.assertTrue(any("BMI Incompatible with Life" in v['rule'] for v in details['6']['violations']))

        # Case 9: Potential Duplicate Cluster
        self.assertTrue(any("Potential Duplicate Cluster" in v['rule'] for v in details['9']['violations']))

    def test_save_filtered_datasets_creates_files(self):
        original_df = pd.read_csv(self.test_csv_filename)
        _, _, case_data_for_filtering = detect_anomalies_from_file(self.test_csv_filename)
        
        save_filtered_datasets(original_df, case_data_for_filtering, ANOMALY_SCORE_THRESHOLD, self.test_csv_filename)

        base_name = os.path.splitext(self.test_csv_filename)[0]
        cleaned_file = f"{base_name}_cleaned.csv"
        anomalous_file = f"{base_name}_anomalous_rule_based.csv"

        self.assertTrue(os.path.exists(cleaned_file))
        self.assertTrue(os.path.exists(anomalous_file))

        df_cleaned = pd.read_csv(cleaned_file)
        df_anomalous = pd.read_csv(anomalous_file)
        self.assertIn(1, df_cleaned['id_cases'].values)
        self.assertIn(2, df_anomalous['id_cases'].values)

    def test_clinical_description_helper(self):
        """Test the get_clinical_description helper function directly."""
        self.assertIn("Invalid Age Entry", get_clinical_description("Impossible Age", 150, ""))
        self.assertIn("Missing Data", get_clinical_description("Missing Weight", "NaN", ""))
        self.assertIn("Possible Duplicate Record", get_clinical_description("Potential Duplicate Cluster", "", ""))
        self.assertIn("Unlikely Height", get_clinical_description("Unlikely Height", 300, ""))


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)