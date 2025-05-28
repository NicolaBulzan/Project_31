import unittest
import pandas as pd
import os
from detect_anomalies import detect_anomalies_from_file, save_filtered_datasets, ANOMALY_SCORE_THRESHOLD

class TestAnomalyDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a minimal dummy CSV
        cls.test_csv = "test_cases.csv"
        data = {
            'id_cases': [1, 2],
            'age_v': [30, 200],  # 200 is impossible
            'greutate': [70, -5],  # -5 is invalid
            'inaltime': [170, 160],
            'IMC': [24, 45],
            'data1': ['2023-01-01 10:00:00', '2023-01-01 10:30:00'],
            'imcINdex': ['normal', 'obese']
        }
        df = pd.DataFrame(data)
        df.to_csv(cls.test_csv, index=False)

    @classmethod
    def tearDownClass(cls):
        # Clean up test files
        if os.path.exists(cls.test_csv):
            os.remove(cls.test_csv)
        for suffix in ["_cleaned.csv", "_anomalous_rule_based.csv"]:
            f = cls.test_csv.replace(".csv", suffix)
            if os.path.exists(f):
                os.remove(f)

    def test_anomaly_detection_returns_expected_outputs(self):
        messages, scores, details = detect_anomalies_from_file(self.test_csv)
        self.assertIsInstance(messages, list)
        self.assertIsInstance(scores, list)
        self.assertIsInstance(details, dict)
        self.assertGreater(len(messages), 0)
        self.assertGreater(len(scores), 0)

    def test_save_filtered_datasets_creates_files(self):
        df = pd.read_csv(self.test_csv)
        _, _, details = detect_anomalies_from_file(self.test_csv)
        save_filtered_datasets(df, details, ANOMALY_SCORE_THRESHOLD, self.test_csv)

        self.assertTrue(os.path.exists("test_cases_cleaned.csv"))
        self.assertTrue(os.path.exists("test_cases_anomalous_rule_based.csv"))

if __name__ == "__main__":
    unittest.main()