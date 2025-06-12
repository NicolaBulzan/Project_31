import unittest
import pandas as pd
import os
import sys

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from classification import run_imc_classification

class TestIMCClassificationPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_csv_filename = "classification_data_test.csv"
        cls.output_plot_dir = "plots"
        cls.cm_plot_path = os.path.join(cls.output_plot_dir, "imc_confusion_matrix.png")

        data = {
            'id_cases': list(range(1, 21)),
            'age_v': [25, 30, 35, 40, 45, 28, 32, 38, 42, 48, 22, 50, 60, 33, 29, 31, 36, 43, 55, 27],
            'sex_v': ['male', 'female'] * 10,
            'greutate': [70, 60, 80, 55, 90, 65, 75, 58, 85, 62, 45, 100, 110, 72, 68, 73, 77, 83, 66, 69],
            'inaltime': [175, 165, 180, 160, 185, 170, 178, 162, 182, 168, 160, 170, 175, 170, 165, 172, 174, 169, 167, 171],
            'imcINdex': ['Normal', 'Normal', 'Overweight', 'Normal', 'Obese',
                         'Normal', 'Overweight', 'Normal', 'Overweight', 'Normal',
                         'Underweight', 'Obese', 'Obese', 'Normal', 'Normal',
                         'Underweight', 'Normal', 'Overweight', 'Obese', 'Normal'],
            'IMC': ['Normal', 'Normal', 'Overweight', 'Normal', 'Obese',
                    'Normal', 'Overweight', 'Normal', 'Overweight', 'Normal',
                    'Underweight', 'Obese', 'Obese', 'Normal', 'Normal',
                    'Underweight', 'Normal', 'Overweight', 'Obese', 'Normal']
        }
        pd.DataFrame(data).to_csv(cls.test_csv_filename, index=False)
        os.makedirs(cls.output_plot_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_csv_filename):
            os.remove(cls.test_csv_filename)
        if os.path.exists(cls.cm_plot_path):
             os.remove(cls.cm_plot_path)
        if os.path.exists(cls.output_plot_dir) and not os.listdir(cls.output_plot_dir):
            try:
                os.rmdir(cls.output_plot_dir)
            except OSError as e:
                print(f"Warning: Could not remove plots directory '{cls.output_plot_dir}': {e}")

    def test_run_imc_classification_success(self):
        report, matrix, error, model, X_test, y_test, label_encoder = run_imc_classification(self.test_csv_filename)
        self.assertIsNone(error, f"Expected no error, but got: {error}")
        self.assertIsInstance(report, str)
        self.assertTrue(len(report) > 0)
        self.assertIsNotNone(model)

    def test_run_imc_classification_file_not_found(self):
        report, matrix, error, model, X_test, y_test, label_encoder = run_imc_classification("non_existent_file.csv")
        self.assertIsNotNone(error)
        self.assertIn("no such file or directory", error.lower())
        self.assertIsNone(report)
        self.assertIsNone(model)

    def test_run_imc_classification_missing_columns(self):
        bad_data_filename = "classification_bad_data_test.csv"
        bad_data = {'age_v': [25, 30], 'IMC': ['Normal', 'Normal']}
        pd.DataFrame(bad_data).to_csv(bad_data_filename, index=False)

        report, matrix, error, model, X_test, y_test, label_encoder = run_imc_classification(bad_data_filename)
        self.assertIsNotNone(error)
        self.assertIn("not found in the csv", error.lower())

        if os.path.exists(bad_data_filename):
            os.remove(bad_data_filename)

    def test_run_imc_classification_empty_data(self):
        """Test the case where the input CSV is empty or has only headers."""
        empty_filename = "empty_data.csv"
        with open(empty_filename, 'w') as f:
            f.write("id_cases,age_v,sex_v,greutate,inaltime,imcINdex,IMC\n")

        _, _, error, _, _, _, _ = run_imc_classification(empty_filename)
        self.assertIsNotNone(error)
        self.assertIn("no data available", error.lower())

        os.remove(empty_filename)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)