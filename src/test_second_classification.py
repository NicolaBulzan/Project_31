import unittest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
import numpy as np
import os
import second_classification as up # MODIFIED IMPORT

class TestUnsupervisedPipeline(unittest.TestCase): # You can rename this class if you like

    def setUp(self):
        self.test_csv_filepath = "dummy_test_data.csv"
        self.sample_data_dict = {
            'id_cases': [1, 2, 3, 4, 5, 6, 7],
            'age_v': [30, 45, 22, "30", "40", 50, 35],
            'sex_v': ['male', 'female', 'male', 'female', 'male', 'female', "1"],
            'greutate': [70, 60, 80, "65", "75", 55, 90],
            'inaltime': [175, 160, 180, "165", "170", 155, 120],
            'other_col': ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        }
        self.df_sample = pd.DataFrame(self.sample_data_dict)
        self.df_sample.to_csv(self.test_csv_filepath, index=False)
        self.features_for_clustering = ["age_v", "sex_v", "greutate", "inaltime"]
        self.other_numeric_features = ["age_v", "greutate", "inaltime"]

    def tearDown(self):
        if os.path.exists(self.test_csv_filepath):
            os.remove(self.test_csv_filepath)
        plot_files_to_remove = ["elbow_plot_unsupervised.png",
                                f"clusters_visualization_k{up.K_OPTIMAL_CONFIG}.png",
                                "test_elbow.png", "test_cluster_viz_k3.png"]
        for f in plot_files_to_remove:
            if os.path.exists(f):
                os.remove(f)

    def test_load_data_success(self):
        df = up.load_data(self.test_csv_filepath)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.sample_data_dict['id_cases']))

    def test_load_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            up.load_data("non_existent_file.csv")

    def test_check_features_exist_success(self):
        df = pd.DataFrame(columns=self.features_for_clustering)
        self.assertTrue(up.check_features_exist(df, self.features_for_clustering))

    def test_check_features_exist_missing(self):
        df = pd.DataFrame(columns=['age_v', 'sex_v'])
        with self.assertRaises(ValueError):
            up.check_features_exist(df, self.features_for_clustering)

    def test_preprocess_data_encoding_and_conversion(self):
        df_loaded = pd.read_csv(self.test_csv_filepath)
        df_processed, X_cluster_processed = up.preprocess_data(
            df_loaded, self.features_for_clustering, self.other_numeric_features
        )
        
        # These assertions check if the columns are numeric, which should now be passing
        self.assertTrue(pd.api.types.is_numeric_dtype(df_processed['sex_v']))
        self.assertTrue(pd.api.types.is_numeric_dtype(X_cluster_processed['sex_v']))
        
        # --- Corrected Assertions for Label Encoded Values ---
        # Original 'male' at loc[0] is encoded to 2
        self.assertEqual(df_processed.loc[0, 'sex_v'], 2) 
        # Original 'female' at loc[1] is encoded to 1
        self.assertEqual(df_processed.loc[1, 'sex_v'], 1) 
        # Original string "1" at loc[6] is encoded to 0
        self.assertEqual(df_processed.loc[6, 'sex_v'], 0)
        # --- End Corrected Assertions ---
        
        for col in self.other_numeric_features:
            self.assertTrue(pd.api.types.is_numeric_dtype(df_processed[col]))
            self.assertTrue(pd.api.types.is_numeric_dtype(X_cluster_processed[col]))
        self.assertEqual(df_processed.loc[3, 'age_v'], 30) # Checks numeric conversion of age "30"

    def test_preprocess_data_unconvertible_value_error(self):
        bad_data_df = self.df_sample.copy()
        bad_data_df.loc[0, 'age_v'] = 'thirty_text'
        with self.assertRaises(ValueError):
            up.preprocess_data(bad_data_df, self.features_for_clustering, self.other_numeric_features)

    def test_scale_data(self):
        data = {'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]}
        X_cluster_processed = pd.DataFrame(data)
        X_scaled_df, scaler = up.scale_data(X_cluster_processed)
        self.assertIsInstance(X_scaled_df, pd.DataFrame)
        self.assertEqual(X_scaled_df.shape, X_cluster_processed.shape)
        for col in X_scaled_df.columns:
            self.assertAlmostEqual(X_scaled_df[col].mean(), 0, places=5)
        self.assertTrue(hasattr(scaler, 'mean_'))

    def test_get_elbow_inertia(self):
        X_scaled_sample = pd.DataFrame(np.random.rand(50, 4), columns=self.features_for_clustering)
        k_range_config = range(1, 4)
        inertia = up.get_elbow_inertia(X_scaled_sample, k_range_config)
        self.assertIsInstance(inertia, list)
        self.assertEqual(len(inertia), len(k_range_config))
        self.assertTrue(all(isinstance(i, float) for i in inertia))

    @patch('second_classification.plt.savefig') # MODIFIED PATCH TARGET
    @patch('second_classification.plt.close') # MODIFIED PATCH TARGET
    def test_plot_elbow_curve(self, mock_close, mock_savefig):
        up.plot_elbow_curve(range(1, 4), [10, 5, 3], filename="test_elbow.png")
        mock_savefig.assert_called_once_with("test_elbow.png")

    def test_apply_kmeans(self):
        X_scaled_sample = pd.DataFrame(np.random.rand(50, 4), columns=self.features_for_clustering)
        n_clusters = 2
        model, labels = up.apply_kmeans(X_scaled_sample, n_clusters)
        self.assertIsInstance(model, up.KMeans) # This refers to sklearn.cluster.KMeans via the import in second_classification
        self.assertEqual(len(labels), 50)
        self.assertTrue(all(0 <= label < n_clusters for label in labels))

    @patch('builtins.print')
    def test_analyze_clusters_summary(self, mock_print):
        df_processed = self.df_sample.copy()
        df_processed['age_v'] = pd.to_numeric(df_processed['age_v'], errors='coerce').fillna(0)
        df_processed['greutate'] = pd.to_numeric(df_processed['greutate'], errors='coerce').fillna(0)
        df_processed['inaltime'] = pd.to_numeric(df_processed['inaltime'], errors='coerce').fillna(0)
        # Use LabelEncoder from the module being tested or directly if it's just for test data setup
        le = up.LabelEncoder() # Assuming LabelEncoder is available as up.LabelEncoder or from sklearn.preprocessing
        df_processed['sex_v'] = le.fit_transform(df_processed['sex_v'].astype(str))
        df_processed['cluster_label'] = np.random.randint(0, 2, size=len(df_processed))
        cluster_summary = up.analyze_clusters_summary(df_processed, self.features_for_clustering)
        self.assertIsInstance(cluster_summary, pd.DataFrame)
        self.assertEqual(len(cluster_summary), df_processed['cluster_label'].nunique())

    @patch('second_classification.plt.savefig') # MODIFIED PATCH TARGET
    @patch('second_classification.plt.close') # MODIFIED PATCH TARGET
    def test_plot_cluster_visualization(self, mock_close, mock_savefig):
        df_processed = self.df_sample.copy()
        df_processed['cluster_label'] = 0
        up.plot_cluster_visualization(df_processed, 3, filename_template="test_cluster_viz_k{}.png")
        mock_savefig.assert_called_once_with("test_cluster_viz_k3.png")

    def test_calculate_silhouette_score(self):
        X_scaled_sample = pd.DataFrame(np.random.rand(50, 4))
        labels = np.random.randint(0, 2, size=50)
        score = up.calculate_silhouette_score(X_scaled_sample, labels, n_clusters=2)
        self.assertIsInstance(score, float)
        self.assertTrue(-1 <= score <= 1)

    def test_calculate_silhouette_score_k1(self):
        score = up.calculate_silhouette_score(pd.DataFrame(), [], n_clusters=1)
        self.assertIsNone(score)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)