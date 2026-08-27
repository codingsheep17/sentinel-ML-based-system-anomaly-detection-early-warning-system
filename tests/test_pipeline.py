import unittest
import pandas as pd
import os

class TestMLPipeline(unittest.TestCase):
    def test_1_cleaned_data(self):
        file_path = "data/processed/processed.csv"
        # Assertions check if a condition is True. If False, the test fails!
        self.assertTrue(os.path.exists(file_path), "Cleaned data file is missing!")
        
        df = pd.read_csv(file_path)
        self.assertIn("cpu_utilization", df.columns, "Missing cpu_utilization column!")
        self.assertEqual(df.isnull().sum().sum(), 0, "Missing values (NaNs) found in data!")

    def test_2_features_created(self):
        file_path = "data/processed/processed_features.csv"
        self.assertTrue(os.path.exists(file_path), "Features data file is missing!")
        
        df = pd.read_csv(file_path)
        self.assertIn("cpu_rolling_mean_5", df.columns)
        self.assertEqual(df["cpu_rolling_mean_5"].isnull().sum().sum(), 0, "bfill() failed! NaNs in rolling mean.")

    def test_3_model_predictions(self):
        file_path = "data/processed/severity_data.csv"
        self.assertTrue(os.path.exists(file_path), "Severity data file is missing!")
        
        df = pd.read_csv(file_path)
        self.assertIn("anomaly_label", df.columns)
        self.assertIn("severity", df.columns)

    def test_4_explanations(self):
        file_path = "data/processed/explained_data.csv"
        self.assertTrue(os.path.exists(file_path), "Explained data file is missing!")
        
        df = pd.read_csv(file_path)
        self.assertIn("explanation", df.columns)
        
        # Ensure our normal rows got the correct default explanation
        normal_rows = df[df["anomaly_label"] == 1]
        if len(normal_rows) > 0:
            self.assertEqual(normal_rows.iloc[0]["explanation"], "System operating normally.")

if __name__ == "__main__":
    unittest.main()