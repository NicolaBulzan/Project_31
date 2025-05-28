import unittest
import tkinter as tk
from gui_main import mock_detect_anomalies, update_gui_for_file_selection

class TestGuiMain(unittest.TestCase):

    def setUp(self):
        # Simulate minimal tkinter environment
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window during tests
        import gui_main
        gui_main.selected_file_path_var = tk.StringVar()
        gui_main.status_label_var = tk.StringVar()
        gui_main.run_button_widget = tk.Button(self.root, state="disabled")
        gui_main.results_text_widget = tk.Text(self.root)
        gui_main.results_text_widget.pack()
        gui_main.results_text_widget.config(state="normal")
        gui_main.root_window = self.root

    def tearDown(self):
        self.root.destroy()

    def test_mock_detect_anomalies(self):
        result = mock_detect_anomalies("test.csv")
        self.assertIsInstance(result, list)
        self.assertIn("Mock Anomalies", result[0])

    def test_update_gui_for_file_selection(self):
        from gui_main import selected_file_path_var, status_label_var, run_button_widget
        update_gui_for_file_selection("example.csv")
        self.assertIn("example.csv", selected_file_path_var.get())
        self.assertEqual(status_label_var.get(), "File loaded: example.csv")
        self.assertEqual(run_button_widget["state"], "normal")

    def test_update_gui_for_file_selection_with_none(self):
        from gui_main import selected_file_path_var, status_label_var, run_button_widget
        update_gui_for_file_selection(None)
        self.assertEqual(selected_file_path_var.get(), "No file selected.")
        self.assertIn("cancelled", status_label_var.get().lower())
        self.assertEqual(run_button_widget["state"], "disabled")

if __name__ == "__main__":
    unittest.main()