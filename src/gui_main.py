import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd
from PIL import Image, ImageTk
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

# --- Import Backend Logic ---
from detect_anomalies import detect_anomalies_from_file, save_filtered_datasets, ANOMALY_SCORE_THRESHOLD
from classification import run_imc_classification
from second_classification import run_clustering_pipeline

# --- Global variables for application state ---
root_window = None
selected_file_path_var = None
status_label_var = None

# Data holders
original_df = None
cleaned_df = None
clustered_df = None
case_anomaly_data = {}
all_case_ids = []
classification_predictions = None
cohort_names = {}

# --- GUI Widgets ---
# Tab 1
case_listbox_widget = None
patient_profile_text_widget = None
quality_plot_label_widget = None
# Tab 2
cohort_plot_label_widget = None
cohort_summary_text_widget = None

# Image references
quality_plot_image = None
cohort_plot_image = None

# --- Helper Functions ---
def get_quality_status(score):
    if score is None: return "Unknown", "grey"
    if score == 0: return "OK", "light green"
    if 0 < score < ANOMALY_SCORE_THRESHOLD: return "Needs Review", "gold"
    if score >= ANOMALY_SCORE_THRESHOLD: return "Critical Data Errors", "red"
    return "Unknown", "white"

def display_image_in_label(label_widget, image_path, global_image_ref, max_width=500, max_height=400):
    global quality_plot_image, cohort_plot_image
    try:
        img = Image.open(image_path)
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        photo_image = ImageTk.PhotoImage(img)

        if global_image_ref == "quality_plot_image": quality_plot_image = photo_image
        elif global_image_ref == "cohort_plot_image": cohort_plot_image = photo_image
        
        label_widget.config(image=photo_image, text="")
        label_widget.image = photo_image # Keep a reference!
    except Exception as e:
        label_widget.config(image='', text=f"Plot not found or could not be displayed.")
        print(f"Error displaying image {image_path}: {e}")

# --- GUI Update and Display Functions ---
def update_patient_listbox():
    case_listbox_widget.config(state="normal")
    case_listbox_widget.delete(0, tk.END)
    if not all_case_ids:
        case_listbox_widget.config(state="disabled")
        return

    for case_id in all_case_ids:
        score = case_anomaly_data.get(str(case_id), {}).get('total_score', 0)
        status, color = get_quality_status(score)
        case_listbox_widget.insert(tk.END, f"{case_id} - Status: {status}")
    
    for i in range(case_listbox_widget.size()):
        case_id_str = case_listbox_widget.get(i).split(' - ')[0]
        score = case_anomaly_data.get(case_id_str, {}).get('total_score', 0)
        _, color = get_quality_status(score)
        fg_color = "white" if color.lower() in ["red"] else "black"
        case_listbox_widget.itemconfig(i, {'bg': color, 'fg': fg_color})
        
    if case_listbox_widget.size() > 0:
        case_listbox_widget.selection_set(0)
        on_case_select(None)

def on_case_select(event):
    if not case_listbox_widget.curselection():
        return
    
    selected_index = case_listbox_widget.curselection()[0]
    if selected_index >= len(all_case_ids):
        return
    case_id_str = all_case_ids[selected_index]
    
    update_patient_profile_text(case_id_str)

def update_patient_profile_text(case_id):
    profile_text = f"--- Clinical Profile for Patient ID: {case_id} ---\n\n"
    if original_df is None:
        return
        
    case_row = original_df[original_df['id_cases'].astype(str) == str(case_id)]
    if case_row.empty:
        patient_profile_text_widget.config(state="normal")
        patient_profile_text_widget.delete(1.0, tk.END)
        patient_profile_text_widget.insert(tk.END, "Could not retrieve patient data.")
        patient_profile_text_widget.config(state="disabled")
        return

    case_row = case_row.iloc[0]

    profile_text += "Data Quality Review:\n"
    anomaly_info = case_anomaly_data.get(case_id)
    if anomaly_info and anomaly_info['violations']:
        for viol in anomaly_info['violations']:
            profile_text += f"- {viol['clinical_description']}\n"
    else:
        profile_text += "- No data quality flags found for this patient.\n"
    profile_text += "\n"


    profile_text += "Patient Data:\n"
    profile_text += f"  - Age: {case_row.get('age_v', 'N/A')}\n"
    profile_text += f"  - Sex: {case_row.get('sex_v', 'N/A')}\n"
    profile_text += f"  - Weight: {case_row.get('greutate', 'N/A')} kg\n"
    profile_text += f"  - Height: {case_row.get('inaltime', 'N/A')} cm\n\n"

    profile_text += "BMI Profile:\n"
    recorded_bmi_category = case_row.get('imcINdex', 'N/A')
    profile_text += f"  - Recorded BMI Category/Value: {recorded_bmi_category}\n"

    if classification_predictions is not None and case_row.name in classification_predictions:
        predicted_bmi_category = classification_predictions[case_row.name]
        profile_text += f"  - Predicted BMI Category (from features): {predicted_bmi_category}\n"
        
        if recorded_bmi_category is not None and predicted_bmi_category is not None and str(recorded_bmi_category).strip().lower() != str(predicted_bmi_category).strip().lower():
            profile_text += "  - Clinical Insight: The recorded BMI value/category differs from the prediction. This may indicate a data entry error. Please verify.\n"
    profile_text += "\n"

    profile_text += "Cohort Profile:\n"
    if clustered_df is not None:
        patient_cluster_info = clustered_df[clustered_df['id_cases'].astype(str) == str(case_id)]
        if not patient_cluster_info.empty:
            cohort_name = patient_cluster_info.iloc[0]['cohort_name']
            profile_text += f"  - This patient belongs to the '{cohort_name}' cohort.\n"
        else:
            profile_text += "  - Patient not found in the cohort analysis (may have been excluded due to data quality issues).\n"
    
    patient_profile_text_widget.config(state="normal")
    patient_profile_text_widget.delete(1.0, tk.END)
    patient_profile_text_widget.insert(tk.END, profile_text)
    patient_profile_text_widget.config(state="disabled")

# --- Main Actions ---
def load_file_action():
    path = filedialog.askopenfilename(title="Select Patient Dataset (CSV)", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if not path:
        return
    
    clear_all_action()
    selected_file_path_var.set(path)
    status_label_var.set(f"Loaded: {os.path.basename(path)}. Click 'Analyze' to begin.")

def clear_all_action():
    global original_df, cleaned_df, clustered_df, case_anomaly_data, all_case_ids, classification_predictions, cohort_names
    global quality_plot_image, cohort_plot_image
    
    original_df = None
    cleaned_df = None
    clustered_df = None
    case_anomaly_data = {}
    all_case_ids = []
    classification_predictions = None
    cohort_names = {}
    quality_plot_image = None
    cohort_plot_image = None
    
    selected_file_path_var.set("No dataset loaded.")
    status_label_var.set("Ready. Please load a patient dataset.")
    
    case_listbox_widget.config(state="normal")
    case_listbox_widget.delete(0, tk.END)
    case_listbox_widget.config(state="disabled")
    
    patient_profile_text_widget.config(state="normal")
    patient_profile_text_widget.delete(1.0, tk.END)
    patient_profile_text_widget.config(state="disabled")
    
    quality_plot_label_widget.config(image='', text="Plot appears after analysis.")
    cohort_plot_label_widget.config(image='', text="Plot appears after analysis.")

    cohort_summary_text_widget.config(state="normal")
    cohort_summary_text_widget.delete(1.0, tk.END)
    cohort_summary_text_widget.config(state="disabled")

def run_full_analysis():
    filepath = selected_file_path_var.get()
    if not filepath or not os.path.exists(filepath):
        messagebox.showerror("Error", "Please load a patient dataset first.")
        return

    current_filepath = selected_file_path_var.get()

    global original_df, cleaned_df, clustered_df, case_anomaly_data, all_case_ids, classification_predictions, cohort_names

    try:
        status_label_var.set("Step 1/3: Checking data quality...")
        root_window.update_idletasks()
        
        original_df = pd.read_csv(current_filepath)
        anomaly_msgs, anomaly_scores, case_anomaly_data = detect_anomalies_from_file(current_filepath)
        if any(msg.startswith("CRITICAL ERROR:") for msg in anomaly_msgs):
            messagebox.showerror("Critical Error", "\n".join(anomaly_msgs))
            status_label_var.set("Analysis failed during data quality check.")
            return

        plt.figure(figsize=(10, 6))
        sns.histplot(anomaly_scores if anomaly_scores else [0], bins=20, kde=True)
        plt.title('Population Data Quality Overview', fontsize=16)
        plt.xlabel('Data Anomaly Score (Higher is worse)', fontsize=12)
        plt.ylabel('Number of Patient Records', fontsize=12)
        plt.tight_layout()
        plt.savefig("quality_plot.png")
        plt.close()
        display_image_in_label(quality_plot_label_widget, "quality_plot.png", "quality_plot_image")

        cleaned_file_path = save_filtered_datasets(original_df.copy(), case_anomaly_data, ANOMALY_SCORE_THRESHOLD, os.path.basename(current_filepath))

        all_case_ids = sorted(original_df['id_cases'].astype(str).unique())
        update_patient_listbox()
        
        status_label_var.set("Step 2/3: Analyzing BMI predictions...")
        root_window.update_idletasks()
        classification_predictions = None
        if cleaned_file_path and os.path.exists(cleaned_file_path):
            cleaned_df = pd.read_csv(cleaned_file_path)
            report, matrix, error_msg, model, X_test, y_test, label_encoder = run_imc_classification(cleaned_file_path)
            if not error_msg and model is not None:
                y_pred = model.predict(X_test)
                pred_series = pd.Series(y_pred, index=y_test.index)
                classification_predictions = pred_series.map(lambda x: label_encoder.inverse_transform([x])[0])
            elif error_msg:
                print(f"IMC Classification warning: {error_msg}")
        else:
            print("Warning: Cleaned file for classification not found. Skipping.")

        status_label_var.set("Step 3/3: Discovering patient cohorts...")
        root_window.update_idletasks()
        clustered_df = None
        cohort_names = {}
        if cleaned_file_path and os.path.exists(cleaned_file_path):
            scatter_path, summary_text, clustered_df_result, cohort_names_result, error_msg = run_clustering_pipeline(cleaned_file_path)
            if not error_msg:
                clustered_df = clustered_df_result
                cohort_names = cohort_names_result
                display_image_in_label(cohort_plot_label_widget, scatter_path, "cohort_plot_image")
                cohort_summary_text_widget.config(state="normal")
                cohort_summary_text_widget.delete(1.0, tk.END)
                cohort_summary_text_widget.insert(tk.END, summary_text)
                cohort_summary_text_widget.config(state="disabled")
            else:
                messagebox.showwarning("Clustering Warning", error_msg)
        else:
            print("Warning: Cleaned file for clustering not found. Skipping.")

        status_label_var.set("Analysis complete. Review results in the tabs.")
        messagebox.showinfo("Success", "Full analysis is complete. You can now explore the patient data.")
        
        if case_listbox_widget.size() > 0:
            on_case_select(None)

    except Exception as e:
        status_label_var.set("An error occurred during analysis.")
        messagebox.showerror("Analysis Error", f"An unexpected error occurred: {e}\n\n{traceback.format_exc()}")


# --- Main Application Window Setup ---
def create_main_application_window():
    global root_window, selected_file_path_var, status_label_var
    global case_listbox_widget, patient_profile_text_widget, quality_plot_label_widget
    global cohort_plot_label_widget, cohort_summary_text_widget
    
    root_window = tk.Tk()
    root_window.title("Clinical Data Analyzer")
    root_window.geometry("1200x800")
    
    top_controls_frame = ttk.Frame(root_window, padding="10")
    top_controls_frame.pack(side="top", fill="x", pady=(5,0))
    
    load_button = ttk.Button(top_controls_frame, text="Load Patient Dataset (CSV)", command=load_file_action)
    load_button.pack(side="left", padx=(0, 10))
    
    selected_file_path_var = tk.StringVar(value="No dataset loaded.")
    ttk.Label(top_controls_frame, textvariable=selected_file_path_var, relief="sunken", anchor="w").pack(side="left", fill="x", expand=True, padx=10)

    analysis_controls_frame = ttk.Frame(root_window)
    analysis_controls_frame.pack(pady=(5,10))

    analyze_button = ttk.Button(analysis_controls_frame, text="Analyze Patient Dataset", command=run_full_analysis)
    analyze_button.pack(side="left", padx=5)

    clear_button = ttk.Button(analysis_controls_frame, text="Clear All", command=clear_all_action)
    clear_button.pack(side="left", padx=5)
    
    notebook = ttk.Notebook(root_window)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    overview_tab = ttk.Frame(notebook, padding="10")
    notebook.add(overview_tab, text='1. Patient Overview')
    
    overview_paned_window = ttk.PanedWindow(overview_tab, orient=tk.HORIZONTAL)
    overview_paned_window.pack(fill=tk.BOTH, expand=True)

    left_overview_pane = ttk.PanedWindow(overview_paned_window, orient=tk.VERTICAL)
    overview_paned_window.add(left_overview_pane, weight=2)
    
    list_frame = ttk.Frame(left_overview_pane, padding=5)
    left_overview_pane.add(list_frame, weight=1)
    ttk.Label(list_frame, text="Patient Records", font=("Helvetica", 12, "bold")).pack(anchor="w")
    listbox_frame = ttk.Frame(list_frame)
    listbox_frame.pack(fill="both", expand=True, pady=5)
    case_list_y_scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical")
    case_listbox_widget = tk.Listbox(listbox_frame, yscrollcommand=case_list_y_scrollbar.set, exportselection=False, height=10)
    case_list_y_scrollbar.config(command=case_listbox_widget.yview)
    case_listbox_widget.pack(side="left", fill="both", expand=True)
    case_list_y_scrollbar.pack(side="right", fill="y")
    case_listbox_widget.bind("<<ListboxSelect>>", on_case_select)
    case_listbox_widget.config(state="disabled")

    profile_frame = ttk.Frame(left_overview_pane, padding=5)
    left_overview_pane.add(profile_frame, weight=2)
    ttk.Label(profile_frame, text="Clinical Profile", font=("Helvetica", 12, "bold")).pack(anchor="w")
    patient_profile_text_widget = tk.Text(profile_frame, wrap="word", font=("Helvetica", 11), state="disabled")
    patient_profile_text_widget.pack(fill="both", expand=True, pady=5)


    right_overview_pane = ttk.Frame(overview_paned_window, padding=5)
    overview_paned_window.add(right_overview_pane, weight=3)
    ttk.Label(right_overview_pane, text="Population Data Quality Overview", font=("Helvetica", 12, "bold")).pack()
    quality_plot_label_widget = ttk.Label(right_overview_pane, text="Plot appears after analysis.", relief="groove", anchor="center")
    quality_plot_label_widget.pack(fill="both", expand=True, pady=5)
    
    cohort_tab = ttk.Frame(notebook, padding="10")
    notebook.add(cohort_tab, text='2. Population Cohorts')
    
    cohort_paned_window = ttk.PanedWindow(cohort_tab, orient=tk.VERTICAL)
    cohort_paned_window.pack(fill=tk.BOTH, expand=True)

    top_cohort_pane = ttk.Frame(cohort_paned_window, padding=5)
    cohort_paned_window.add(top_cohort_pane, weight=3)
    ttk.Label(top_cohort_pane, text="Population Cohort Explorer", font=("Helvetica", 12, "bold")).pack()
    cohort_plot_label_widget = ttk.Label(top_cohort_pane, text="Plot appears after analysis.", relief="groove", anchor="center")
    cohort_plot_label_widget.pack(fill="both", expand=True, pady=5)

    bottom_cohort_pane = ttk.Frame(cohort_paned_window, padding=5)
    cohort_paned_window.add(bottom_cohort_pane, weight=2)
    ttk.Label(bottom_cohort_pane, text="Cohort Characteristics", font=("Helvetica", 12, "bold")).pack()
    cohort_summary_text_widget = tk.Text(bottom_cohort_pane, wrap="word", state="disabled", height=8)
    cohort_summary_text_widget.pack(fill="both", expand=True, pady=5)
    
    status_bar = ttk.Frame(root_window, relief="sunken", padding=2)
    status_bar.pack(side="bottom", fill="x")
    status_label_var = tk.StringVar(value="Ready. Please load a patient dataset.")
    ttk.Label(status_bar, textvariable=status_label_var, anchor="w").pack(fill="x")
    
    root_window.mainloop()

if __name__ == "__main__":
    create_main_application_window()