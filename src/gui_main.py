import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time
import os
import pandas as pd
from PIL import Image, ImageTk
import traceback # For detailed error logging

# --- Import Anomaly Detection ---
try:
    from detect_anomalies import detect_anomalies_from_file, save_filtered_datasets, ANOMALY_SCORE_THRESHOLD
except ImportError:
    messagebox.showerror("Import Error", "Could not import 'detect_anomalies.py'. Ensure it's in the same directory or Python path.")
    def detect_anomalies_from_file(file_path): return ["Mock critical error: 'detect_anomalies.py' not found"], [], {}
    def save_filtered_datasets(df, data, threshold, name): pass
    ANOMALY_SCORE_THRESHOLD = 3.0

# --- Import IMC Classification ---
try:
    from classification import run_imc_classification # Use the refactored script name
except ImportError:
    messagebox.showerror("Import Error", "Could not import 'classification.py'. Ensure it's in the same directory or Python path.")
    def run_imc_classification(csv_path): return "Mock IMC Classification: 'classification.py' not found.", None, "Import Error"

# --- Import Clustering ---
try:
    from second_classification import run_clustering_pipeline # Use the refactored script name
except ImportError:
    messagebox.showerror("Import Error", "Could not import 'second_classification.py'. Ensure it's in the same directory or Python path.")
    def run_clustering_pipeline(filepath, **kwargs): return None, None, "Mock Clustering: 'second_classification.py' not found.", None, "Import Error"


# --- Global variables for GUI state ---
selected_file_path_var = None
status_label_var = None
run_button_widget = None # For anomaly detection
load_button_widget = None
results_text_widget = None # For anomaly detection summary
root_window = None
loaded_df = None
case_anomaly_data = {} 
all_case_ids = [] 
case_listbox_widget = None 
case_detail_text_widget = None
plot_label_widget = None # For anomaly plot
anomaly_plot_image = None

# Globals for IMC Classification Tab
imc_report_text_widget = None
imc_cm_plot_label_widget = None
imc_cm_plot_image = None

# Globals for Clustering Tab
clustering_elbow_plot_label_widget = None
clustering_elbow_plot_image = None
clustering_scatter_plot_label_widget = None
clustering_scatter_plot_image = None
clustering_summary_text_widget = None


# --- Helper function for color coding (listbox) ---
def get_anomaly_color_and_status(score):
    if score is None: return "white", "Unknown"
    if score == 0: return "light green", "Normal"
    if 0 < score < ANOMALY_SCORE_THRESHOLD / 2: return "LightOrange", "Minor Anomaly"
    if ANOMALY_SCORE_THRESHOLD / 2 <= score < ANOMALY_SCORE_THRESHOLD: return "OrangeRed", "Moderate Anomaly"
    if score >= ANOMALY_SCORE_THRESHOLD: return "red", "Major Anomaly"
    return "white", "Error in scoring"

def display_selected_case_details(item_list_idx):
    global loaded_df, case_anomaly_data, all_case_ids, case_detail_text_widget, status_label_var
    if not loaded_df_is_valid() or not all_case_ids or not (0 <= item_list_idx < len(all_case_ids)):
        if case_detail_text_widget:
            case_detail_text_widget.config(state="normal", bg="white")
            case_detail_text_widget.delete(1.0, tk.END)
            case_detail_text_widget.insert(tk.END, "Invalid selection or no data loaded." if loaded_df_is_valid() else "No data loaded.")
            case_detail_text_widget.config(state="disabled")
        return

    case_id_to_display = all_case_ids[item_list_idx]
    case_row = None
    if 'id_cases' in loaded_df.columns:
        try:
            df_id_type = loaded_df['id_cases'].dtype
            comp_id = case_id_to_display
            if pd.api.types.is_numeric_dtype(df_id_type): # Try to match type if DF column is numeric
                try: comp_id = pd.to_numeric(case_id_to_display)
                except ValueError: pass # Keep comp_id as string if conversion fails
            
            match = loaded_df[loaded_df['id_cases'] == comp_id]
            if match.empty: # Fallback to string comparison if typed match failed or types differ
                match = loaded_df[loaded_df['id_cases'].astype(str) == str(case_id_to_display)]
            
            if not match.empty: case_row = match.iloc[[0]]
        except Exception as e: print(f"Error matching case_id {case_id_to_display}: {e}")

    details_str = f"Displaying Case (List Index: {item_list_idx})\nID: {case_id_to_display}\n"
    if case_row is not None and not case_row.empty:
        if 'id_cases' in case_row.columns and str(case_row['id_cases'].values[0]) != str(case_id_to_display):
            details_str += f"(Original DataFrame ID: {case_row['id_cases'].values[0]})\n"
        details_str += "--- Data ---\n"
        for col in ['age_v', 'sex_v', 'greutate', 'inaltime', 'IMC', 'imcINdex', 'data1']:
            if col in case_row.columns: details_str += f"{col}: {case_row[col].values[0]}\n"
        details_str += "\n"
    else: details_str += "--- Data not found in original CSV for this ID ---\n\n"
    
    score, violations_str, status_text_for_bar = 0, "No anomalies recorded.", "Unknown"
    key_to_check = str(case_id_to_display)
    if key_to_check in case_anomaly_data:
        anomaly_info = case_anomaly_data[key_to_check]
        score = anomaly_info.get('total_score', 0)
        violations = anomaly_info.get('violations', [])
        if violations:
            violations_str = f"--- Anomalies (Score: {score:.1f}) ---\n"
            for viol in violations: violations_str += f"- Rule: '{viol['rule']}' on '{viol['column']}' (Value: {viol['value']})\n  Details: {viol['details']} [+ {viol['points']} pts]\n"
        else: violations_str = f"--- No Specific Anomalies (Score: {score:.1f}) ---"
        _, status_text_for_bar = get_anomaly_color_and_status(score) # Get status text from helper
    else:
        violations_str = f"Anomaly data not found for ID: {key_to_check}. Assuming normal."
        _, status_text_for_bar = get_anomaly_color_and_status(0) # Normal if not in dict

    if case_detail_text_widget:
        case_detail_text_widget.config(state="normal", bg="white") # Ensure background is white
        case_detail_text_widget.delete(1.0, tk.END)
        case_detail_text_widget.insert(tk.END, details_str + violations_str)
        case_detail_text_widget.config(state="disabled")
    if status_label_var: status_label_var.set(f"Case ID: {case_id_to_display} | Status: {status_text_for_bar} (Score: {score:.1f})")

def on_case_select(event):
    global case_listbox_widget
    if case_listbox_widget and case_listbox_widget.curselection(): # Check if listbox exists and has a selection
        display_selected_case_details(case_listbox_widget.curselection()[0])

def display_overall_results(anomaly_messages_summary):
    global results_text_widget
    if results_text_widget:
        results_text_widget.config(state="normal")
        results_text_widget.delete(1.0, tk.END)
        results_text_widget.insert(tk.END, "\n".join(map(str, anomaly_messages_summary)) if anomaly_messages_summary else "No summary messages.")
        results_text_widget.config(state="disabled")

def update_gui_for_file_selection(file_path):
    global selected_file_path_var, status_label_var, run_button_widget, loaded_df, case_anomaly_data, all_case_ids, case_listbox_widget, case_detail_text_widget, plot_label_widget
    loaded_df, case_anomaly_data, all_case_ids = None, {}, []
    if case_listbox_widget:
        case_listbox_widget.config(state="normal"); case_listbox_widget.delete(0, tk.END); case_listbox_widget.config(state="disabled")
    if case_detail_text_widget:
        case_detail_text_widget.config(state="normal", bg="white"); case_detail_text_widget.delete(1.0, tk.END)
        case_detail_text_widget.insert(tk.END, "Load a file and run detection."); case_detail_text_widget.config(state="disabled")
    if plot_label_widget: plot_label_widget.config(image=None, text="Plot appears here.") # Reset text

    if file_path:
        selected_file_path_var.set(file_path)
        status_label_var.set(f"File: {os.path.basename(file_path)}. Ready for detection.")
        if run_button_widget: run_button_widget.config(state="normal") # Enable anomaly detection button
        display_overall_results(["File loaded. Click 'Run Anomaly Detection' in the first tab to process."])
        try:
            loaded_df = pd.read_csv(file_path) # Load df for potential use by other tabs later
            # Initial population of all_case_ids (primarily for display count, will be refined by detection)
            if 'id_cases' in loaded_df.columns:
                ids = loaded_df['id_cases'].dropna().unique()
                try: all_case_ids = sorted(pd.to_numeric(pd.Series(ids)).astype(str).tolist())
                except ValueError: all_case_ids = sorted(pd.Series(ids).astype(str).tolist())
            else: all_case_ids = [f"RowIndex_{i}" for i in range(len(loaded_df))]
            status_label_var.set(f"File: {os.path.basename(file_path)} ({len(loaded_df)} rows). Ready.")
        except Exception as e:
            messagebox.showerror("File Load Error", f"Could not read CSV: {e}")
            selected_file_path_var.set("Error loading file."); status_label_var.set("Error loading file.")
            if run_button_widget: run_button_widget.config(state="disabled")
            loaded_df, all_case_ids = None, []
    else:
        selected_file_path_var.set("No file selected.")
        status_label_var.set("File selection cancelled.")
        if run_button_widget: run_button_widget.config(state="disabled")
        display_overall_results([])

def open_file_dialog_action():
    path = filedialog.askopenfilename(title="Select Primary CSV File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if path: update_gui_for_file_selection(path)

def clear_file_and_results_action():
    update_gui_for_file_selection(None) # Clears loaded_df, anomaly data, etc.
    # Clear IMC tab
    if imc_report_text_widget: imc_report_text_widget.config(state="normal"); imc_report_text_widget.delete(1.0, tk.END); imc_report_text_widget.config(state="disabled")
    if imc_cm_plot_label_widget: imc_cm_plot_label_widget.config(image='', text="Confusion Matrix Plot")
    # Clear Clustering tab
    if clustering_summary_text_widget: clustering_summary_text_widget.config(state="normal"); clustering_summary_text_widget.delete(1.0, tk.END); clustering_summary_text_widget.config(state="disabled")
    if clustering_elbow_plot_label_widget: clustering_elbow_plot_label_widget.config(image='', text="Elbow Plot")
    if clustering_scatter_plot_label_widget: clustering_scatter_plot_label_widget.config(image='', text="Scatter Plot")
    
    if status_label_var: status_label_var.set("All cleared. Please load a CSV file via the 'Anomaly Detection' tab.")
    display_overall_results(["All data and results cleared."])


def loaded_df_is_valid(): return loaded_df is not None and isinstance(loaded_df, pd.DataFrame) and not loaded_df.empty

def _display_image_in_label(label_widget, image_path, global_image_ref_name, max_width=400, max_height=300):
    global anomaly_plot_image, imc_cm_plot_image, clustering_elbow_plot_image, clustering_scatter_plot_image

    if not image_path or not os.path.exists(image_path): # Check if image_path is None or empty
        label_widget.config(image='', text=f"Plot not found or path invalid:\n{os.path.basename(str(image_path)) if image_path else 'No Path'}")
        return
    try:
        img = Image.open(image_path)
        label_widget.update_idletasks()
        widget_width = label_widget.winfo_width()
        widget_height = label_widget.winfo_height()

        if widget_width > 1 and widget_height > 1: 
             current_max_width = widget_width - 10 
             current_max_height = widget_height - 10 
        else: 
            current_max_width = max_width
            current_max_height = max_height
        
        img.thumbnail((current_max_width, current_max_height), Image.Resampling.LANCZOS)
        
        photo_image = ImageTk.PhotoImage(img)

        if global_image_ref_name == "anomaly_plot_image": anomaly_plot_image = photo_image
        elif global_image_ref_name == "imc_cm_plot_image": imc_cm_plot_image = photo_image
        elif global_image_ref_name == "clustering_elbow_plot_image": clustering_elbow_plot_image = photo_image
        elif global_image_ref_name == "clustering_scatter_plot_image": clustering_scatter_plot_image = photo_image
        else: label_widget.config(image='', text="Invalid image ref"); return
        
        label_widget.config(image=photo_image, text="")
    except Exception as e:
        print(f"Error displaying image {image_path}: {e}")
        label_widget.config(image='', text=f"Error loading plot:\n{os.path.basename(image_path)}")


def run_detection_action():
    global selected_file_path_var, status_label_var, run_button_widget, load_button_widget, root_window, loaded_df, case_anomaly_data, all_case_ids, case_listbox_widget, plot_label_widget
    file_to_process = selected_file_path_var.get()
    if not file_to_process or not os.path.exists(file_to_process):
        messagebox.showerror("Input Error", "No valid CSV file loaded."); return
    if not loaded_df_is_valid(): 
        try: loaded_df = pd.read_csv(file_to_process)
        except Exception as e: messagebox.showerror("File Read Error", f"Could not read CSV: {e}"); return
        if not loaded_df_is_valid(): messagebox.showerror("File Error", "CSV is empty/invalid."); return
    
    run_button_widget.config(state="disabled"); load_button_widget.config(state="disabled")
    status_label_var.set(f"Detecting anomalies in {os.path.basename(file_to_process)}..."); root_window.update_idletasks()
    try:
        anomaly_messages, _, detected_data = detect_anomalies_from_file(file_to_process)
        case_anomaly_data = {str(k): v for k, v in detected_data.items()}
        display_overall_results(anomaly_messages)
        if not any(m.startswith("CRITICAL ERROR:") for m in anomaly_messages):
            status_label_var.set(f"Detection complete for {os.path.basename(file_to_process)}.")
            try: save_filtered_datasets(pd.read_csv(file_to_process), case_anomaly_data, ANOMALY_SCORE_THRESHOLD, os.path.basename(file_to_process))
            except Exception as e: print(f"Error saving filtered datasets: {e}")
            
            if case_anomaly_data:
                keys = list(case_anomaly_data.keys()) 
                try: 
                    num_keys = sorted([int(k) for k in keys if k.isdigit()])
                    str_keys_rowindex = sorted([k for k in keys if not k.isdigit() and k.startswith("RowIndex_")])
                    str_keys_other = sorted([k for k in keys if not k.isdigit() and not k.startswith("RowIndex_")])
                    all_case_ids = [str(k) for k in num_keys] + str_keys_other + str_keys_rowindex
                except: all_case_ids = sorted(keys) 
            elif loaded_df_is_valid() and 'id_cases' in loaded_df.columns: 
                ids = loaded_df['id_cases'].dropna().unique()
                try: all_case_ids = sorted(pd.to_numeric(pd.Series(ids)).astype(str).tolist())
                except ValueError: all_case_ids = sorted(pd.Series(ids).astype(str).tolist())
            else: all_case_ids = [f"RowIndex_{i}" for i in range(len(loaded_df))] if loaded_df_is_valid() else []

            if case_listbox_widget:
                case_listbox_widget.config(state="normal"); case_listbox_widget.delete(0, tk.END)
                if all_case_ids:
                    for cid_str in all_case_ids: case_listbox_widget.insert(tk.END, cid_str)
                    root_window.update_idletasks() 
                    if case_listbox_widget.size() > 0:
                        for i in range(case_listbox_widget.size()): 
                            if i < len(all_case_ids): 
                                score = case_anomaly_data.get(all_case_ids[i], {}).get('total_score', 0)
                                clr, _ = get_anomaly_color_and_status(score)
                                fg_clr = "white" if clr.lower() in ["red", "orangered"] else "black"
                                try: case_listbox_widget.itemconfig(i, bg=clr, fg=fg_clr)
                                except tk.TclError as e: print(f"TclError itemconfig idx {i} for '{all_case_ids[i]}': {e}")
                        case_listbox_widget.selection_set(0); display_selected_case_details(0)
                    else: case_listbox_widget.config(state="disabled") 
                else: case_listbox_widget.config(state="disabled")
            _display_image_in_label(plot_label_widget, "anomaly_plot.png", "anomaly_plot_image", max_width=500, max_height=400)
        else: status_label_var.set("Critical error during detection.")
    except Exception as e:
        messagebox.showerror("Processing Error", f"Error during detection: {e}"); print(traceback.format_exc())
    finally: run_button_widget.config(state="normal"); load_button_widget.config(state="normal")


def run_imc_classification_action():
    global selected_file_path_var, status_label_var, imc_report_text_widget, imc_cm_plot_label_widget, root_window
    
    original_file = selected_file_path_var.get()
    if not original_file or not os.path.exists(original_file):
        messagebox.showerror("Input Error", "Please load a primary CSV file first (via Anomaly Detection tab).")
        return

    base_name = os.path.splitext(os.path.basename(original_file))[0]
    original_dir = os.path.dirname(original_file) if os.path.dirname(original_file) else "."
    cleaned_file_path = os.path.join(original_dir, f"{base_name}_cleaned.csv")
    
    if not os.path.exists(cleaned_file_path):
        cleaned_file_path_cwd = f"{base_name}_cleaned.csv" # Check CWD as well
        if os.path.exists(cleaned_file_path_cwd):
            cleaned_file_path = cleaned_file_path_cwd
        else:
            messagebox.showerror("File Not Found", f"Cleaned data file '{os.path.basename(cleaned_file_path)}' not found. Please run anomaly detection first to generate it.")
            return

    status_label_var.set(f"Running IMC Classification on {os.path.basename(cleaned_file_path)}..."); root_window.update_idletasks()
    report_str, cm_img_path, error_msg = run_imc_classification(cleaned_file_path) # Use cleaned file
    
    if imc_report_text_widget:
        imc_report_text_widget.config(state="normal"); imc_report_text_widget.delete(1.0, tk.END)
        if error_msg:
            imc_report_text_widget.insert(tk.END, f"Error during IMC Classification:\n{error_msg}")
            status_label_var.set("IMC Classification failed.")
        elif report_str:
            imc_report_text_widget.insert(tk.END, "Classification Report (on Cleaned Data):\n" + report_str)
            status_label_var.set("IMC Classification complete.")
        else:
            imc_report_text_widget.insert(tk.END, "IMC Classification did not produce a report.")
            status_label_var.set("IMC Classification finished with no report.")
        imc_report_text_widget.config(state="disabled")

    if imc_cm_plot_label_widget: 
        if cm_img_path:
            _display_image_in_label(imc_cm_plot_label_widget, cm_img_path, "imc_cm_plot_image", max_width=400, max_height=320) # Increased size
        else:
            imc_cm_plot_label_widget.config(image='', text="Confusion matrix plot not generated.")
        
def run_clustering_action():
    global selected_file_path_var, status_label_var, clustering_elbow_plot_label_widget, clustering_scatter_plot_label_widget, clustering_summary_text_widget, root_window
    
    original_file = selected_file_path_var.get()
    if not original_file or not os.path.exists(original_file):
        messagebox.showerror("Input Error", "Please load and run anomaly detection on a primary CSV file first.")
        return

    base_name = os.path.splitext(os.path.basename(original_file))[0]
    original_dir = os.path.dirname(original_file) if os.path.dirname(original_file) else "." 
    cleaned_file_path = os.path.join(original_dir, f"{base_name}_cleaned.csv")
    
    if not os.path.exists(cleaned_file_path):
        cleaned_file_path_cwd = f"{base_name}_cleaned.csv"
        if os.path.exists(cleaned_file_path_cwd):
            cleaned_file_path = cleaned_file_path_cwd
        else:
            messagebox.showerror("File Not Found", f"Cleaned data file '{os.path.basename(cleaned_file_path)}' not found. Please run anomaly detection first.")
            return

    status_label_var.set(f"Running Clustering on {os.path.basename(cleaned_file_path)}..."); root_window.update_idletasks()
    features = ["age_v", "sex_v", "greutate", "inaltime"] 
    other_numeric = ["age_v", "greutate", "inaltime"]
    k_opt = 3

    elbow_p, scatter_p, summary_s, sil_score, error_c = run_clustering_pipeline(
        filepath=cleaned_file_path, features_to_cluster=features, other_numeric_features=other_numeric, k_optimal=k_opt
    )

    if clustering_summary_text_widget:
        clustering_summary_text_widget.config(state="normal"); clustering_summary_text_widget.delete(1.0, tk.END)
        if error_c:
            clustering_summary_text_widget.insert(tk.END, f"Error during Clustering:\n{error_c}")
            status_label_var.set("Clustering failed.")
        elif summary_s:
            summary_display = "Cluster Analysis Summary (on Cleaned Data):\n" + summary_s
            if sil_score is not None: summary_display += f"\n\nSilhouette Score (k={k_opt}): {sil_score:.3f}"
            clustering_summary_text_widget.insert(tk.END, summary_display)
            status_label_var.set("Clustering complete.")
        else:
            clustering_summary_text_widget.insert(tk.END, "Clustering did not produce a summary.")
            status_label_var.set("Clustering finished with no summary.")
        clustering_summary_text_widget.config(state="disabled")

    # Use larger dimensions for clustering plots
    if clustering_elbow_plot_label_widget:
        if elbow_p: _display_image_in_label(clustering_elbow_plot_label_widget, elbow_p, "clustering_elbow_plot_image", max_width=450,max_height=350)
        else: clustering_elbow_plot_label_widget.config(image='', text="Elbow plot not generated.")
    if clustering_scatter_plot_label_widget:
        if scatter_p: _display_image_in_label(clustering_scatter_plot_label_widget, scatter_p, "clustering_scatter_plot_image", max_width=450,max_height=350)
        else: clustering_scatter_plot_label_widget.config(image='', text="Cluster scatter plot not generated.")


def exit_application_action():
    if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
        if root_window: root_window.destroy()

def create_main_application_window():
    global selected_file_path_var, status_label_var, run_button_widget, load_button_widget, results_text_widget, root_window, case_listbox_widget, case_detail_text_widget, plot_label_widget
    global imc_report_text_widget, imc_cm_plot_label_widget, clustering_elbow_plot_label_widget, clustering_scatter_plot_label_widget, clustering_summary_text_widget

    root_window = tk.Tk()
    root_window.title("Doctor31 Comprehensive Analyzer - v0.4.3") 
    root_window.geometry("1150x850") # Slightly larger window

    style = ttk.Style(root_window)
    if 'clam' in style.theme_names(): style.theme_use('clam')

    # Top Controls Frame
    controls_frame = ttk.LabelFrame(root_window, text="Input File Management", padding="10") 
    controls_frame.pack(side="top", fill="x", padx=10, pady=(10,5))
    load_button_widget = ttk.Button(controls_frame, text="Load Primary CSV", command=open_file_dialog_action) 
    load_button_widget.pack(side="left", padx=(0,10), pady=5)
    selected_file_path_var = tk.StringVar(value="No file selected.")
    file_path_label = ttk.Label(controls_frame, textvariable=selected_file_path_var, relief="sunken", anchor="w", padding=3, wraplength=500)
    file_path_label.pack(side="left", fill="x", expand=True, pady=5, padx=(0,10))
    clear_button_widget = ttk.Button(controls_frame, text="Clear All Data & Results", command=clear_file_and_results_action) 
    clear_button_widget.pack(side="left", pady=5, padx=(5,0)) 
    
    notebook = ttk.Notebook(root_window)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # --- Tab 1: Anomaly Detection ---
    anomaly_tab = ttk.Frame(notebook, padding="5")
    notebook.add(anomaly_tab, text='Rule-Based Anomaly Detection')
    run_button_widget = ttk.Button(anomaly_tab, text="Run Anomaly Detection on Loaded CSV", command=run_detection_action, state="disabled") 
    run_button_widget.pack(pady=(5,10), anchor="center")
    main_paned_window = ttk.PanedWindow(anomaly_tab, orient=tk.HORIZONTAL)
    main_paned_window.pack(fill=tk.BOTH, expand=True)
    left_pane_frame = ttk.Frame(main_paned_window, padding="5")
    main_paned_window.add(left_pane_frame, weight=2) 
    case_list_outer_frame = ttk.LabelFrame(left_pane_frame, text="Cases", padding="5")
    case_list_outer_frame.pack(fill="both", pady=(0,5), expand=True) 
    case_list_y_scrollbar = ttk.Scrollbar(case_list_outer_frame, orient="vertical")
    case_listbox_widget = tk.Listbox(case_list_outer_frame, yscrollcommand=case_list_y_scrollbar.set, exportselection=False, height=10, selectbackground="#0078D7", selectforeground="white") 
    case_list_y_scrollbar.config(command=case_listbox_widget.yview)
    case_listbox_widget.pack(side="left", fill="both", expand=True); case_list_y_scrollbar.pack(side="right", fill="y")
    case_listbox_widget.bind("<<ListboxSelect>>", on_case_select); case_listbox_widget.config(state="disabled")
    case_detail_frame = ttk.LabelFrame(left_pane_frame, text="Selected Case Details", padding="5")
    case_detail_frame.pack(fill="both", expand=True)
    case_detail_text_widget = tk.Text(case_detail_frame, wrap="word", height=10, width=50, borderwidth=1, relief="sunken", font=("Arial", 9), bg="white") 
    case_detail_y_scrollbar_td = ttk.Scrollbar(case_detail_frame, orient="vertical", command=case_detail_text_widget.yview)
    case_detail_text_widget.configure(yscrollcommand=case_detail_y_scrollbar_td.set)
    case_detail_text_widget.pack(side="left", fill="both", expand=True); case_detail_y_scrollbar_td.pack(side="right", fill="y")
    case_detail_text_widget.insert(tk.END, "Details appear here after running detection and selecting a case."); case_detail_text_widget.config(state="disabled")
    right_pane_frame = ttk.Frame(main_paned_window, padding="5")
    main_paned_window.add(right_pane_frame, weight=3) 
    plot_display_frame = ttk.LabelFrame(right_pane_frame, text="Anomaly Score Distribution Plot", padding="10")
    plot_display_frame.pack(fill="both", expand=True, pady=(0,5))
    plot_label_widget = ttk.Label(plot_display_frame, text="Plot appears here after detection.", anchor="center")
    plot_label_widget.pack(fill="both", expand=True)
    results_outer_frame = ttk.LabelFrame(right_pane_frame, text="Overall Detection Summary", padding="10")
    results_outer_frame.pack(fill="both", expand=True, pady=5)
    results_text_widget = tk.Text(results_outer_frame, wrap="word", height=8, width=60, borderwidth=1, relief="sunken", font=("Arial", 9)) 
    y_scrollbar_res = ttk.Scrollbar(results_outer_frame, orient="vertical", command=results_text_widget.yview)
    results_text_widget.configure(yscrollcommand=y_scrollbar_res.set)
    results_text_widget.pack(side="left", fill="both", expand=True); y_scrollbar_res.pack(side="right", fill="y")
    results_text_widget.insert(tk.END, "Summary of detection messages appears here."); results_text_widget.config(state="disabled")

    # --- Tab 2: IMC Classification ---
    imc_tab = ttk.Frame(notebook, padding="10")
    notebook.add(imc_tab, text='IMC Classification (Random Forest)')
    run_imc_button = ttk.Button(imc_tab, text="Run IMC Classification (on Cleaned Data)", command=run_imc_classification_action) 
    run_imc_button.pack(pady=10)
    imc_results_frame = ttk.Frame(imc_tab)
    imc_results_frame.pack(fill="both", expand=True)
    imc_report_text_widget = tk.Text(imc_results_frame, wrap="word", height=15, width=70, font=("Courier", 9)) 
    imc_report_text_widget.pack(side="left", fill="both", expand=True, padx=(0,5))
    imc_cm_plot_label_widget = ttk.Label(imc_results_frame, text="Confusion Matrix Plot", anchor="center", relief="groove", borderwidth=2)
    imc_cm_plot_label_widget.pack(side="right", fill="both", expand=True, padx=(5,0))
    imc_report_text_widget.insert(tk.END, "Click 'Run IMC Classification'. Operates on the '*_cleaned.csv' file generated by anomaly detection."); imc_report_text_widget.config(state="disabled")

    # --- Tab 3: Clustering Analysis ---
    clustering_tab = ttk.Frame(notebook, padding="10")
    notebook.add(clustering_tab, text='Clustering Analysis (K-Means)')
    run_clustering_button = ttk.Button(clustering_tab, text="Run Clustering (on '*_cleaned.csv')", command=run_clustering_action)
    run_clustering_button.pack(pady=10)
    
    # Main frame for clustering results, allowing plots and text to share space better
    clustering_results_main_frame = ttk.Frame(clustering_tab)
    clustering_results_main_frame.pack(fill="both", expand=True, pady=5)

    clustering_plots_frame = ttk.Frame(clustering_results_main_frame) 
    clustering_plots_frame.pack(fill="both", expand=True, side="top", pady=5) # Allow this frame to expand

    clustering_elbow_plot_label_widget = ttk.Label(clustering_plots_frame, text="Elbow Plot", anchor="center", relief="groove", borderwidth=2)
    clustering_elbow_plot_label_widget.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    clustering_scatter_plot_label_widget = ttk.Label(clustering_plots_frame, text="Scatter Plot", anchor="center", relief="groove", borderwidth=2)
    clustering_scatter_plot_label_widget.pack(side="right", fill="both", expand=True, padx=5, pady=5)
    
    clustering_summary_text_widget = tk.Text(clustering_results_main_frame, wrap="word", height=10, width=80, font=("Courier", 9)) # Height can be adjusted
    clustering_summary_text_widget.pack(fill="both", expand=True, pady=(5,0), side="bottom")
    clustering_summary_text_widget.insert(tk.END, "Click 'Run Clustering'. Operates on the '*_cleaned.csv' file generated by anomaly detection."); clustering_summary_text_widget.config(state="disabled")
    
    # Status Bar
    status_bar = ttk.Frame(root_window, relief="sunken", padding=(2,2))
    status_bar.pack(side="bottom", fill="x")
    status_label_var = tk.StringVar(value="Ready. Load a Primary CSV via the 'Anomaly Detection' tab.")
    status_display_label = ttk.Label(status_bar, textvariable=status_label_var, anchor="w")
    status_display_label.pack(side="left", fill="x", expand=True, padx=5)
    
    root_window.protocol("WM_DELETE_WINDOW", exit_application_action)
    root_window.mainloop()

if __name__ == "__main__":
    try: from PIL import Image, ImageTk
    except ImportError:
        print("Pillow library not found. Please install it: pip install Pillow")
        if tk._default_root is None: 
            root_err = tk.Tk(); root_err.withdraw()
            messagebox.showerror("Dependency Error", "Pillow library not found. Please install it: pip install Pillow")
            root_err.destroy()
        exit()
    create_main_application_window()
