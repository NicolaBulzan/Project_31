import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time  
import os   

selected_file_path_var = None
status_label_var = None
run_button_widget = None
load_button_widget = None
results_text_widget = None
root_window = None 

def mock_detect_anomalies(file_path_for_detection):
    """
    This function simulates the anomaly detection process.
    It will be replaced by the actual function 
    It takes a file path and returns a list of strings describing mock anomalies.
    """
    global status_label_var, root_window

    if status_label_var and root_window:
        status_label_var.set(f"Mock Processing: {os.path.basename(file_path_for_detection)}... Please wait...")
        root_window.update_idletasks() 

    time.sleep(2.5)  
    filename_only = os.path.basename(file_path_for_detection)
    mock_results = [
        f"--- Mock Anomalies for: {filename_only} ---",
        "Right now this function is not available"
    ]
    return mock_results


def display_results(anomalies_list):
 
    global results_text_widget
    if results_text_widget:
        current_state = results_text_widget.cget("state")
        results_text_widget.config(state="normal")  
        results_text_widget.delete(1.0, tk.END)     
        if anomalies_list:
            for anomaly_info in anomalies_list:
                results_text_widget.insert(tk.END, str(anomaly_info) + "\n\n") 
        else:
            results_text_widget.insert(tk.END, "No anomalies detected or an error occurred.")
        results_text_widget.config(state=current_state) 

def update_gui_for_file_selection(file_path):
    """Updates GUI elements after a file is selected or selection is cleared."""
    global selected_file_path_var, status_label_var, run_button_widget
    
    if file_path:
        selected_file_path_var.set(file_path) 
        status_label_var.set(f"File loaded: {os.path.basename(file_path)}")
        if run_button_widget:
            run_button_widget.config(state="normal") 
        display_results(["File loaded successfully. Click 'Run Detection' to (mock) process."])
    else:
        selected_file_path_var.set("No file selected.")
        status_label_var.set("File selection cancelled. Please load a CSV file.")
        if run_button_widget:
            run_button_widget.config(state="disabled") 
        display_results([]) 

def open_file_dialog_action():
    """Handles the 'Load CSV File' button click and menu action."""
    file_path = filedialog.askopenfilename(
        title="Select CSV File for Anomaly Detection",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    update_gui_for_file_selection(file_path)

def clear_file_and_results_action():
    """Clears the selected file path and the results display area."""
    update_gui_for_file_selection(None) 
    if status_label_var: 
        status_label_var.set("Ready. Please load a CSV file.")


def run_detection_action():
    """Handles the 'Run Detection' button click. Calls the (mock) backend."""
    global selected_file_path_var, status_label_var, run_button_widget, load_button_widget, root_window

    file_to_process = selected_file_path_var.get()
    if not file_to_process or file_to_process == "No file selected.":
        messagebox.showerror("Input Error", "No CSV file has been loaded. Please load a file first.")
        status_label_var.set("Error: No file loaded. Please use 'File > Load CSV File...'.")
        return

    # Disable buttons to prevent multiple clicks during processing
    if run_button_widget: run_button_widget.config(state="disabled")
    if load_button_widget: load_button_widget.config(state="disabled")
    
    status_label_var.set(f"Starting detection for: {os.path.basename(file_to_process)}...")
    root_window.update_idletasks() 
    try:
        # === INTEGRATION POINT: Replace with call to Team 2's actual function ===
        # Example: detected_anomalies = classification_module.detect_all_anomalies(file_to_process)
        detected_anomalies = mock_detect_anomalies(file_to_process)
    

        display_results(detected_anomalies)
        if detected_anomalies and "Mock: No anomalies found" not in detected_anomalies[0]:
             status_label_var.set(f"Mock detection complete for {os.path.basename(file_to_process)}. {len(detected_anomalies)-1} anomalies found.")
        else:
            status_label_var.set(f"Mock detection complete for {os.path.basename(file_to_process)}. No anomalies found.")

    except Exception as e:
        error_msg = f"Error during mock detection: {str(e)}"
        messagebox.showerror("Processing Error", error_msg)
        status_label_var.set(error_msg)
        display_results([error_msg]) # Display error in results area
    finally:
        
        if run_button_widget: run_button_widget.config(state="normal")
        if load_button_widget: load_button_widget.config(state="normal")

def exit_application_action():
    """Handles the application exit."""
    if messagebox.askokcancel("Quit Doctor31 Anomaly Detector", "Are you sure you want to quit?"):
        root_window.destroy()

# --- Main GUI Setup Function ---
def create_main_application_window():
    global selected_file_path_var, status_label_var, run_button_widget, load_button_widget, results_text_widget, root_window

    root_window = tk.Tk()
    root_window.title("Doctor31 Anomaly Detector - v0.1 (Mock Backend)")
    root_window.geometry("850x650") # Adjusted size


    style = ttk.Style(root_window)
    available_themes = style.theme_names()
   
    if 'clam' in available_themes:
        style.theme_use('clam')
    elif 'alt' in available_themes:
        style.theme_use('alt')
    
    menubar = tk.Menu(root_window)
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Load CSV File...", command=open_file_dialog_action, accelerator="Ctrl+O")
    file_menu.add_command(label="Clear File & Results", command=clear_file_and_results_action)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=exit_application_action, accelerator="Ctrl+Q")
    menubar.add_cascade(label="File", menu=file_menu)
    root_window.config(menu=menubar)
    
    root_window.bind_all("<Control-o>", lambda event: open_file_dialog_action())
    root_window.bind_all("<Control-q>", lambda event: exit_application_action())

    root_window.protocol("WM_DELETE_WINDOW", exit_application_action)

    controls_frame = ttk.LabelFrame(root_window, text="Input and Actions", padding="10")
    controls_frame.pack(side="top", fill="x", padx=10, pady=(10, 5))

    load_button_widget = ttk.Button(controls_frame, text="1. Load CSV", command=open_file_dialog_action)
    load_button_widget.pack(side="left", padx=(0, 10), pady=5)

    selected_file_path_var = tk.StringVar(value="No file selected.")
    file_path_label = ttk.Label(controls_frame, textvariable=selected_file_path_var, relief="sunken", anchor="w", padding=3, wraplength=500)
    file_path_label.pack(side="left", fill="x", expand=True, pady=5, padx=(0,10))

    run_button_widget = ttk.Button(controls_frame, text="2. Run Detection", command=run_detection_action, state="disabled")
    run_button_widget.pack(side="left", pady=5, padx=(0,5))
    
    clear_button_widget = ttk.Button(controls_frame, text="Clear", command=clear_file_and_results_action)
    clear_button_widget.pack(side="left", pady=5)


    results_outer_frame = ttk.LabelFrame(root_window, text="Detected Anomalies", padding="10")
    results_outer_frame.pack(side="top", fill="both", expand=True, padx=10, pady=5)

    results_text_widget = tk.Text(results_outer_frame, wrap="word", height=20, width=100, borderwidth=1, relief="sunken", font=("Arial", 10))
    y_scrollbar = ttk.Scrollbar(results_outer_frame, orient="vertical", command=results_text_widget.yview)
    results_text_widget.configure(yscrollcommand=y_scrollbar.set)
    
    results_text_widget.pack(side="left", fill="both", expand=True)
    y_scrollbar.pack(side="right", fill="y")
    
    results_text_widget.insert(tk.END, "Welcome to Doctor31 Anomaly Detector!\nPlease load a CSV file using 'File > Load CSV File...' or the button above.\nThen click 'Run Detection' to (currently) see mock results.")
    results_text_widget.config(state="disabled") 
    status_bar = ttk.Frame(root_window, relief="sunken", padding=(2, 2))
    status_bar.pack(side="bottom", fill="x")
    
    status_label_var = tk.StringVar(value="Ready. Please load a CSV file.")
    status_display_label = ttk.Label(status_bar, textvariable=status_label_var, anchor="w")
    status_display_label.pack(side="left", fill="x", expand=True, padx=5)

    root_window.mainloop()

if __name__ == "__main__":
    create_main_application_window()