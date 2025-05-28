import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration for Anomaly Rules & Scoring ---
POINTS_CRITICAL = 3
POINTS_HIGH = 2
POINTS_MEDIUM = 1
POINTS_LOW = 0.5 

MAX_AGE = 120
MIN_ADULT_WEIGHT_KG = 20 
MIN_CHILD_WEIGHT_KG = 5 
MIN_HEIGHT_CM = 50     
MAX_HEIGHT_CM = 230    
MIN_BMI = 12
MAX_BMI = 60
ELDERLY_AGE_THRESHOLD = 85
DUPLICATE_TIMEFRAME_HOURS = 1
CHILD_AGE_THRESHOLD = 16

ANOMALY_SCORE_THRESHOLD = 3.0 #

# --- Helper Functions ---
def parse_timestamp(timestamp_str):
    if pd.isna(timestamp_str):
        return None
    try:
        return datetime.strptime(str(timestamp_str), '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            return datetime.strptime(str(timestamp_str), '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            return None

def add_anomaly(case_scores_dict, case_id, column_affected, rule_description, points, actual_value, details=""):
    if case_id not in case_scores_dict:
        case_scores_dict[case_id] = {'total_score': 0, 'violations': []}
    
    violation_details = {
        "column": column_affected,
        "rule": rule_description,
        "value": str(actual_value),
        "details": details,
        "points": points
    }
    case_scores_dict[case_id]['violations'].append(violation_details)
    case_scores_dict[case_id]['total_score'] += points

# --- Main Detection Function ---
def detect_anomalies_from_file(file_path):
    case_anomaly_data = {}
    error_messages = []

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        error_messages.append(f"CRITICAL ERROR: File not found at '{file_path}'")
        return error_messages, [], {} 
    except Exception as e:
        error_messages.append(f"CRITICAL ERROR: Could not read file '{file_path}'. Reason: {e}")
        return error_messages, [], {}

    required_cols = ['id_cases', 'age_v', 'greutate', 'inaltime', 'IMC', 'data1', 'imcINdex']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        error_messages.append(f"CRITICAL ERROR: Missing required columns in CSV: {', '.join(missing_cols)}")
        return error_messages, [], {}

    for index, row in df.iterrows():
        case_id = row.get('id_cases', f"RowIndex_{index}")

        try:
            age = pd.to_numeric(row.get('age_v'), errors='coerce')
            weight = pd.to_numeric(row.get('greutate'), errors='coerce')
            height_cm = pd.to_numeric(row.get('inaltime'), errors='coerce')
            bmi_original = pd.to_numeric(row.get('IMC'), errors='coerce')
            bmi_category = str(row.get('imcINdex', '')).strip().lower()
        except Exception as e:
            add_anomaly(case_anomaly_data, case_id, "Row Data", "Data Parsing Error", POINTS_CRITICAL, str(e))
            continue

        if pd.isna(age):
            add_anomaly(case_anomaly_data, case_id, "age_v", "Missing Age", POINTS_MEDIUM, "NaN")
        elif age > MAX_AGE or age < 0:
            add_anomaly(case_anomaly_data, case_id, "age_v", "Impossible Age", POINTS_CRITICAL, age, f"Outside 0-{MAX_AGE} range")

        if pd.isna(weight):
            add_anomaly(case_anomaly_data, case_id, "greutate", "Missing Weight", POINTS_MEDIUM, "NaN")
        elif not pd.isna(age):
            if age > CHILD_AGE_THRESHOLD and weight < MIN_ADULT_WEIGHT_KG:
                add_anomaly(case_anomaly_data, case_id, "greutate", "Implausible Adult Weight", POINTS_HIGH, weight, f"< {MIN_ADULT_WEIGHT_KG}kg for age {age}")
            elif age <= CHILD_AGE_THRESHOLD and weight < MIN_CHILD_WEIGHT_KG and weight > 0 :
                add_anomaly(case_anomaly_data, case_id, "greutate", "Very Low Child Weight", POINTS_MEDIUM, weight, f"< {MIN_CHILD_WEIGHT_KG}kg for age {age}")
            if weight <= 0:
                 add_anomaly(case_anomaly_data, case_id, "greutate", "Invalid Weight (<=0kg)", POINTS_CRITICAL, weight)

        if pd.isna(height_cm):
            add_anomaly(case_anomaly_data, case_id, "inaltime", "Missing Height", POINTS_MEDIUM, "NaN")
        elif height_cm < MIN_HEIGHT_CM or height_cm > MAX_HEIGHT_CM:
            add_anomaly(case_anomaly_data, case_id, "inaltime", "Unlikely Height", POINTS_HIGH, height_cm, f"Outside {MIN_HEIGHT_CM}-{MAX_HEIGHT_CM}cm range")

        current_bmi_to_check = None
        bmi_source_info = ""
        if not pd.isna(bmi_original):
            current_bmi_to_check = bmi_original
            bmi_source_info = " (original)"
        elif pd.notna(weight) and pd.notna(height_cm) and height_cm > 0:
            height_m = height_cm / 100.0
            try:
                current_bmi_to_check = weight / (height_m ** 2)
                bmi_source_info = " (calculated)"
            except ZeroDivisionError:
                add_anomaly(case_anomaly_data, case_id, "IMC", "BMI Calculation Error", POINTS_MEDIUM, "Height is 0")
            except Exception:
                add_anomaly(case_anomaly_data, case_id, "IMC", "BMI Calculation Error", POINTS_MEDIUM, "Unknown error during calculation")
        elif pd.isna(bmi_original):
             add_anomaly(case_anomaly_data, case_id, "IMC", "Missing BMI (and cannot calculate)", POINTS_MEDIUM, "NaN")

        if pd.notna(current_bmi_to_check):
            if current_bmi_to_check < MIN_BMI or current_bmi_to_check > MAX_BMI:
                add_anomaly(case_anomaly_data, case_id, "IMC", f"BMI Incompatible with Life{bmi_source_info}", POINTS_CRITICAL, f"{current_bmi_to_check:.1f}", f"Outside {MIN_BMI}-{MAX_BMI} range")

        if pd.notna(age) and age > ELDERLY_AGE_THRESHOLD:
            if pd.isna(bmi_category) or bmi_category == "":
                add_anomaly(case_anomaly_data, case_id, "imcINdex", "Missing BMI Category for Elderly", POINTS_LOW, "NaN", f"Age: {age}")
            elif "obese" in bmi_category:
                add_anomaly(case_anomaly_data, case_id, "imcINdex", "Suspicious Elderly Obesity", POINTS_MEDIUM, bmi_category, f"Age: {age}")

    df['parsed_data1'] = df['data1'].apply(parse_timestamp)
    df_valid_for_duplicates = df.dropna(subset=['id_cases', 'age_v', 'greutate', 'inaltime', 'parsed_data1']).copy()
    
    df_valid_for_duplicates['duplicate_key_tuple'] = df_valid_for_duplicates.apply(
        lambda x: (int(x['age_v']), float(x['greutate']), float(x['inaltime']))
        if pd.notna(x['age_v']) and pd.notna(x['greutate']) and pd.notna(x['inaltime'])
        else None,
        axis=1
    )
    df_valid_for_duplicates.dropna(subset=['duplicate_key_tuple'], inplace=True)

    duplicate_relationships = {} 
    if not df_valid_for_duplicates.empty:
        grouped = df_valid_for_duplicates.sort_values('parsed_data1').groupby('duplicate_key_tuple')
        
        for key_tuple_val, group in grouped:
            if len(group) > 1:
                group_records = group[['id_cases', 'parsed_data1']].to_dict('records')
                for i in range(len(group_records)):
                    for j in range(i + 1, len(group_records)):
                        record1 = group_records[i]
                        record2 = group_records[j]
                        id1, time1 = record1['id_cases'], record1['parsed_data1']
                        id2, time2 = record2['id_cases'], record2['parsed_data1']
                        if pd.isna(time1) or pd.isna(time2): continue
                        time_diff = abs(time1 - time2)
                        if time_diff < timedelta(hours=DUPLICATE_TIMEFRAME_HOURS):
                            duplicate_relationships.setdefault(id1, set()).add(id2)
                            duplicate_relationships.setdefault(id2, set()).add(id1)

    for case_id_in_duplicate_cluster, partners in duplicate_relationships.items():
        if partners:
            original_row_for_key_series = df[df['id_cases'] == case_id_in_duplicate_cluster]
            if not original_row_for_key_series.empty:
                original_row_for_key = original_row_for_key_series.iloc[0]
                key_details_for_msg = f"A/W/H: {original_row_for_key.get('age_v', 'N/A')}/{original_row_for_key.get('greutate', 'N/A')}/{original_row_for_key.get('inaltime', 'N/A')}"
                details_str = f"Part of a duplicate cluster. Matches cases: {', '.join(map(str, sorted(list(partners))))}. Key: {key_details_for_msg}."
                
                already_flagged_for_duplicate = False
                if case_id_in_duplicate_cluster in case_anomaly_data:
                    for viol in case_anomaly_data[case_id_in_duplicate_cluster]['violations']:
                        if viol['rule'] == "Potential Duplicate Cluster":
                            already_flagged_for_duplicate = True
                            break
                if not already_flagged_for_duplicate:
                    add_anomaly(case_anomaly_data, case_id_in_duplicate_cluster, 
                                "Multiple Columns", "Potential Duplicate Cluster", 
                                POINTS_MEDIUM, "Matches other records", details_str)
            # else:
                # This case_id from duplicate_relationships was not in the original df? Should not happen if id_cases is consistent.
                # print(f"Warning: Case ID {case_id_in_duplicate_cluster} from duplicate check not found in original DataFrame for details.")


    final_anomaly_messages = []
    all_total_scores = []
    if not case_anomaly_data and not error_messages:
        final_anomaly_messages.append("No anomalies detected based on the current rules.")
    elif error_messages:
        return error_messages, [], case_anomaly_data # Return case_anomaly_data even on file error for potential partial processing
    else:
        sorted_cases = sorted(case_anomaly_data.items(), key=lambda item: item[1]['total_score'], reverse=True)
        for case_id, data in sorted_cases:
            if data['violations']:
                all_total_scores.append(data['total_score'])
                header = f"--- Anomalies for Case ID {case_id} (Total Anomaly Score: {data['total_score']:.1f}) ---"
                final_anomaly_messages.append(header)
                for viol in data['violations']:
                    final_anomaly_messages.append(
                        f"  - Rule: '{viol['rule']}' on '{viol['column']}' (Value: {viol['value']}) "
                        f"[+{viol['points']} pts]. Details: {viol['details']}"
                    )
                final_anomaly_messages.append("") 
            
    return final_anomaly_messages, all_total_scores, case_anomaly_data


def save_filtered_datasets(original_df, case_anomaly_data_dict, threshold, input_file_name):
    """
    Filters the original DataFrame based on anomaly scores and saves two CSVs:
    one for cleaned data and one for anomalous data.
    """
    ids_to_exclude = set()
    for case_id, data in case_anomaly_data_dict.items():
        if data['total_score'] >= threshold:
            ids_to_exclude.add(case_id)

    # Ensure id_cases is the correct type for comparison if it was read as float/object
    if 'id_cases' in original_df.columns:
        try:
            # Attempt to convert to int, then to str for robust comparison if IDs are mixed types
            original_df['id_cases_str_temp'] = original_df['id_cases'].astype(float).astype(int).astype(str)
            ids_to_exclude_str = {str(int(float(id_val))) for id_val in ids_to_exclude}
            
            cleaned_df = original_df[~original_df['id_cases_str_temp'].isin(ids_to_exclude_str)].copy()
            anomalous_df = original_df[original_df['id_cases_str_temp'].isin(ids_to_exclude_str)].copy()
            
            cleaned_df.drop(columns=['id_cases_str_temp'], inplace=True, errors='ignore')
            anomalous_df.drop(columns=['id_cases_str_temp'], inplace=True, errors='ignore')

        except Exception as e:
            print(f"Warning: Could not robustly filter by id_cases due to type issues: {e}. Falling back to index-based filtering if id_cases is not reliable.")
            # Fallback or more robust type handling might be needed if id_cases are not clean integers
            # For now, we'll assume id_cases can be matched or this will raise an error to be fixed.
            # A simple approach if id_cases is problematic is to work with DataFrame indices,
            # but that requires case_anomaly_data to be keyed by index.
            # Let's proceed assuming id_cases is mostly usable as a key.
            cleaned_df = original_df[~original_df['id_cases'].isin(ids_to_exclude)]
            anomalous_df = original_df[original_df['id_cases'].isin(ids_to_exclude)]

    else: # Should not happen if initial column check passes
        print("Error: 'id_cases' column not found for filtering.")
        return

    base_name = os.path.splitext(input_file_name)[0]
    cleaned_file_name = f"{base_name}_cleaned.csv"
    anomalous_file_name = f"{base_name}_anomalous_rule_based.csv"

    try:
        cleaned_df.to_csv(cleaned_file_name, index=False)
        print(f"\nSuccessfully saved cleaned data to: {cleaned_file_name} ({len(cleaned_df)} rows)")
    except Exception as e:
        print(f"\nError saving cleaned data CSV: {e}")

    try:
        anomalous_df.to_csv(anomalous_file_name, index=False)
        print(f"Successfully saved rule-based anomalous data to: {anomalous_file_name} ({len(anomalous_df)} rows)")
    except Exception as e:
        print(f"Error saving anomalous data CSV: {e}")


if __name__ == "__main__":
    input_csv_path = 'doctor31_cazuri.csv' 
    
    print(f"--- Detecting anomalies in: {input_csv_path} ---")
    anomaly_messages, anomaly_scores, case_data_for_filtering = detect_anomalies_from_file(input_csv_path)
    
    if anomaly_messages:
        # Check if the first message is a critical error message
        is_critical_error = any(msg.startswith("CRITICAL ERROR:") for msg in anomaly_messages)
        
        if not is_critical_error:
            print(f"\n--- Anomaly Report ({len(anomaly_messages)} lines generated): ---")
            for msg in anomaly_messages:
                print(msg)
            
            # Save filtered datasets only if no critical file/column errors occurred
            # Re-load original df here for the standalone script context
            # In an integrated app, you might pass the df from GUI to detection, then to saving.
            try:
                original_df_for_saving = pd.read_csv(input_csv_path)
                save_filtered_datasets(original_df_for_saving, case_data_for_filtering, ANOMALY_SCORE_THRESHOLD, os.path.basename(input_csv_path))
            except Exception as e:
                print(f"\nError loading original DataFrame for saving filtered datasets: {e}")

        else: # Is a critical error
             print("\n--- Critical Errors Encountered: ---")
             for msg in anomaly_messages:
                print(msg)
             print("\nFiltered CSV files will not be generated due to critical errors.")


    if anomaly_scores: # Only plot if scores were generated (i.e., no critical file errors and some anomalies found)
        print(f"\n--- Plotting Anomaly Score Distribution ({len(anomaly_scores)} cases with anomalies) ---")
        try:
            plt.figure(figsize=(12, 7))
            n_bins = 'auto' 
            
            sns.histplot(anomaly_scores, bins=n_bins, kde=True, color='skyblue')
            plt.title('Distribution of Total Anomaly Scores for Cases with Anomalies', fontsize=16)
            plt.xlabel('Total Anomaly Score', fontsize=14)
            plt.ylabel('Number of Cases', fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            
            if anomaly_scores: # Ensure list is not empty for mean/median calculation
                mean_score = sum(anomaly_scores) / len(anomaly_scores)
                # For median, list must be sorted and non-empty
                sorted_scores = sorted(anomaly_scores)
                median_score = sorted_scores[len(sorted_scores) // 2]
                
                plt.axvline(mean_score, color='r', linestyle='dashed', linewidth=1)
                plt.axvline(median_score, color='g', linestyle='dashed', linewidth=1)
                min_ylim, max_ylim = plt.ylim()
                plt.text(mean_score*1.05, max_ylim*0.9, f'Mean: {mean_score:.2f}', color='r')
                plt.text(median_score*1.05, max_ylim*0.8, f'Median: {median_score:.2f}', color='g')
            
            plt.tight_layout()
            plt.savefig("anomaly_plot.png")
        except Exception as e:
            print(f"\nError generating plot: {e}")
            print("Please ensure matplotlib and seaborn are installed ('pip install matplotlib seaborn').")
    else:
        print("No anomaly scores to plot as no anomalies were detected (or only critical errors occurred).")

