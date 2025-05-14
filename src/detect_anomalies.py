import pandas as pd
from datetime import datetime, timedelta

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

def parse_timestamp(timestamp_str):
    """
    Safely parses a timestamp string.
    Returns a datetime object or None if parsing fails.
    """
    if pd.isna(timestamp_str):
        return None
    try:
        return datetime.strptime(str(timestamp_str), '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            return datetime.strptime(str(timestamp_str), '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            return None

def format_anomaly(case_id, rule_description, actual_value, details=""):
    """Helper to format anomaly messages consistently."""
    return f"Case ID {case_id}: {rule_description} (Value: {actual_value}). {details}".strip()

def format_anomaly_structured(case_id, column_affected, rule_description, actual_value, details=""):
    """Helper to return structured anomaly information."""
    return {
        "case_id": case_id,
        "column": column_affected,
        "rule": rule_description,
        "value": actual_value,
        "details": details
    }

def detect_anomalies_from_file(file_path):
    """
    Loads data from a CSV file and detects anomalies based on predefined rules.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of strings describing detected anomalies.
    """
    anomalies_found_structured = []

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        anomalies_found_structured.append(format_anomaly_structured("N/A", "File", "File Not Found", file_path))
        return [res["details"] for res in anomalies_found_structured]
    except Exception as e:
        anomalies_found_structured.append(format_anomaly_structured("N/A", "File", "File Read Error", str(e)))
        return [res["details"] for res in anomalies_found_structured]

    required_cols = ['id_cases', 'age_v', 'greutate', 'inaltime', 'IMC', 'data1', 'imcINdex']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        anomalies_found_structured.append(format_anomaly_structured("N/A", "File Structure", "Missing Columns", ", ".join(missing_cols)))
        return [f"Error: Missing required columns in CSV: {', '.join(missing_cols)}"]

    for index, row in df.iterrows():
        case_id = row.get('id_cases', f"RowIndex_{index}")

        try:
            age = pd.to_numeric(row.get('age_v'), errors='coerce')
            weight = pd.to_numeric(row.get('greutate'), errors='coerce')
            height_cm = pd.to_numeric(row.get('inaltime'), errors='coerce')
            bmi_original = pd.to_numeric(row.get('IMC'), errors='coerce')
            bmi_category = str(row.get('imcINdex', '')).strip().lower()
            timestamp_str = row.get('data1')
        except Exception as e:
            anomalies_found_structured.append(format_anomaly_structured(case_id, "Row", "Data Parsing Error", str(e)))
            continue

        if pd.isna(age):
            anomalies_found_structured.append(format_anomaly_structured(case_id, "age_v", "Missing Age", "NaN"))
        elif age > MAX_AGE or age < 0:
            anomalies_found_structured.append(format_anomaly_structured(case_id, "age_v", "Impossible Age", age, f"Outside 0-{MAX_AGE} range"))

        if pd.isna(weight):
            anomalies_found_structured.append(format_anomaly_structured(case_id, "greutate", "Missing Weight", "NaN"))
        elif not pd.isna(age):
            if age > CHILD_AGE_THRESHOLD and weight < MIN_ADULT_WEIGHT_KG:
                anomalies_found_structured.append(format_anomaly_structured(case_id, "greutate", "Implausible Adult Weight", weight, f"< {MIN_ADULT_WEIGHT_KG}kg for age {age}"))
            elif age <= CHILD_AGE_THRESHOLD and weight < MIN_CHILD_WEIGHT_KG:
                 anomalies_found_structured.append(format_anomaly_structured(case_id, "greutate", "Implausible Child Weight", weight, f"< {MIN_CHILD_WEIGHT_KG}kg for age {age}"))
            elif weight <= 0:
                 anomalies_found_structured.append(format_anomaly_structured(case_id, "greutate", "Invalid Weight", weight, "Weight <= 0 kg"))

        if pd.isna(height_cm):
            anomalies_found_structured.append(format_anomaly_structured(case_id, "inaltime", "Missing Height", "NaN"))
        elif height_cm < MIN_HEIGHT_CM or height_cm > MAX_HEIGHT_CM:
            anomalies_found_structured.append(format_anomaly_structured(case_id, "inaltime", "Unlikely Height", height_cm, f"Outside {MIN_HEIGHT_CM}-{MAX_HEIGHT_CM}cm range"))

        current_bmi_to_check = None
        bmi_source = ""

        if not pd.isna(bmi_original):
            current_bmi_to_check = bmi_original
            bmi_source = "original"
        elif pd.notna(weight) and pd.notna(height_cm) and height_cm > 0:
            height_m = height_cm / 100.0
            try:
                current_bmi_to_check = weight / (height_m ** 2)
                bmi_source = "calculated"
            except ZeroDivisionError:
                 anomalies_found_structured.append(format_anomaly_structured(case_id, "IMC", "BMI Calculation Error", "Height is 0", "Cannot calculate BMI if height is 0"))
            except Exception:
                 anomalies_found_structured.append(format_anomaly_structured(case_id, "IMC", "BMI Calculation Error", "Unknown error", "Error during BMI recalculation"))
        else:
            if pd.isna(bmi_original):
                anomalies_found_structured.append(format_anomaly_structured(case_id, "IMC", "Missing BMI", "NaN", "And cannot be calculated due to missing/invalid height/weight."))

        if pd.notna(current_bmi_to_check):
            if current_bmi_to_check < MIN_BMI or current_bmi_to_check > MAX_BMI:
                anomalies_found_structured.append(format_anomaly_structured(
                    case_id, "IMC", f"BMI Incompatible with Life ({bmi_source})", f"{current_bmi_to_check:.1f}",
                    f"Outside {MIN_BMI}-{MAX_BMI} range"
                ))

        if pd.notna(age) and age > ELDERLY_AGE_THRESHOLD:
            if pd.isna(bmi_category) or bmi_category == "":
                anomalies_found_structured.append(format_anomaly_structured(case_id, "imcINdex", "Missing BMI Category for Elderly", "NaN", f"Age: {age}"))
            elif "obese" in bmi_category:
                anomalies_found_structured.append(format_anomaly_structured(case_id, "imcINdex", "Suspicious Elderly Obesity", bmi_category, f"Age: {age}"))

    df['parsed_data1'] = df['data1'].apply(parse_timestamp)
    df_valid_for_duplicates = df.dropna(subset=['age_v', 'greutate', 'inaltime', 'parsed_data1']).copy()
    
    df_valid_for_duplicates['duplicate_key'] = df_valid_for_duplicates.apply(
        lambda x: (int(x['age_v']), float(x['greutate']), float(x['inaltime']))
        if pd.notna(x['age_v']) and pd.notna(x['greutate']) and pd.notna(x['inaltime'])
        else None,
        axis=1
    )
    df_valid_for_duplicates.dropna(subset=['duplicate_key'], inplace=True)

    if not df_valid_for_duplicates.empty:
        grouped = df_valid_for_duplicates.sort_values('parsed_data1').groupby('duplicate_key')
        processed_duplicate_pairs = set()

        for key_tuple, group in grouped:
            if len(group) > 1:
                ids_in_group = group['id_cases'].tolist()
                timestamps_in_group = group['parsed_data1'].tolist()
                
                for i in range(len(ids_in_group)):
                    for j in range(i + 1, len(ids_in_group)):
                        id1, time1 = ids_in_group[i], timestamps_in_group[i]
                        id2, time2 = ids_in_group[j], timestamps_in_group[j]
                        
                        pair1 = tuple(sorted((id1, id2)))
                        if pair1 in processed_duplicate_pairs:
                            continue

                        time_diff = abs(time1 - time2)
                        if time_diff < timedelta(hours=DUPLICATE_TIMEFRAME_HOURS):
                            details_str = (f"Cases {id1} ({time1.strftime('%Y-%m-%d %H:%M')}) & "
                                           f"{id2} ({time2.strftime('%Y-%m-%d %H:%M')}) "
                                           f"have same Age/Weight/Height and are within {DUPLICATE_TIMEFRAME_HOURS}hr(s).")
                            anomalies_found_structured.append(format_anomaly_structured(
                                f"{id1}, {id2}", "Multiple", "Potential Duplicate Record", 
                                f"A/W/H: {key_tuple[0]}/{key_tuple[1]}/{key_tuple[2]}",
                                details_str
                            ))
                            processed_duplicate_pairs.add(pair1)

    final_anomaly_messages = []
    if not anomalies_found_structured:
        final_anomaly_messages.append("No anomalies detected based on the current rules.")
    else:
        for anom in anomalies_found_structured:
            final_anomaly_messages.append(
                f"Case ID {anom['case_id']}: {anom['rule']} affecting '{anom['column']}' (Value: {anom['value']}). Details: {anom['details']}"
            )
            
    return final_anomaly_messages

if __name__ == "__main__":
    test_file_path = 'doctor31_cazuri.csv' 
    
    print(f"--- Detecting anomalies in: {test_file_path} ---")
    detected_anomalies_list = detect_anomalies_from_file(test_file_path)
    
    if detected_anomalies_list:
        print(f"\n--- {len(detected_anomalies_list)} Anomaly Messages Generated: ---")
        for i, anomaly_msg in enumerate(detected_anomalies_list):
            print(f"{i+1}. {anomaly_msg}")
    else:
        print("No anomaly messages were generated (this indicates an issue in the script itself if anomalies were expected).")
