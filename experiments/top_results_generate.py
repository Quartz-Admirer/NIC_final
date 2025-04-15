import pandas as pd
import os
import glob 

BASE_RESULTS_DIR = "."
LIMITS_TO_PROCESS = [1000, 5000, 10000, 20000]

TOP_N = 10 
OUTPUT_FILENAME = "top_results_summary.txt" 

COLUMNS_TO_SHOW = ['num_boids', 'dimension', 'max_speed', 'perception_radius', 'rmse', 'mae', 'best_val_loss', 'duration_sec', 'error']

def find_limit_from_filename(filename):
    """Извлекает число лимита из имени файла."""
    try:
        parts = os.path.splitext(filename)[0].split('_')
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
    except:
        pass
    return None

def format_and_print_top(df_top, metric_name, limit_value, top_n):
    output_lines = []
    title = f"\n--- Top {top_n} Results for Limit {limit_value} (Sorted by {metric_name.upper()}) ---"
    print(title)
    output_lines.append(title)

    if df_top.empty:
        message = f"No valid {metric_name.upper()} results found for limit {limit_value}."
        print(message)
        output_lines.append(message)
    else:
        cols_present = [col for col in COLUMNS_TO_SHOW if col in df_top.columns]
        table_str = df_top[cols_present].to_string(index=False, na_rep='NaN')
        print(table_str)
        output_lines.append(table_str)
    return "\n".join(output_lines)

print(f"Starting parsing of Top {TOP_N} results...")
print(f"Looking for summary files in: {os.path.abspath(BASE_RESULTS_DIR)}")

all_results_output = []

summary_files = glob.glob(os.path.join(BASE_RESULTS_DIR, "experiment_summary_limit_*.csv"))

sorted_files = sorted(summary_files, key=lambda x: find_limit_from_filename(os.path.basename(x)) or float('inf'))

processed_limits = set()

for file_path in sorted_files:
    limit = find_limit_from_filename(os.path.basename(file_path))
    if limit is None or limit not in LIMITS_TO_PROCESS:
        continue 

    if limit in processed_limits:
        continue

    print(f"\n{'=' * 15} Processing Limit: {limit} {'=' * 15}")
    all_results_output.append(f"\n\n{'=' * 15} Processing Limit: {limit} {'=' * 15}")

    try:
        df = pd.read_csv(file_path)

        if df.empty:
            warning_msg = f"WARNING: Summary file is empty: {file_path}"
            print(warning_msg)
            all_results_output.append(warning_msg)
            processed_limits.add(limit)
            continue

        if 'rmse' in df.columns:
            df_valid_rmse = df.dropna(subset=['rmse'])
            df_top_rmse = df_valid_rmse.sort_values(by="rmse", ascending=True).head(TOP_N)
            rmse_output_str = format_and_print_top(df_top_rmse, "rmse", limit, TOP_N)
            all_results_output.append(rmse_output_str)
        else:
            no_col_msg = f"WARNING: Column 'rmse' not found in {file_path}"
            print(no_col_msg)
            all_results_output.append(no_col_msg)

        if 'mae' in df.columns:
            df_valid_mae = df.dropna(subset=['mae'])
            df_top_mae = df_valid_mae.sort_values(by="mae", ascending=True).head(TOP_N)
            mae_output_str = format_and_print_top(df_top_mae, "mae", limit, TOP_N)
            all_results_output.append(mae_output_str)
        else:
            no_col_msg = f"WARNING: Column 'mae' not found in {file_path}"
            print(no_col_msg)
            all_results_output.append(no_col_msg)

        if 'best_val_loss' in df.columns:
            df_valid_best_val_loss = df.dropna(subset=['best_val_loss'])
            df_top_best_val_loss = df_valid_best_val_loss.sort_values(by="best_val_loss", ascending=True).head(TOP_N)
            best_val_loss_output_str = format_and_print_top(df_top_best_val_loss, "best_val_loss", limit, TOP_N)
            all_results_output.append(best_val_loss_output_str)
        else:
            no_col_msg = f"WARNING: Column 'best_val_loss' not found in {file_path}"
            print(no_col_msg)
            all_results_output.append(no_col_msg)
            
        processed_limits.add(limit)

    except Exception as e:
        error_msg = f"ERROR: Could not process file {file_path}. Error: {e}"
        print(error_msg)
        all_results_output.append(error_msg)
        processed_limits.add(limit)

    if OUTPUT_FILENAME:
        output_file_path = os.path.join(BASE_RESULTS_DIR, OUTPUT_FILENAME)
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(all_results_output))
            print(f"\nSummary of top results saved to: {output_file_path}")
        except Exception as e:
            print(f"\nERROR: Could not write summary file '{output_file_path}'. Error: {e}")

    print("\nParsing finished.")
    if limit in processed_limits:
        continue

    print(f"\n{'='*15} Processing Limit: {limit} {'='*15}")
    all_results_output.append(f"\n\n{'='*15} Processing Limit: {limit} {'='*15}")

    summary_file_path = os.path.join(folder_path, f"experiment_summary_limit_{limit}.csv")

    if not os.path.exists(summary_file_path):
        warning_msg = f"WARNING: Summary file not found: {summary_file_path}"
        print(warning_msg)
        all_results_output.append(warning_msg)
        processed_limits.add(limit)
        continue

    try:
        df = pd.read_csv(summary_file_path)

        if df.empty:
            warning_msg = f"WARNING: Summary file is empty: {summary_file_path}"
            print(warning_msg)
            all_results_output.append(warning_msg)
            processed_limits.add(limit)
            continue
            
        if 'rmse' in df.columns:
            df_valid_rmse = df.dropna(subset=['rmse'])
            df_top_rmse = df_valid_rmse.sort_values(by="rmse", ascending=True).head(TOP_N)
            rmse_output_str = format_and_print_top(df_top_rmse, "rmse", limit, TOP_N)
            all_results_output.append(rmse_output_str)
        else:
            no_col_msg = f"WARNING: Column 'rmse' not found in {summary_file_path}"
            print(no_col_msg)
            all_results_output.append(no_col_msg)


        if 'mae' in df.columns:
            df_valid_mae = df.dropna(subset=['mae'])
            df_top_mae = df_valid_mae.sort_values(by="mae", ascending=True).head(TOP_N)
            mae_output_str = format_and_print_top(df_top_mae, "mae", limit, TOP_N)
            all_results_output.append(mae_output_str)
        else:
            no_col_msg = f"WARNING: Column 'mae' not found in {summary_file_path}"
            print(no_col_msg)
            all_results_output.append(no_col_msg)
            
        if 'best_val_loss' in df.columns:
            df_valid_mae = df.dropna(subset=['best_val_loss'])
            df_top_mae = df_valid_mae.sort_values(by="best_val_loss", ascending=True).head(TOP_N)
            mae_output_str = format_and_print_top(df_top_mae, "best_val_loss", limit, TOP_N)
            all_results_output.append(mae_output_str)
        else:
            no_col_msg = f"WARNING: Column 'best_val_loss' not found in {summary_file_path}"
            print(no_col_msg)
            all_results_output.append(no_col_msg)

        processed_limits.add(limit)

    except Exception as e:
        error_msg = f"ERROR: Could not process file {summary_file_path}. Error: {e}"
        print(error_msg)
        all_results_output.append(error_msg)
        processed_limits.add(limit)

    if OUTPUT_FILENAME:
        output_file_path = os.path.join(BASE_RESULTS_DIR, OUTPUT_FILENAME)
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(all_results_output))
            print(f"\nSummary of top results saved to: {output_file_path}")
        except Exception as e:
            print(f"\nERROR: Could not write summary file '{output_file_path}'. Error: {e}")

    print("\nParsing finished.")
