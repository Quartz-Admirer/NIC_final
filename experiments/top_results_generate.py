import pandas as pd
import os
import glob  # Для поиска файлов по шаблону

# --- Настройки ---
# Текущая директория, где запускается скрипт
BASE_RESULTS_DIR = "."

# Лимиты, которые нужно обработать (можно изменить, если обрабатывали не все)
LIMITS_TO_PROCESS = [1000, 5000, 10000, 20000]

TOP_N = 10  # Сколько лучших результатов выводить
OUTPUT_FILENAME = "top_results_summary.txt"  # Имя файла для сохранения общего отчета
# Установите None, чтобы только печатать в консоль

# Колонки для отображения в топе (можно настроить)
COLUMNS_TO_SHOW = ['num_boids', 'dimension', 'max_speed', 'perception_radius', 'rmse', 'mae', 'best_val_loss', 'duration_sec', 'error']

# --- Основная логика ---

def find_limit_from_filename(filename):
    """Извлекает число лимита из имени файла."""
    try:
        # Ищем последнее число в имени файла
        parts = os.path.splitext(filename)[0].split('_')
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
    except:
        pass
    return None

def format_and_print_top(df_top, metric_name, limit_value, top_n):
    """Форматирует и печатает топ N результатов."""
    output_lines = []
    title = f"\n--- Top {top_n} Results for Limit {limit_value} (Sorted by {metric_name.upper()}) ---"
    print(title)
    output_lines.append(title)

    if df_top.empty:
        message = f"No valid {metric_name.upper()} results found for limit {limit_value}."
        print(message)
        output_lines.append(message)
    else:
        # Убедимся, что колонки существуют в DataFrame
        cols_present = [col for col in COLUMNS_TO_SHOW if col in df_top.columns]
        # Используем to_string для красивого вывода таблицы
        table_str = df_top[cols_present].to_string(index=False, na_rep='NaN')  # na_rep для отображения NaN
        print(table_str)
        output_lines.append(table_str)
    return "\n".join(output_lines)

# --- Старт парсинга ---
print(f"Starting parsing of Top {TOP_N} results...")
print(f"Looking for summary files in: {os.path.abspath(BASE_RESULTS_DIR)}")

all_results_output = []  # Собираем весь текстовый вывод для файла

# Ищем файлы summary по шаблону
summary_files = glob.glob(os.path.join(BASE_RESULTS_DIR, "experiment_summary_limit_*.csv"))

# Сортируем найденные файлы по лимиту для упорядоченного вывода
sorted_files = sorted(summary_files, key=lambda x: find_limit_from_filename(os.path.basename(x)) or float('inf'))

processed_limits = set()

for file_path in sorted_files:
    limit = find_limit_from_filename(os.path.basename(file_path))
    if limit is None or limit not in LIMITS_TO_PROCESS:
        # print(f"Skipping file (limit not recognized or not in list): {file_path}")
        continue  # Пропускаем, если не смогли определить лимит или его нет в списке

    if limit in processed_limits:
        continue  # Пропускаем, если уже обработали этот лимит (на случай дублей)

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

        # --- Топ N по RMSE ---
        # Проверяем наличие колонки перед сортировкой
        if 'rmse' in df.columns:
            df_valid_rmse = df.dropna(subset=['rmse'])
            df_top_rmse = df_valid_rmse.sort_values(by="rmse", ascending=True).head(TOP_N)
            rmse_output_str = format_and_print_top(df_top_rmse, "rmse", limit, TOP_N)
            all_results_output.append(rmse_output_str)
        else:
            no_col_msg = f"WARNING: Column 'rmse' not found in {file_path}"
            print(no_col_msg)
            all_results_output.append(no_col_msg)

        # --- Топ N по MAE ---
        # Проверяем наличие колонки перед сортировкой
        if 'mae' in df.columns:
            df_valid_mae = df.dropna(subset=['mae'])
            df_top_mae = df_valid_mae.sort_values(by="mae", ascending=True).head(TOP_N)
            mae_output_str = format_and_print_top(df_top_mae, "mae", limit, TOP_N)
            all_results_output.append(mae_output_str)
        else:
            no_col_msg = f"WARNING: Column 'mae' not found in {file_path}"
            print(no_col_msg)
            all_results_output.append(no_col_msg)

        # --- Топ N по BEST_VAL_LOSS ---
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
        processed_limits.add(limit)  # Отмечаем как обработанный, чтобы не пытаться снова

    # --- Сохранение в файл (если указано имя файла) ---
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
        continue # Пропускаем, если уже обработали этот лимит (на случай дублей)

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

        # --- Топ N по RMSE ---
        # Проверяем наличие колонки перед сортировкой
        if 'rmse' in df.columns:
            df_valid_rmse = df.dropna(subset=['rmse'])
            df_top_rmse = df_valid_rmse.sort_values(by="rmse", ascending=True).head(TOP_N)
            rmse_output_str = format_and_print_top(df_top_rmse, "rmse", limit, TOP_N)
            all_results_output.append(rmse_output_str)
        else:
            no_col_msg = f"WARNING: Column 'rmse' not found in {summary_file_path}"
            print(no_col_msg)
            all_results_output.append(no_col_msg)


        # --- Топ N по MAE ---
        # Проверяем наличие колонки перед сортировкой
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
        processed_limits.add(limit) # Отмечаем как обработанный, чтобы не пытаться снова

    # --- Сохранение в файл (если указано имя файла) ---
    if OUTPUT_FILENAME:
        output_file_path = os.path.join(BASE_RESULTS_DIR, OUTPUT_FILENAME)
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(all_results_output))
            print(f"\nSummary of top results saved to: {output_file_path}")
        except Exception as e:
            print(f"\nERROR: Could not write summary file '{output_file_path}'. Error: {e}")

    print("\nParsing finished.")