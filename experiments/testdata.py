import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
import time as timer
import sys

try:
    import load_data
    import boids
except ImportError:
    print("Error: Ensure load_data.py and boids.py are in the same directory or Python path.")
    sys.exit(1)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(df, feature_cols, target_col, seq_length=10):
    data = df[feature_cols].values
    target = df[target_col].values
    X, y = [], []
    if len(df) <= seq_length:
        return np.array(X), np.array(y)
    for i in range(len(df) - seq_length):
        seq_x = data[i : i + seq_length]
        seq_y = target[i + seq_length]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def train_test_split(X, y, train_size=0.8):
    split_idx = int(len(X) * train_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, epochs, lr=1e-3, checkpoint_path="best_model.pth", output_dir=".", plot_losses=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1).to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().unsqueeze(1).to(device)

    best_val_loss = float("inf")
    train_losses_log=[]
    val_losses_log=[]
    print(f"   Training Progress (Epochs: {epochs}, LR: {lr}, Device: {device}):")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        val_loss_item = float("inf")
        if len(X_test_t) > 0:
            with torch.no_grad():
                val_outputs = model(X_test_t)
                val_loss = criterion(val_outputs, y_test_t)
                val_loss_item = val_loss.item()

        is_best = val_loss_item < best_val_loss
        if is_best:
            best_val_loss = val_loss_item
            torch.save(model.state_dict(), checkpoint_path)

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1 or is_best:
            print(f"   [Epoch {epoch+1:>{len(str(epochs))}}/{epochs}] Train Loss: {loss.item():.6f} | Val Loss: {val_loss_item:.6f} {'(New Best)' if is_best else ''}")

        train_losses_log.append(loss.item())
        val_losses_log.append(val_loss_item)

    if plot_losses and train_losses_log and val_losses_log and best_val_loss != float("inf"):
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses_log, label='Train Loss')
            valid_val_losses = [l for l in val_losses_log if l != float("inf")]
            if valid_val_losses:
                plt.plot(val_losses_log, label='Validation Loss')

            plot_base_filename = os.path.basename(checkpoint_path).replace(".pth", "_losses.png")
            plot_save_path = os.path.join(output_dir, plot_base_filename)

            plt.title(f'Loss vs Val Loss ({os.path.basename(checkpoint_path)})')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.grid(True)
            max_loss_val = max(max(train_losses_log), max(valid_val_losses)) if valid_val_losses else max(train_losses_log)
            plt.ylim(bottom=0, top=max(max_loss_val * 1.2, 1e-5))
            plt.savefig(plot_save_path)
            print(f"   Loss plot saved to {plot_save_path}")
            plt.close()
        except Exception as e:
            print(f"   Warning: Could not generate loss plot. Error: {e}")

    return best_val_loss

def evaluate_model(model, X_test, y_test, target_min, target_max, output_dir=".", plot_base_filename="results", plot_results=True):
    if len(X_test) == 0:
        print("   Evaluation skipped: Test set is empty.")
        return {"mae": None, "rmse": None}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    X_test_t = torch.from_numpy(X_test).float().to(device)

    with torch.no_grad():
        y_pred_t = model(X_test_t)
    y_pred = y_pred_t.cpu().numpy().squeeze()

    if y_pred.ndim == 0: y_pred = np.array([y_pred])
    if y_test.ndim > 1: y_test = y_test.squeeze()
    if y_pred.shape != y_test.shape:
        print(f"   Warning: Shape mismatch in evaluation. y_pred: {y_pred.shape}, y_test: {y_test.shape}.")
        if y_pred.ndim == 0 and y_test.ndim == 1 and len(y_test) == 1: y_pred = y_pred.reshape(1)
        elif y_pred.ndim == 1 and y_test.ndim == 0 and len(y_pred) == 1: y_test = y_test.reshape(1)

        if y_pred.shape != y_test.shape:
            print("   Error: Could not resolve shape mismatch. Skipping evaluation plotting/metrics.")
            return {"mae": None, "rmse": None}

    epsilon = 1e-9
    y_pred_inv = y_pred * (target_max - target_min + epsilon) + target_min
    y_true_inv = y_test * (target_max - target_min + epsilon) + target_min

    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    mse = mean_squared_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mse)

    print(f"   Evaluation Results:")
    print(f"     MAE:  {mae:.4f}")
    print(f"     RMSE: {rmse:.4f}")

    if plot_results:
        try:
            plt.figure(figsize=(12, 7))
            plt.plot(y_true_inv, label="Real", marker='.', linestyle='-', alpha=0.7)
            plt.plot(y_pred_inv, label="Predicted", marker='.', linestyle='--', alpha=0.7)
            plt.legend()
            plt.title(f"Real vs Predicted (Test Set) - {plot_base_filename}")
            plt.xlabel("Sample index")
            plt.ylabel("Price")
            plt.grid(True)

            safe_filename_base = "".join(c if c.isalnum() or c in ['_','-'] else "_" for c in plot_base_filename)
            plot_save_path = os.path.join(output_dir, f"{safe_filename_base}.png")

            plt.savefig(plot_save_path)
            print(f"   Results plot saved to {plot_save_path}")
            plt.close()
        except Exception as e:
            print(f"   Warning: Could not generate results plot. Error: {e}")

    return {"mae": mae, "rmse": rmse}

def load_specific_data(data_params, feature_cols, target_col, base_path="."):
    param_str = f"{data_params['limit']}_{data_params['num_boids']}_{data_params['dimension']}_{data_params['max_speed']}_{data_params['perception_radius']}"
    processed_path = os.path.join(base_path, f"{param_str}.csv")
    scaling_path = os.path.join(base_path, f"{param_str}.json")

    if not os.path.exists(processed_path):
        print(f"ERROR: Required data file not found: {processed_path}")
        return None, None, None, None
    if not os.path.exists(scaling_path):
        print(f"ERROR: Required scaling info file not found: {scaling_path}")
        return None, None, None, None

    try:
        df = pd.read_csv(processed_path)
    except Exception as e:
        print(f"Error loading CSV file {processed_path}: {e}")
        return None, None, None, None

    try:
        with open(scaling_path, "r") as f:
            scaling = json.load(f)
            target_min = scaling["min"]
            target_max = scaling["max"]
    except Exception as e:
        print(f"Error loading JSON file {scaling_path}: {e}")
        return None, None, None, None

    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns in {processed_path}: {missing_cols}")
        return None, None, None, None

    return df, None, None, target_min, target_max

if __name__ == "__main__":

    param_ranges = {
        'num_boids': [50, 100, 200, 400],
        'dimension': [100, 200, 400, 800],
        'max_speed': [5, 10, 20],
        'perception_radius': [50, 100, 150, 200]
    }
    keys, values = zip(*param_ranges.items())
    base_parameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    num_base_combinations = len(base_parameter_combinations)

    limits_to_run = [1000, 5000, 10000, 20000]

    print(f"--- Starting Experiment Run ---")
    print(f"Limits to process: {limits_to_run}")
    print(f"Base parameter combinations per limit: {num_base_combinations}")
    print(f"Parameter ranges: {param_ranges}")

    feature_cols = ["close", "ma_close",
                    "boids_mean_x", "boids_mean_y", "boids_mean_vx", "boids_mean_vy",
                    "boids_std_x", "boids_std_y", "boids_std_vx", "boids_std_vy"]
    target_col = "future_close"
    seq_length = 20
    hidden_dim = 64
    num_layers = 2
    epochs = 500
    learning_rate = 1e-3
    train_ratio = 0.8
    base_data_path = "datasets"

    print("\n--- Experiment Settings ---")
    print(f"Feature Cols: {feature_cols}")
    print(f"Target Col: {target_col}")
    print(f"Sequence Length: {seq_length}")
    print(f"LSTM Hidden Dim: {hidden_dim}, Layers: {num_layers}")
    print(f"Epochs per run: {epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Train Ratio: {train_ratio}")
    print(f"Base Data Path: {os.path.abspath(base_data_path)}")


    start_time_total_experiment = timer.time()

    for limit_value in limits_to_run:
        print(f"\n{'='*20} Processing Limit: {limit_value} {'='*20}")

        current_results_dir = f"experiment_results_limit_{limit_value}"
        os.makedirs(current_results_dir, exist_ok=True)
        print(f"Results directory for this limit: {os.path.abspath(current_results_dir)}")

        results_for_current_limit = []
        start_time_limit = timer.time()

        for i, base_params in enumerate(base_parameter_combinations):
            current_run_params = base_params.copy()
            current_run_params['limit'] = limit_value

            param_str = f"{current_run_params['limit']}_{current_run_params['num_boids']}_{current_run_params['dimension']}_{current_run_params['max_speed']}_{current_run_params['perception_radius']}"
            model_checkpoint_path = os.path.join(current_results_dir, f"model_{param_str}.pth")
            eval_plot_base_filename = f"eval_{param_str}"

            print(f"\n--- Run {i+1}/{num_base_combinations} (Limit {limit_value}): Parameters: {param_str} ---")
            start_time_run = timer.time()

            current_result = current_run_params.copy()
            current_result.update({"mae": None, "rmse": None, "best_val_loss": None, "duration_sec": None, "error": None})

            processed_path_check = os.path.join(base_data_path, f"{param_str}.csv")
            scaling_path_check = os.path.join(base_data_path, f"{param_str}.json")

            if not os.path.exists(processed_path_check) or not os.path.exists(scaling_path_check):
                missing_files = []
                if not os.path.exists(processed_path_check): missing_files.append(os.path.basename(processed_path_check))
                if not os.path.exists(scaling_path_check): missing_files.append(os.path.basename(scaling_path_check))
                error_msg = f"Data file(s) not found: {', '.join(missing_files)}"
                print(f"   SKIPPING: {error_msg} in '{base_data_path}'")
                current_result["error"] = error_msg
                run_duration = timer.time() - start_time_run
                current_result["duration_sec"] = round(run_duration, 2)
                results_for_current_limit.append(current_result)
                continue

            try:
                print("   Loading data...")
                df_boids, _, _, target_min, target_max = load_specific_data(
                    current_run_params, feature_cols, target_col, base_path=base_data_path
                )
                if df_boids is None:
                    raise ValueError("Failed to load data (files might be corrupted or missing columns).")

                print(f"   Creating sequences (length={seq_length})...")
                X, y = create_sequences(df_boids, feature_cols, target_col, seq_length=seq_length)
                if len(X) == 0:
                    raise ValueError(f"Not enough data ({len(df_boids)} rows) for sequences of length {seq_length}.")

                X_train, y_train, X_test, y_test = train_test_split(X, y, train_size=train_ratio)
                print(f"   Data split: Train {X_train.shape}, Test {X_test.shape}")
                if len(X_test) == 0:
                    print("   WARNING: Test set is empty. Cannot evaluate model performance.")

                input_dim = len(feature_cols)
                model = LSTMModel(input_dim, hidden_dim, num_layers)

                print("   Starting training...")
                best_val_loss = train_model(model, X_train, y_train, X_test, y_test,
                                            epochs=epochs, lr=learning_rate,
                                            checkpoint_path=model_checkpoint_path,
                                            output_dir=current_results_dir,
                                            plot_losses=True)
                current_result["best_val_loss"] = best_val_loss if best_val_loss != float("inf") else None

                metrics = {"mae": None, "rmse": None}
                if len(X_test) > 0:
                    if os.path.exists(model_checkpoint_path):
                        print("   Loading best model and evaluating...")
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        print(f"   Evaluation device: {device}")

                        eval_model = LSTMModel(input_dim, hidden_dim, num_layers)
                        state_dict = torch.load(model_checkpoint_path, map_location=device)
                        eval_model.load_state_dict(state_dict)
                        eval_model.to(device)

                        metrics = evaluate_model(eval_model, X_test, y_test, target_min, target_max,
                                                output_dir=current_results_dir,
                                                plot_base_filename=eval_plot_base_filename,
                                                plot_results=True)
                        current_result.update(metrics)
                    else:
                        print("   WARNING: Best model checkpoint not found. Cannot evaluate.")
                        if not current_result.get("error"):
                            current_result["error"] = "Checkpoint not found (no improvement?)"
                else:
                    print("   Skipping evaluation due to empty test set.")
                    if not current_result.get("error"):
                        current_result["error"] = "Empty test set"

            except Exception as e:
                error_msg = f"ERROR during run: {e}"
                print(f"   {error_msg}")
                current_result["error"] = str(e).replace(',', ';').replace('\n', ' ')
            finally:
                run_duration = timer.time() - start_time_run
                current_result["duration_sec"] = round(run_duration, 2)
                results_for_current_limit.append(current_result)
                print(f"   Run duration: {run_duration:.2f} seconds")

                if torch.cuda.is_available():
                    try:
                        if 'model' in locals(): del model
                        if 'eval_model' in locals(): del eval_model
                        if 'X_train_t' in locals(): del X_train_t, y_train_t, X_test_t, y_test_t
                        if 'outputs' in locals(): del outputs
                        if 'val_outputs' in locals(): del val_outputs
                    except NameError:
                        pass
                    torch.cuda.empty_cache()

        limit_duration = timer.time() - start_time_limit
        print(f"\n--- Finished Processing Limit: {limit_value} ---")
        print(f"Duration for limit {limit_value}: {limit_duration/60:.2f} minutes ({limit_duration:.2f} seconds)")

        if not results_for_current_limit:
            print(f"No results were collected for limit {limit_value}.")
        else:
            results_df_limit = pd.DataFrame(results_for_current_limit)

            results_csv_path = os.path.join(current_results_dir, f"experiment_summary_limit_{limit_value}.csv")
            try:
                results_df_limit.to_csv(results_csv_path, index=False)
                print(f"\nResults summary for limit {limit_value} saved to: {results_csv_path}")
            except Exception as e:
                print(f"\nError saving results summary to CSV for limit {limit_value}: {e}")

            results_df_valid_rmse = results_df_limit.dropna(subset=['rmse'])
            if not results_df_valid_rmse.empty:
                results_df_sorted_rmse = results_df_valid_rmse.sort_values(by="rmse", ascending=True)
                print(f"\n--- Top 10 Best Runs (Sorted by RMSE) for Limit {limit_value} ---")
                print(results_df_sorted_rmse[['num_boids', 'dimension', 'max_speed', 'perception_radius', 'rmse', 'mae', 'best_val_loss', 'duration_sec', 'error']].head(10).to_string())
            else:
                print(f"\nNo valid RMSE results to display for limit {limit_value}.")

            results_df_valid_mae = results_df_limit.dropna(subset=['mae'])
            if not results_df_valid_mae.empty:
                results_df_sorted_mae = results_df_valid_mae.sort_values(by="mae", ascending=True)
                print(f"\n--- Top 10 Best Runs (Sorted by MAE) for Limit {limit_value} ---")
                print(results_df_sorted_mae[['num_boids', 'dimension', 'max_speed', 'perception_radius', 'rmse', 'mae', 'best_val_loss', 'duration_sec', 'error']].head(10).to_string())
            else:
                print(f"\nNo valid MAE results to display for limit {limit_value}.")

            error_runs = results_df_limit[results_df_limit['error'].notna()]
            if not error_runs.empty:
                print(f"\n--- Runs with Errors for Limit {limit_value} ---")
                cols_to_show = ['limit', 'num_boids', 'dimension', 'max_speed', 'perception_radius', 'error']
                print(error_runs[cols_to_show].to_string())
            else:
                print(f"\nNo errors reported during the runs for limit {limit_value}.")

    total_duration_experiment = timer.time() - start_time_total_experiment
    print(f"\n{'='*20} Experiment Finished {'='*20}")
    print(f"Total experiment duration: {total_duration_experiment/60:.2f} minutes ({total_duration_experiment:.2f} seconds)")