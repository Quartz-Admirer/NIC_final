# noboids.py (Does NOT use Boids features)
import os
import sys
import json
import time as timer
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler # <<< Added import
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dotenv import load_dotenv

# Attempt to import local modules needed for data generation
try:
    import load_data
    import boids
except ImportError:
    print("Warning: load_data.py or boids.py not found. Data generation will fail if needed.")
    pass

# --- 1. Model Definition (Identical to model.py) ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# --- 2. Data Preparation (Identical to model.py) ---
def create_sequences(df: pd.DataFrame, feature_cols: list[str], target_col: str, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    required_cols = feature_cols + [target_col]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns in DataFrame: {missing}")
    data = df[feature_cols].values
    target = df[target_col].values
    X, y = [], []
    if len(df) <= seq_length:
        print(f"Warning: DataFrame length ({len(df)}) <= seq_length ({seq_length}). No sequences generated.")
        return np.array(X), np.array(y)
    for i in range(len(df) - seq_length):
        seq_x = data[i:i + seq_length]
        seq_y = target[i + seq_length]
        X.append(seq_x)
        y.append(seq_y)
    if not X:
        print("Warning: No sequences generated despite sufficient df length. Check data quality.")
    return np.array(X), np.array(y)

def train_test_split(X: np.ndarray, y: np.ndarray, train_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(X) == 0:
        print("Warning: Input X for train_test_split is empty.")
        return np.array([]), np.array([]), np.array([]), np.array([])
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    if len(X_test) == 0:
        print("Warning: Test set is empty after split.")
    return X_train, y_train, X_test, y_test

# --- 3. Training and Evaluation (Identical to model.py) ---
def train_model(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    epochs: int, lr: float, device: torch.device,
    checkpoint_path: Path, output_dir: Path, plot_losses: bool
) -> tuple[float, int]:
    if len(X_train) == 0 or len(y_train) == 0:
        print("Error: Training data is empty. Cannot train model.")
        return float("inf"), -1
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    try:
        X_train_t = torch.from_numpy(X_train).float().to(device)
        y_train_t = torch.from_numpy(y_train).float().unsqueeze(1).to(device)
        X_test_t = torch.from_numpy(X_test).float().to(device) if len(X_test) > 0 else torch.empty(0, X_train.shape[1], X_train.shape[2], device=device)
        y_test_t = torch.from_numpy(y_test).float().unsqueeze(1).to(device) if len(y_test) > 0 else torch.empty(0, 1, device=device)
    except Exception as e:
        print(f"Error converting data to tensors or moving to device: {e}")
        return float("inf"), -1
    best_val_loss = float("inf")
    best_epoch = -1
    train_losses_log = []
    val_losses_log = []
    print(f"Starting training for {epochs} epochs on {device}...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        try:
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            train_losses_log.append(loss.item())
        except Exception as e:
            print(f"Error during training epoch {epoch+1}: {e}")
            break
        current_val_loss_item = float("inf")
        if len(X_test_t) > 0 and len(y_test_t) > 0:
            model.eval()
            try:
                with torch.no_grad():
                    val_outputs = model(X_test_t)
                    val_loss = criterion(val_outputs, y_test_t)
                    current_val_loss_item = val_loss.item()
                val_losses_log.append(current_val_loss_item)
            except Exception as e:
                 print(f"Warning: Error during validation epoch {epoch+1}: {e}")
                 val_losses_log.append(float("inf"))
        else:
             current_val_loss_item = loss.item()
             val_losses_log.append(current_val_loss_item)
             if epoch == 0: print("Warning: No test data for validation. Using train loss for checkpointing.")
        if current_val_loss_item < best_val_loss:
            best_val_loss = current_val_loss_item
            best_epoch = epoch + 1
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
            except Exception as e:
                print(f"Error saving model checkpoint at epoch {epoch+1}: {e}")
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
             status_log = f"  Epoch {epoch+1}/{epochs}. Train Loss: {loss.item():.6f}."
             if len(X_test_t) > 0:
                 status_log += f" Val Loss: {current_val_loss_item:.6f}. Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})"
             else:
                  status_log += f" Best Train Loss (CP): {best_val_loss:.6f} (Epoch {best_epoch})"
             print(status_log)
    if plot_losses and train_losses_log:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses_log, label='Train Loss')
            if len(X_test_t) > 0 and val_losses_log:
                 valid_val_losses = [l for l in val_losses_log if l != float("inf")]
                 if valid_val_losses:
                    plt.plot(val_losses_log, label='Validation Loss', linestyle='--')
            plt.title(f'Loss History ({checkpoint_path.stem})')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.grid(True)
            plt.ylim(bottom=0)
            plot_save_path = output_dir / f"{checkpoint_path.stem}_losses.png"
            plt.savefig(plot_save_path)
            print(f"Loss plot saved to {plot_save_path}")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate loss plot. Error: {e}")
    print(f"Training finished. Best model from Epoch {best_epoch} with Loss: {best_val_loss:.6f}")
    return best_val_loss, best_epoch


def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray, y_test: np.ndarray,
    target_min: float, target_max: float, device: torch.device,
    output_dir: Path, plot_filename_suffix: str = "", plot_results: bool = True
) -> dict:
    if len(X_test) == 0 or len(y_test) == 0:
        print("Evaluation skipped: Test data is empty.")
        return {"mae": float('nan'), "rmse": float('nan')}
    model.eval()
    model.to(device)
    metrics = {"mae": float('nan'), "rmse": float('nan')}
    try:
        X_test_t = torch.from_numpy(X_test).float().to(device)
        y_test_t = torch.from_numpy(y_test).float().unsqueeze(1).to(device)
        with torch.no_grad():
            y_pred_t = model(X_test_t)
        y_pred = y_pred_t.squeeze().cpu().numpy()
        y_true = y_test_t.squeeze().cpu().numpy()
        if y_pred.shape != y_true.shape:
             if y_pred.ndim == 0 and y_true.ndim == 0:
                 y_pred = np.array([y_pred])
                 y_true = np.array([y_true])
             elif y_pred.shape != y_true.shape:
                 raise ValueError(f"Shape mismatch after squeeze: y_pred {y_pred.shape}, y_true {y_true.shape}")
        epsilon = 1e-9
        y_pred_inv = y_pred * (target_max - target_min + epsilon) + target_min
        y_true_inv = y_true * (target_max - target_min + epsilon) + target_min
        metrics["mae"] = mean_absolute_error(y_true_inv, y_pred_inv)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
        print(f"\n--- Evaluation Metrics ({plot_filename_suffix}) ---")
        print(f"MAE:  {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        if plot_results:
            plt.figure(figsize=(12, 6))
            plt.plot(y_true_inv, label="Real", marker='.', linestyle='-', alpha=0.7)
            plt.plot(y_pred_inv, label=f"Predicted ({plot_filename_suffix})", marker='.', linestyle='--', alpha=0.7)
            plt.legend()
            plt.title(f"Test Set: Real vs Predicted ({plot_filename_suffix})")
            plt.xlabel("Sample index (Test Set)")
            plt.ylabel("Price")
            plt.grid(True)
            plot_save_path = output_dir / f"evaluation_plot_{plot_filename_suffix}.png"
            plt.savefig(plot_save_path)
            print(f"Evaluation plot saved to {plot_save_path}")
            plt.close()
    except Exception as e:
        print(f"Error during evaluation: {e}")
    return metrics

# --- 4. Data Loading Function ---
# Define the full potential feature set used during generation/saving
ALL_FEATURE_COLS = [
    "close", "ma_close",
    "boids_mean_x", "boids_mean_y", "boids_mean_vx", "boids_mean_vy",
    "boids_std_x", "boids_std_y", "boids_std_vx", "boids_std_vy"
]

def load_or_generate_data(
    base_data_path: Path,
    data_params: dict,
    target_col: str,
    ma_window: int
) -> tuple[pd.DataFrame | None, float | None, float | None]:
    try:
        import load_data
        import boids
    except ImportError:
        print("Error: Cannot generate data. Ensure load_data.py and boids.py are accessible.")
        pass

    param_str = f"{data_params['limit']}_{data_params['num_boids']}_{data_params['dimension']}_{data_params['max_speed']}_{data_params['perception_radius']}"
    csv_path = base_data_path / f"{param_str}.csv"
    json_path = base_data_path / f"{param_str}.json"

    if csv_path.is_file() and json_path.is_file():
        print(f"Attempting to load cached data from {csv_path} and {json_path}...")
        try:
            df = pd.read_csv(csv_path)
            with open(json_path, "r") as f:
                scaling = json.load(f)
                target_min = scaling["min"]
                target_max = scaling["max"]
            if target_col not in df.columns:
                 raise ValueError(f"Target column '{target_col}' not found in loaded CSV {csv_path}")
            print("Cached data loaded successfully.")
            return df, target_min, target_max
        except Exception as e:
            print(f"Warning: Failed to load cached data ({e}). Attempting to regenerate...")

    print(f"\n--- Generating data for parameters: {param_str} ---")
    try:
        if 'load_data' not in sys.modules or 'boids' not in sys.modules:
             raise ImportError("load_data or boids module not loaded, cannot generate.")

        print(f"Downloading price data (limit={data_params['limit']}, ma_window={ma_window})...")
        df_raw_price = load_data.load_binance_data(symbol="BTCUSDT", interval="1h", total_limit=data_params['limit'])
        if df_raw_price.empty: raise ValueError("Failed to download price data from Binance.")
        df_price = load_data.prepare_data(df_raw_price.copy(), ma_window=ma_window)
        if df_price.empty: raise ValueError("Price DataFrame is empty after MA calculation and dropna.")

        if target_col not in df_price.columns: raise ValueError(f"Target column '{target_col}' not found after load_data.prepare_data")
        target_min = df_price[target_col].min()
        target_max = df_price[target_col].max()
        if pd.isna(target_min) or pd.isna(target_max): raise ValueError("Could not determine target min/max from price data.")

        print("Generating Boids features...")
        boids_df = boids.generate_boids_features(
            num_days=len(df_price),
            num_boids=data_params['num_boids'], width=data_params['dimension'],
            height=data_params['dimension'], max_speed=data_params['max_speed'],
            perception_radius=data_params['perception_radius']
        )
        if boids_df.empty: raise ValueError("Boids DataFrame is empty after generation.")

        df_price = df_price.reset_index(drop=True)
        boids_df = boids_df.reset_index(drop=True)
        df_combined = pd.concat([df_price, boids_df], axis=1)

        features_for_dropna = [col for col in ALL_FEATURE_COLS if col in df_combined.columns]
        df_combined.dropna(subset=features_for_dropna + [target_col], inplace=True)
        if df_combined.empty: raise ValueError("DataFrame is empty after combining price/boids and final dropna.")

        print("Scaling data...")
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        features_to_scale = [col for col in ALL_FEATURE_COLS if col in df_combined.columns]
        df_combined[features_to_scale] = scaler_X.fit_transform(df_combined[features_to_scale])
        df_combined[[target_col]] = scaler_y.fit_transform(df_combined[[target_col]])

        print(f"Saving generated data to {csv_path} and {json_path}...")
        base_data_path.mkdir(parents=True, exist_ok=True)
        columns_to_save = features_to_scale + [target_col]
        df_to_save = df_combined[columns_to_save]
        df_to_save.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump({"min": target_min, "max": target_max}, f)

        print("Data generation and saving complete.")
        return df_to_save, target_min, target_max

    except Exception as e:
        print(f"Error during data generation for {param_str}: {e}")
        print("Please ensure load_data.py and boids.py are available and functional.")
        if csv_path.is_file(): csv_path.unlink(missing_ok=True)
        if json_path.is_file(): json_path.unlink(missing_ok=True)
        return None, None, None


# --- 5. Main Script Execution ---
def main():
    load_dotenv() # Load environment variables from .env file

    # --- Configuration from .env ---
    try:
        base_data_path = Path(os.getenv("BASE_DATA_PATH", "datasets"))
        data_limit = int(os.getenv("DATA_LIMIT", 10000))
        num_boids = int(os.getenv("NUM_BOIDS", 400))
        dimension = int(os.getenv("DIMENSION", 100))
        max_speed = int(os.getenv("MAX_SPEED", 10))
        perception_radius = int(os.getenv("PERCEPTION_RADIUS", 150))
        target_col = os.getenv("TARGET_COL", "future_close")
        ma_window = int(os.getenv("MA_WINDOW", 50)) # <<< Load MA_WINDOW

        seq_length = int(os.getenv("SEQ_LENGTH", 20))
        train_ratio = float(os.getenv("TRAIN_RATIO", 0.8))
        hidden_dim = int(os.getenv("HIDDEN_DIM", 128))
        num_layers = int(os.getenv("NUM_LAYERS", 1))
        learning_rate = float(os.getenv("LEARNING_RATE", 1e-3))
        epochs = int(os.getenv("EPOCHS", 500))

        device_choice = os.getenv("DEVICE", "auto").lower()
        results_base_dir = Path(os.getenv("RESULTS_BASE_DIR", "experiment_results_final"))
        plot_losses = os.getenv("PLOT_LOSSES", "True").lower() == "true"
        plot_evaluation = os.getenv("PLOT_EVALUATION", "True").lower() == "true"
    except Exception as e:
        print(f"Error reading configuration from .env file: {e}")
        sys.exit(1)

    # --- Feature Definition (WITHOUT Boids) ---
    feature_cols = ["close", "ma_close"] # <<< Key difference for this script
    script_identifier = "no_boids" # For output naming

    # --- Device Setup ---
    if device_choice == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_choice == "cpu":
        device = torch.device("cpu")
    else: # auto
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Output Directory ---
    data_params_dict = { # Use params to identify source data file
        'limit': data_limit, 'num_boids': num_boids, 'dimension': dimension,
        'max_speed': max_speed, 'perception_radius': perception_radius
    }
    param_str = f"{data_limit}_{num_boids}_{dimension}_{max_speed}_{perception_radius}"
    output_dir = results_base_dir / f"{param_str}_{script_identifier}" # Add suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir.resolve()}")

    # --- Load or Generate Data ---
    # Define columns needed JUST for this script to run (must exist in file)
    columns_needed_from_file = feature_cols + [target_col]
    df_full, target_min, target_max = load_or_generate_data(
        base_data_path, data_params_dict, target_col, ma_window
        # Note: load_or_generate_data internally uses ALL_FEATURE_COLS for dropna during generation
        # but here we only check if the columns needed for *this script* are present after loading.
    )
    if df_full is None:
        print("Failed to load or generate data. Exiting.")
        sys.exit(1)

    # --- Prepare Sequences ---
    # Pass the specific feature_cols for *this script*
    print(f"Creating sequences with seq_length={seq_length} using features: {feature_cols}")
    X, y = create_sequences(df_full, feature_cols, target_col, seq_length=seq_length)
    if len(X) == 0:
        print("Error: No sequences created. Exiting.")
        sys.exit(1)

    # --- Split Data ---
    X_train, y_train, X_test, y_test = train_test_split(X, y, train_ratio=train_ratio)
    print(f"Data shapes: Train X={X_train.shape}, Train y={y_train.shape}; Test X={X_test.shape}, Test y={y_test.shape}")

    # --- Initialize Model ---
    input_dim = len(feature_cols) # <<< Calculated based on non-boids features
    model_to_train = LSTMModel(input_dim, hidden_dim, num_layers)
    print(f"Model initialized: input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")

    # --- Train Model ---
    print("\n--- Starting Model Training ---")
    checkpoint_path = output_dir / f"best_model_{script_identifier}.pth" # <<< Different name
    best_loss, best_epoch = train_model(
        model=model_to_train,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        epochs=epochs, lr=learning_rate, device=device,
        checkpoint_path=checkpoint_path, output_dir=output_dir, plot_losses=plot_losses
    )

    # --- Evaluate Best Model ---
    print(f"\n--- Evaluating Best Model (from Epoch {best_epoch}, Loss: {best_loss:.6f}) ---")
    if best_epoch != -1 and checkpoint_path.is_file():
        best_model = LSTMModel(input_dim, hidden_dim, num_layers) # Use correct input_dim
        try:
            best_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            best_model.to(device)

            final_metrics = evaluate_model(
                model=best_model,
                X_test=X_test, y_test=y_test,
                target_min=target_min, target_max=target_max, device=device,
                output_dir=output_dir, plot_filename_suffix=script_identifier, plot_results=plot_evaluation
            )
            print(f"Final Metrics ({script_identifier}): MAE={final_metrics.get('mae', 'N/A'):.4f}, RMSE={final_metrics.get('rmse', 'N/A'):.4f}")

            summary = {
                "script": script_identifier, **data_params_dict,
                "features_used": feature_cols,
                "seq_length": seq_length, "train_ratio": train_ratio,
                "hidden_dim": hidden_dim, "num_layers": num_layers,
                "learning_rate": learning_rate, "epochs_run": epochs,
                "best_epoch": best_epoch, "best_val_loss": best_loss if best_loss != float('inf') else None,
                "final_mae": final_metrics.get('mae'), "final_rmse": final_metrics.get('rmse')
            }
            summary_path = output_dir / f"summary_{script_identifier}.json"
            try:
                with open(summary_path, 'w') as f: json.dump(summary, f, indent=4)
                print(f"Summary saved to {summary_path}")
            except Exception as e: print(f"Error saving summary JSON: {e}")

        except Exception as e: print(f"Error loading best model or during evaluation: {e}")
    else:
        print(f"Could not load best model from {checkpoint_path}. Evaluation skipped.")
        if not checkpoint_path.is_file(): print("(Checkpoint file not found)")
        if best_epoch == -1: print("(Best epoch not reached or training error)")

    print(f"\n--- Script {script_identifier} Finished ---")


if __name__ == "__main__":
    start_time = timer.time()
    main()
    end_time = timer.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")