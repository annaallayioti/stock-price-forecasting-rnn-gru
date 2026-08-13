import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim

def load_and_combine(folder_path):
    """Loading and combing all CSV in one folder."""
    file_pattern = os.path.join(folder_path, "*.csv")
    file_list = glob.glob(file_pattern)
    df_list = []

    for file in file_list:
        df = pd.read_csv(file)
        stock_id = os.path.basename(file).replace('.csv', '')
        df['Stock_ID'] = stock_id
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


def create_sequences_per_stock(df, feature_cols, target_col, input_width=5, label_width=5):
    """Creating 3D sliding window sequences per stoch to prevent data leakage."""
    X, y = [], []
    for stock_id, group in df.groupby('Stock_ID'):
        group = group.sort_values('Date')
        features_matrix = group[feature_cols].values
        target_array = group[target_col].values

        for i in range(len(features_matrix) - input_width - label_width + 1):
            X.append(features_matrix[i: i + input_width])
            y.append(target_array[i + input_width: i + input_width + label_width])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)[..., np.newaxis]


def preprocess_data(data_dir):
    """Data preprocesing (Scaling, One-hot encoding)."""
    train_df = load_and_combine(os.path.join(data_dir, 'train'))
    val_df = load_and_combine(os.path.join(data_dir, 'validation'))
    test_df = load_and_combine(os.path.join(data_dir, 'test'))

    for df in [train_df, val_df, test_df]:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Day_of_Week'] = df['Date'].dt.day_name()

    # One-hot encoding
    train_df = pd.get_dummies(train_df, columns=['Day_of_Week'], drop_first=False, dtype=int)
    val_df = pd.get_dummies(val_df, columns=['Day_of_Week'], drop_first=False, dtype=int)
    test_df = pd.get_dummies(test_df, columns=['Day_of_Week'], drop_first=False, dtype=int)

    # Scaling
    feature_cols = train_df.columns.drop(['Date', 'Stock_ID'])
    target_col = 'Close'

    scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    target_scaler.fit(train_df[['Close']])

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    INPUT_DAYS = 5
    PREDICT_DAYS = 5

    X_train, y_train = create_sequences_per_stock(train_df, feature_cols, target_col, INPUT_DAYS, PREDICT_DAYS)
    X_val, y_val = create_sequences_per_stock(val_df, feature_cols, target_col, INPUT_DAYS, PREDICT_DAYS)
    X_test, y_test = create_sequences_per_stock(test_df, feature_cols, target_col, INPUT_DAYS, PREDICT_DAYS)

    return (X_train, y_train, X_val, y_val, X_test, y_test), target_scaler, PREDICT_DAYS, feature_cols


class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VanillaRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, predict_days, target_idx):
        # Process the 5-day historical sequence to build the initial hidden state
        _, hidden = self.rnn(x)

        # Extract the last timestep to use as the starting point for predictions
        current_input = x[:, -1:, :].clone()

        predictions = []

        # Autoregressive loop
        for _ in range(predict_days):
            # Step forward one day
            out, hidden = self.rnn(current_input, hidden)
            pred = self.fc(out)  # Shape: (Batch, 1, 1)
            predictions.append(pred)

            # Feed the prediction back into the input for the next step
            current_input = current_input.clone()
            current_input[:, 0, target_idx] = pred[:, 0, 0]

        # Concatenate predictions along the sequence dimension
        return torch.cat(predictions, dim=1)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_days):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_days)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        out = self.fc(last_hidden)
        out = out.unsqueeze(-1)
        return out

def train_and_evaluate_rnn(data, target_scaler, predict_days, epochs, input_size, target_idx):
    X_train, y_train, X_val, y_val, X_test, y_test = data

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)
    X_test_t = torch.tensor(X_test)

    model = VanillaRNN(input_size=input_size, hidden_size=64)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    patience = 10
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    print("Training Vanilla RNN with Early Stopping ---")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        train_outputs = model(X_train_t, predict_days, target_idx)
        train_loss = criterion(train_outputs, y_train_t)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t, predict_days, target_idx)
            val_loss = criterion(val_outputs, y_val_t)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load the best performing model
    model.load_state_dict(best_model_state)

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_test_t, predict_days, target_idx)
        pred_reshaped = predictions_scaled.view(-1, 1).numpy()
        real_prices = target_scaler.inverse_transform(pred_reshaped)
        final_predictions = real_prices.reshape(-1, predict_days)

    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = final_predictions.flatten()

    print("\nVanilla RNN Results")
    print(f"MAE  : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"MAPE : {np.mean(np.abs((y_true - y_pred) / y_true)) * 100:.2f}%\n")


def train_and_evaluate_gru(data, target_scaler, predict_days, epochs, input_size):
    X_train, y_train, X_val, y_val, X_test, y_test = data

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)
    X_test_t = torch.tensor(X_test)

    gru_model = GRUModel(input_size=input_size, hidden_size=128, output_days=predict_days)
    gru_criterion = nn.MSELoss()
    gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.0005)

    patience = 10
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    print("Training GRU with Early Stopping ---")
    for epoch in range(epochs):
        gru_model.train()
        gru_optimizer.zero_grad()
        train_outputs = gru_model(X_train_t)
        train_loss = gru_criterion(train_outputs, y_train_t)
        train_loss.backward()
        gru_optimizer.step()

        gru_model.eval()
        with torch.no_grad():
            val_outputs = gru_model(X_val_t)
            val_loss = gru_criterion(val_outputs, y_val_t)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = gru_model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    gru_model.load_state_dict(best_model_state)

    # Evaluation
    gru_model.eval()
    with torch.no_grad():
        gru_predictions_scaled = gru_model(X_test_t)
        gru_pred_reshaped = gru_predictions_scaled.view(-1, 1).numpy()
        gru_real_prices = target_scaler.inverse_transform(gru_pred_reshaped)
        gru_final_predictions = gru_real_prices.reshape(-1, predict_days)

    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = gru_final_predictions.flatten()

    print("\n--- GRU Results ---")
    print(f"MAE  : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"MAPE : {np.mean(np.abs((y_true - y_pred) / y_true)) * 100:.2f}%\n")

def main():
    parser = argparse.ArgumentParser(description="Athens Stock Exchange - Stock Price Predictor")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='The central directory that contains the train, validation, and test folders.')
    parser.add_argument('--model', type=str, choices=['rnn', 'gru', 'both'], default='both',
                        help='Model choice to train.')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum epoch number.')

    args = parser.parse_args()

    print("Loading and preprocessing data...")
    data, target_scaler, predict_days, feature_cols = preprocess_data(args.data_dir)

    input_size = data[0].shape[2]
    target_idx = list(feature_cols).index('Close')

    if args.model in ['rnn', 'both']:
        train_and_evaluate_rnn(data, target_scaler, predict_days, args.epochs, input_size, target_idx)

    if args.model in ['gru', 'both']:
        train_and_evaluate_gru(data, target_scaler, predict_days, args.epochs, input_size)


if __name__ == '__main__':
    main()