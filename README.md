# Athens Stock Exchange Price Forecasting with RNN and GRU

A deep learning project for **multi-step stock price forecasting** using historical data from the Athens Stock Exchange (ASE).

The system uses **5 trading days of historical data** to predict closing prices for the following **5 trading days**, comparing a Vanilla Recurrent Neural Network (RNN) with a Gated Recurrent Unit (GRU) architecture.

---

## Project Overview

This project implements an end-to-end time-series forecasting pipeline for stock price prediction.

Two recurrent neural network architectures are implemented and compared:

- **Vanilla RNN** — an autoregressive recurrent model used as the baseline.
- **GRU (Gated Recurrent Unit)** — a deeper recurrent architecture designed to capture temporal dependencies more effectively.

The pipeline includes data preprocessing, feature scaling, day-of-week encoding, stock-specific sliding-window sequence generation, model training with early stopping, and evaluation on unseen test data.

---

## Forecasting Task

The forecasting problem is formulated as:

**5 historical trading days → 5 future closing prices**

Sliding-window sequences are generated independently for each stock to ensure that sequences never cross stock boundaries.

The target variable is:

`Close`

---

## Data Preprocessing

The preprocessing pipeline includes:

### Stock-wise Sequence Generation

Each CSV file represents an individual stock and receives a unique `Stock_ID`.

Sliding windows are then generated separately for each stock, preventing sequences from combining observations belonging to different securities.

### Day-of-Week Encoding

The trading date is converted into a categorical day-of-week feature and represented using one-hot encoding.

This provides the models with additional temporal information associated with the trading calendar.

### Feature Scaling

Numerical features are normalized using `MinMaxScaler`.

The scaler is fitted exclusively on the training data and subsequently applied to the validation and test sets.

A dedicated scaler is also maintained for the `Close` target variable so predictions can be transformed back to their original scale during evaluation.

---

## Models

### Vanilla RNN

The baseline model uses a recurrent neural network with:

- 64 hidden units
- ReLU activation
- autoregressive multi-step prediction
- Adam optimizer
- MSE loss

The model processes the historical 5-day sequence and predicts future closing prices recursively, feeding each predicted value back into the model when generating the next forecast.

### GRU

The second architecture uses a two-layer GRU network with:

- 128 hidden units
- 2 recurrent layers
- dropout of 0.2
- direct 5-day output
- Adam optimizer
- MSE loss

Unlike the autoregressive RNN, the GRU generates all five future predictions directly from the final hidden representation.

---

## Training

Both models are trained using **Mean Squared Error (MSE)** as the loss function and the **Adam optimizer**.

Early stopping monitors validation loss with a patience of **10 epochs**. The best-performing model weights are stored during training and restored before evaluation.

Default maximum training duration:

`100 epochs`

---

## Evaluation Metrics

Performance on the unseen test set is evaluated using:

- **MAE (Mean Absolute Error)** — average absolute difference between predicted and actual prices.
- **RMSE (Root Mean Squared Error)** — gives greater weight to larger forecasting errors.
- **MAPE (Mean Absolute Percentage Error)** — expresses prediction error relative to the true stock price.

Predictions are converted back to the original price scale before calculating these metrics.

---

## Project Structure

```text
stock-price-forecasting-rnn-gru/
│
├── main.py
├── README.md
├── .gitignore
└── LICENSE
```

The stock market dataset is not included in the repository.

The expected data directory has the following structure:

```text
Time_series_data/
│
├── train/
│   └── *.csv
├── validation/
│   └── *.csv
└── test/
    └── *.csv
```

---

## How to Run

The program provides a command-line interface using Python's `argparse`.

### Arguments

- `--data_dir` — path to the directory containing the training, validation, and test folders (**required**)
- `--model` — model to train: `rnn`, `gru`, or `both` (default: `both`)
- `--epochs` — maximum number of training epochs (default: `100`)

### Run Both Models

```bash
python main.py --data_dir ./Time_series_data --model both --epochs 100
```

### Run Only the Vanilla RNN

```bash
python main.py --data_dir ./Time_series_data --model rnn --epochs 100
```

### Run Only the GRU

```bash
python main.py --data_dir ./Time_series_data --model gru --epochs 100
```

---

## Technologies

- Python
- PyTorch
- pandas
- NumPy
- scikit-learn

---

## Key Implementation Features

- Multi-step **5-day stock price forecasting**
- Independent sequence generation for each stock
- Training-only fitting of feature scalers
- Day-of-week feature engineering
- Autoregressive Vanilla RNN forecasting
- Direct multi-output GRU forecasting
- Early stopping based on validation loss
- Evaluation on unseen stocks using MAE, RMSE, and MAPE
- Command-line interface for model selection and training configuration

---

## Author

**Anna Allagioti**  
**Maria Karkoglou**

---

## License

This project is licensed under the MIT License.
