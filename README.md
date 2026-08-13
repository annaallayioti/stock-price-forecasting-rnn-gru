# Athens Stock Exchange Price Forecasting with RNN and GRU

Multi-step stock price forecasting on Athens Stock Exchange data: given 5 trading days of history, predict the closing price for the next 5 trading days.

Two recurrent architectures are compared — a Vanilla RNN that forecasts autoregressively and a GRU that predicts all five days at once — to explore how the forecasting strategy affects multi-step predictions.

MSc coursework project.

---

## The Two Approaches

The models differ mainly in how they produce the 5-day forecast.

### Vanilla RNN — Autoregressive

The Vanilla RNN uses 64 hidden units with ReLU activation. It processes the 5-day input window and then predicts one day at a time.

After each prediction, the predicted closing price is written back into the `Close` position of the input before predicting the next day.

This allows each forecast step to depend on the previous prediction, but it also means that errors made early in the forecast can propagate to later days.

### GRU — Direct Multi-Output

The GRU uses two recurrent layers with 128 hidden units and dropout of 0.2.

Instead of predicting recursively, the final hidden state is mapped directly to the five future closing prices through a linear layer.

This avoids feeding predictions back into the network and therefore avoids direct error propagation between forecast steps.

Both models use MSE loss, the Adam optimizer, and early stopping based on validation loss with `patience=10`. The best model weights are restored before evaluation.

---

## Data Handling

Each CSV file represents one stock and is assigned a `Stock_ID` when loaded.

### Sequences are created separately for each stock

Sliding windows are generated independently for every `Stock_ID` after sorting observations by date.

This prevents sequences from accidentally crossing from one stock into another when the individual CSV files are combined into a single DataFrame.

### Scaling

`MinMaxScaler` is fitted only on the training data and then applied to the validation and test sets.

A separate scaler is fitted to the `Close` column so that model predictions can be converted back to their original price scale before evaluation.

### Day-of-week feature

The trading date is converted into a day-of-week feature and one-hot encoded before training.

Each model receives:

- **Input:** 5 trading days
- **Output:** next 5 closing prices

---

## Evaluation

The models are evaluated on the held-out test set using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

Metrics are calculated after converting the predictions back to the original price scale.

The implementation prints the evaluation results directly to the terminal after training.

---

## Running the Project

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run both models:

```bash
python main.py --data_dir ./Time_series_data --model both --epochs 100
```

Run only the Vanilla RNN:

```bash
python main.py --data_dir ./Time_series_data --model rnn
```

Run only the GRU:

```bash
python main.py --data_dir ./Time_series_data --model gru
```

### Command-Line Arguments

| Argument | Description | Default |
|---|---|---|
| `--data_dir` | Root folder containing `train/`, `validation/`, and `test/` | required |
| `--model` | Model to train: `rnn`, `gru`, or `both` | `both` |
| `--epochs` | Maximum number of training epochs | `100` |

---

## Expected Data Layout

The dataset is not included in this repository.

```text
Time_series_data/
├── train/
│   └── *.csv
├── validation/
│   └── *.csv
└── test/
    └── *.csv
```

Each CSV must contain at least:

- `Date`
- `Close`

The remaining numeric columns are used as input features.

---

## Limitations

### Full-batch training

Training is currently performed without mini-batching. Each epoch therefore performs one optimizer update over the complete training set.

Introducing mini-batches would be one of the first improvements to the current implementation.

### Scaling across different stocks

The `MinMaxScaler` is fitted on the training data. Stocks in the test set whose prices fall outside the range observed during training may therefore produce scaled values outside the range seen by the models.

Per-stock normalization or forecasting returns instead of absolute prices could be explored as alternatives.

### Autoregressive feature updates

During autoregressive forecasting, only the predicted `Close` value is updated between forecast steps.

The remaining features stay fixed at their last observed values, which is a simplification of the real forecasting problem.

### Reproducibility

No fixed random seed is currently used, so model initialization and results can vary slightly between runs.

### One-hot encoding

Day-of-week encoding is performed separately for each split. If one split does not contain a particular weekday, the resulting feature columns may not align.

### Naive forecasting baseline

Stock prices can behave similarly to a random walk over short horizons. A useful extension would therefore be to compare both neural models against a naive baseline that predicts future prices from the most recently observed closing price.

---

## Tech Stack

- Python
- PyTorch
- pandas
- NumPy
- scikit-learn
- Matplotlib
- seaborn

---

## Repository Structure

```text
stock-price-forecasting-rnn-gru/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## Authors

**Anna Allagioti**  
**Maria Karkoglou**

## License

MIT

## License

This project is licensed under the MIT License.
