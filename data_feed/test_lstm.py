import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

WINDOW_SIZE = 30

# Load saved scaler parameters
scaler_data = np.load("../model/scaler_params.npy", allow_pickle=True).item()
scaler = MinMaxScaler()
scaler.min_ = scaler_data["min"]
scaler.data_min_ = scaler_data["min"]
scaler.data_max_ = scaler_data["max"]
scaler.scale_ = 1 / (scaler.data_max_ - scaler.data_min_ + 1e-8)
scaler.data_range_ = scaler.data_max_ - scaler.data_min_

# Define model
class BTC_LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return self.sigmoid(out)

# Load model
model = BTC_LSTM()
model.load_state_dict(torch.load("../model/lstm_btc_updown.pt"))
model.eval()

# === Example synthetic test prices ===
# Replace with real price history ideally
np.random.seed(7)
test_prices = np.cumsum(np.random.randn(200)) + 20000

# === Generate test sequences and labels ===
def generate_test_data(prices, window_size=30):
    sequences = []
    labels = []
    for i in range(len(prices) - window_size - 1):
        window = prices[i:i + window_size]
        label = 1 if prices[i + window_size] > prices[i + window_size - 1] else 0
        sequences.append(window)
        labels.append(label)
    return sequences, labels

sequences, labels = generate_test_data(test_prices)

# === Evaluate model ===
correct = 0
total = len(sequences)

for i in range(total):
    input_array = np.array(sequences[i]).reshape(1, -1)
    input_scaled = scaler.transform(input_array).reshape(1, WINDOW_SIZE, 1)
    x_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        prob_up = model(x_tensor).item()
    
    predicted = 1 if prob_up >= 0.5 else 0
    if predicted == labels[i]:
        correct += 1

accuracy = correct / total
error_rate = 1 - accuracy

print(f"✅ Tested on {total} sequences")
print(f"✅ Accuracy: {accuracy:.2%}")
print(f"❌ Error Rate: {error_rate:.2%}")
