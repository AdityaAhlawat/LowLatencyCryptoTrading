import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import os

# Create output directory
os.makedirs("../model", exist_ok=True)

WINDOW_SIZE = 150

# Simulated BTC price data (replace with historical for real model)
np.random.seed(42)
sample_prices = np.cumsum(np.random.randn(1000)) + 20000

def generate_dataset(prices, window_size=30):
    X, y = [], []
    for i in range(len(prices) - window_size - 1):
        window = prices[i:i + window_size]
        label = 1 if prices[i + window_size] > prices[i + window_size - 1] else 0
        X.append(window)
        y.append(label)
    return np.array(X), np.array(y)

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

# Prepare data
X, y = generate_dataset(sample_prices, WINDOW_SIZE)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Save scaler
np.save("../model/scaler_params.npy", {"min": scaler.data_min_, "max": scaler.data_max_})

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train model
model = BTC_LSTM()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for batch_x, batch_y in loader:
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "../model/lstm_btc_updown.pt")
print("✅ Model and scaler saved to /model/")
