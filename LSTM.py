import torch
import torch.nn as nn

# Instantiate the model with Bidirectional LSTM
class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(ImprovedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional

    def forward(self, x):
        # Forward propagate through LSTM
        out, _ = self.lstm(x)
        # Pass the last time step output to the fully connected layer
        out = self.fc(out[:, -1, :])
        return out

