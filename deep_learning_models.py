import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None, c0=None):
        # If initial hidden and cell states are not provided, initialize them
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        if x.dim() == 1:
            out, (hn, cn) = self.lstm(x.unsqueeze(1), (h0, c0))
            out = self.fc(out.squeeze(1))
        else:
            out, (hn, cn) = self.lstm(x, (h0, c0))
            out = self.fc(out)
        return out, (hn, cn)
    
lstm_model = LSTM(3, 150, 3, 1)
lstm_model.load_state_dict(torch.load('lstm_correction_models/correction_controller_500.pth', map_location=torch.device('cpu')))
lstm_model_100 = LSTM(3, 100, 3, 1)
lstm_model_100.load_state_dict(torch.load('lstm_correction_models/correction_controller_100.pth', map_location=torch.device('cpu')))