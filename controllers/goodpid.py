from . import BaseController
import torch

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

import deep_learning_models

# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x, h0=None, c0=None):
#         # If initial hidden and cell states are not provided, initialize them
#         if h0 is None:
#             h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
#         if c0 is None:
#             c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

#         if x.dim() == 1:
#             out, (hn, cn) = self.lstm(x.unsqueeze(1), (h0, c0))
#             out = self.fc(out.squeeze(1))
#         else:
#             out, (hn, cn) = self.lstm(x, (h0, c0))
#             out = self.fc(out)
#         return out, (hn, cn)
    
# lstm_model = LSTM(3, 50, 2, 1)
# lstm_model.load_state_dict(torch.load('correction_controller.pth', map_location=torch.device('cpu')))

ROLL_LATA_MEAN = 0.12531
ROLL_LATA_STD = 0.264393
V_EGO_MEAN = 23.2602
V_EGO_STD = 9.40597
A_EGO_MEAN = 0.027936
A_EGO_STD = 0.4448
LAT_ACC_MEAN = 0.0125555
LAT_ACC_STD = 0.567619 

class Controller(BaseController):
  def __init__(self, kp=0.08852291, ki=0.07724999, kd=-0.05190457):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.lstm_model = deep_learning_models.lstm_model
    self.last_err = 0
    self.int = 0
    
  def update(self, target_lataccel, current_lataccel, state):
    curr_err = target_lataccel - current_lataccel
    deriv = curr_err - self.prev_err
    self.int += curr_err
    self.last_err = curr_err
    
    output = self.kp * curr_err + self.ki * self.int + self.kd * deriv
    
    input_tensor = torch.tensor([[
            (state[0] - ROLL_LATA_MEAN)/ROLL_LATA_STD,
            (state[1] - V_EGO_MEAN)/V_EGO_STD,
            (state[2] - A_EGO_MEAN)/A_EGO_STD,
        ]], dtype=torch.float32)

    #Forward pass through LSTM model
    with torch.no_grad():
        lstm_output, _ = self.lstm_model(input_tensor.unsqueeze(0))

    predicted_output = lstm_output.item()

    output += 1 * predicted_output
    output = float(output)
    
    return output
