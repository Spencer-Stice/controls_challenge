import torch
import deep_learning_models
ROLL_LATA_MEAN = 0.12531
ROLL_LATA_STD = 0.264393
V_EGO_MEAN = 23.2602
V_EGO_STD = 9.40597
A_EGO_MEAN = 0.027936
A_EGO_STD = 0.4448
LAT_ACC_MEAN = 0.0125555
LAT_ACC_STD = 0.567619 



class BaseController:
  def update(self, target_lataccel, current_lataccel, state):
    raise NotImplementedError


class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return target_lataccel


class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return (target_lataccel - current_lataccel) * 0.3

class LSTMController(BaseController):
    def __init__(self):
        self.lstm_model = deep_learning_models.lstm_model

    def update(self, target_lataccel, current_lataccel, state):
        # Convert inputs to tensors

        # NEED TO NORMALIZE THESE -----------------------------------
        input_tensor = torch.tensor([[
            (state[0] - ROLL_LATA_MEAN)/ROLL_LATA_STD,
            (state[1] - V_EGO_MEAN)/V_EGO_STD,
            (state[2] - A_EGO_MEAN)/A_EGO_STD,
            (current_lataccel - LAT_ACC_MEAN)/LAT_ACC_STD,
            (target_lataccel - LAT_ACC_MEAN)/LAT_ACC_STD
        ]], dtype=torch.float32)  # Assuming 'device' is properly defined

        # Forward pass through LSTM model
        with torch.no_grad():
            output, _ = self.lstm_model(input_tensor.unsqueeze(0))

        # Extract the predicted output
        predicted_output = output.item()

        return predicted_output

class PIDController(BaseController):
  def __init__(self, Kp = 0.25, Kd = 0.0):
    self.Kp = Kp
    self.Kd = Kd
    self.previous_error = 0

  def compute(self, error):
      # Proportional term
      P = self.Kp * error
      
      # Derivative term
      D = self.Kd * (error - self.previous_error)
      
      # Save the current error as the previous error for the next time step
      self.previous_error = error
      
      # Compute the PID output
      output = P + D
      
      return output

  def update(self, target_lataccel, current_lataccel, state):
    return self.compute(target_lataccel - (current_lataccel + state[0]))
  
class GoodPIDController(BaseController):
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


CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'lstm': LSTMController,
  'pid': PIDController,
  'gpid': GoodPIDController
}
