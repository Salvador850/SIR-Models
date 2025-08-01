import sys
import pandas as pd
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar

# compile data from all files into data frame
data_frame  = None
if len(sys.argv) > 8:        
        for i in range(9, len(sys.argv)):
                
            data_file = open(sys.argv[i], 'r').read()
            target_list = eval(data_file)
            timestamp_list = [i for i in range(len(target_list))]
            item_id_list = [0 for i in range(len(target_list))]
            
            full_data = zip(item_id_list, timestamp_list, target_list)
            additional_data = pd.DataFrame(full_data, columns=["item_id", "timestamp", "target"])
            
            additional_data['target'] /= 4 * (10**6)
            if data_frame != None:
                data_frame = pd.concat([data_frame, additional_data], ignore_index=True)
            else:
                data_frame = additional_data
       

data_frame['target']  /= 30
data_frame = data_frame.drop([str(i) for i in range(len(data_lists[0][0]) - 3)], axis=1)

# SIR model equations
def SIRS_model(y, t, beta, gamma, theta, b_rate, d_rate):
    S, I, R = y
    total = S + I + R
    dSdt = (b_rate * total) - ((beta * S * I) / (S + I + R)) + (theta * R) - (d_rate * S) 
    dIdt = ((beta * S * I) / (S + I + R))  - (gamma * I) - (d_rate * I)
    dRdt = gamma * I - (theta * R) - (d_rate * R)
    return [dSdt, dIdt, dRdt]

# Set initoal vars
y0 = [float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])]

beta = float(sys.argv[4])
gamma = float(sys.argv[5])
theta = float(sys.argv[6])

b_rate = float(sys.argv[7])
d_rate = float(sys.argv[8])

time_series_len = len(data_frame)
t = np.linspace(0, time_series_len, time_series_len)
mse = []

# Solve ODE, call targets, convert to list
solution = odeint(SIRS_model, y0, t, args=(beta, gamma, theta, b_rate, d_rate))
S, I, R = solution.T
new_I = [0] + [(beta * s * i) / (s + i + r) for s, i, r in zip(S, I, R)]
new_I = [i / int(sys.argv[9]) for i in new_I]
data_lists = list(data_frame['target'])

# Get MSE for each timestep
time_series = 0
for data_point in range(time_series_len):
    if data_point % time_series_length == 0 and data_point != 0:
            mse.append((data_lists[data_point - time_series] - new_I[data_point - time_series])**2)
    

# Get MSE every 50 timesteps to observe variation
frame_mse = []           
for i in range(19):
   frame = mse[i*50:][:(i + 1) * 50]
   frame_mse.append(sum(frame)/len(frame))

print("Frame Error:", frame_mse)
print("Error:", sum(mse)/len(mse))
